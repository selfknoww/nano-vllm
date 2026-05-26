from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size

        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


class Llama3RotaryEmbedding(nn.Module):
    """Llama-3 RoPE：按波长分高/中/低频段做差异化插值，支持 8k → 128k 长上下文。"""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        factor: float,
        low_freq_factor: float,
        high_freq_factor: float,
        original_max_position_embeddings: int,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size

        # 1. 算朴素 inv_freq
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        # 2. Llama-3 频段插值
        inv_freq = self._llama3_rescale(
            inv_freq, factor, low_freq_factor, high_freq_factor,
            original_max_position_embeddings,
        )
        # 3. 构造 cos_sin cache
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @staticmethod
    def _llama3_rescale(
        inv_freq: torch.Tensor,
        factor: float,
        low_freq_factor: float,
        high_freq_factor: float,
        old_ctx_len: int,
    ) -> torch.Tensor:
        # 按 transformers _compute_llama3_parameters 的公式
        low_freq_wavelen = old_ctx_len / low_freq_factor
        high_freq_wavelen = old_ctx_len / high_freq_factor
        wavelen = 2 * torch.pi / inv_freq

        # 低频段（波长 > low_freq_wavelen）：除以 factor 压缩到原范围
        inv_freq_scaled = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
        # 中间段：线性插值在"不缩"和"缩 factor"之间过渡，避免突变
        smooth = (old_ctx_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smoothed = (1 - smooth) * (inv_freq / factor) + smooth * inv_freq
        is_medium = (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen)
        return torch.where(is_medium, smoothed, inv_freq_scaled)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_type: str | None = None,
    factor: float | None = None,
    low_freq_factor: float | None = None,
    high_freq_factor: float | None = None,
    original_max_position_embeddings: int | None = None,
):
    if rope_type is None:
        return RotaryEmbedding(head_size, rotary_dim, max_position, base)
    if rope_type == "llama3":
        return Llama3RotaryEmbedding(
            head_size, rotary_dim, max_position, base,
            factor=factor,
            low_freq_factor=low_freq_factor,
            high_freq_factor=high_freq_factor,
            original_max_position_embeddings=original_max_position_embeddings,
        )
    raise ValueError(f"Unknown rope_type: {rope_type}")
