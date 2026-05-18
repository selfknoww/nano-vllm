# NanoVLLM 项目配置

## 技能自动调用

提问时按以下规则自动使用 skill：

- **superpowers:brainstorming**：涉及新增功能、修改行为、设计方案、重构等需要先明确意图的任务时调用。简单问答、概念解释、阅读代码等不用。
- **grill-me**：当我在推敲或质疑某个设计决策、或者说"帮我检查一下这个想法有没有问题"时调用，帮我逐层追问直到想清楚。

两个 skill 不会同时用——brainstorm 是先发散想方案，grill-me 是对已有方案做压力测试。

## 项目背景

NanoVLLM 是 vLLM 的极简复现，单一 Qwen3 模型，支持 chunked prefill、prefix cache、tensor parallelism、CUDA graph。代码约 1500 行，结构扁平。阅读或修改代码时默认不走重量级规划流程，直接看直接改。

## 笔记风格

代码注释和 docstring 用中文，自然叙述。
