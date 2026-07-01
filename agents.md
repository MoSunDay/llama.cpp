Commit: 3fc4e105279105106b08a133a4e3e483116e621f (含本地未提交改动)

# MoSunDay/llama.cpp fork - 逻辑结构图

本仓是 [llama.cpp](https://github.com/ggml-org/llama.cpp) 的私有 fork，基线 commit `3fc4e10`。上游不接收纯 AI 生成的 PR，私有 fork 豁免该限制。本地图只记录 fork 相对上游有意义的改动与结构。

## 模块索引

- [`agents/server-cache/index.md`](agents/server-cache/index.md) - llama-server 的 RAM 提示缓存子系统（`--cache-ram`），含无碎片 arena、radix 前缀树、真 LRU。

核心层（`src/`）的改动（per-seq LRU 淘汰基座，加法、默认关闭）见 server-cache 模块文档的"核心层依赖"一节。

## 能力地图

见 [`features/index.md`](features/index.md)。

## 上游约定

fork 的代码与提交规范遵循上游 `AGENTS.md`（ASCII、简洁注释、复用既有基础设施、大改动前确认）。本地记忆文档不遵循上游规范，按本仓库自身约定维护。
