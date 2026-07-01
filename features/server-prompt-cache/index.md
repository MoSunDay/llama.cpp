Commit: 3fc4e105279105106b08a133a4e3e483116e621f (含本地未提交改动)

# server 提示缓存复用

## 能力

`llama-server` 在请求到来时，按 token 序列与 RAM 缓存条目求最长公共前缀，命中部分从缓存恢复 KV、跳过该前缀的 prefill。降低高并发下共享系统前缀/公共前缀场景的首 token 延迟与算力。

由 `--cache-ram <MiB>` 开关控制（`0` 禁用，与上游一致）。per-request 可用 `cache_prompt`（boolean）关闭单次复用。

## 触发与行为

- 请求到来 → 沿 radix 前缀树求 LCP → 命中则读后保留（真 LRU，blob 只读共享，多请求复用同一前缀）→ 剩余部分正常 prefill。
- 缓存满（arena 容量到上限）→ 按 `last_access` 最久未用者淘汰（arena free + radix erase）。
- `--cache-idle-slots`（默认开）会清空 idle slot KV，使 within-slot 复用失效；观测/启用复用需 `--no-cache-idle-slots`。

## 关键状态

- 命中：响应 `usage.prompt_tokens_details.cached_tokens` > 0（如 473 prompt 中 456 命中 = 96% 复用，实测）。
- slot 处理：`n_prompt_tokens_processed` = 实际 prefill 数 = prompt_tokens − cached_tokens。

## 约束

- hybrid/recurrent 模型（`common_context_can_seq_rm()==FULL`，PR #13194）：slot 前缀处理后的部分截断不可行 → 退化全量重 prefill。复用量受限于 checkpoint 恢复点。彻底修复需核心层 block 共享（见核心层 LRU 淘汰基座，当前加法/默认关闭）。
- 大容量 arena 预留依赖 Linux overcommit（默认 0/1）；严格 overcommit(=2) 下减半降级。

## 相关逻辑

- [`agents/server-cache/index.md`](../../agents/server-cache/index.md) - 缓存子系统结构与主流程。
- [`changelog/2026-06-27/server-prompt-cache-radix-lru-arena.md`](../changelog/2026-06-27/server-prompt-cache-radix-lru-arena.md) - 改造记录。
