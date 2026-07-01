Commit: 3fc4e105279105106b08a133a4e3e483116e621f (含本地未提交改动)

# server-cache - llama-server RAM 提示缓存子系统

## 职责

为 `llama-server` 提供跨请求的 KV cache 复用（避免重复 prefill），由 `--cache-ram` 开关控制（`0` 禁用）。命中前缀时跳过该前缀的 prefill，降低高并发下的首 token 延迟与算力。

## 边界

- 仅缓存已序列化的 KV 状态（host RAM 字节），不缓存 token 文本。
- 仅在"加载时"匹配：请求到来时按 token 序列与缓存条目求最长公共前缀（LCP），命中部分从缓存恢复，剩余部分正常 prefill。
- 非目标：不负责 slot 调度、不做推测解码、不管理 GPU 显存。

## 关键抽象

三个自包含组件，公共 API 与 server 层调用点基本不变：

- **`server_kv_arena`**（`tools/server/server-cache-arena.h`）：单个连续 host buffer（`new[]` 大块走 glibc mmap，按页延迟提交，RSS 按需增长、上限为容量）+ best-fit free-list + 邻块合并。镜像 `ggml_dyn_tallocr`（`ggml-alloc.c`）但用于裸 host 字节、不耦合 tensor。`alloc` 先查 free-list 再切 committed tail；`free` 回收并合并。`resize` **不**预填 free-list（避免与零基 tail 重叠导致 offset 0 重复分配）。
- **`server_radix_index`**（`tools/server/server-cache-radix.h`，header-only 类模板 `server_radix_index_impl<Entry>`，生产别名 `server_radix_index = server_radix_index_impl<server_prompt*>`）：逐 token 的 trie，每个节点 refcount、终点标 entry。`longest_prefix` 走查询 token 到分叉处得 LCP 深度，再在子树 bounded DFS（`kDfsBudget=256`）按 `f_keep≥0.25` 取最优 entry。O(命中前缀长)。支持"entry 比共享前缀长"的常见场景（共享系统前缀）。模板化以便脱离 `server_tokens`/`server_prompt` 单测。
- **`server_prompt_cache`**（`tools/server/server-task.{h,cpp}`）：持有 `arena_` + `radix_` + `tick_`，编排 alloc/load/update/evict。`server_prompt_data` 现为非拥有 arena 切片（`uint8_t* + size`），消除 per-entry 堆分配。

## 主流程

- `alloc(n)`：用 arena 切片（满则 `evict_one` 按 `last_access` 最小者淘汰并 `release_blob` 重试）。
- `load(tokens)`：radix `longest_prefix` 命中则 **读后保留**（blob 只读共享，多请求复用同一前缀），命中刷新 `last_access`。这是"真 LRU"语义，区别于旧的命中即消费（FIFO）。
- `update(id, tokens, blob)`：写回时 radix insert + arena alloc。

## 核心层依赖（Workstream C，加法/默认关闭）

per-seq LRU 淘汰基座，为未来跨序列淘汰打地基，当前不改变分配/淘汰行为：

- `llama_kv_cache`（`src/llama-kv-cache.{h,cpp}`）：per-seq `seq_last_use_[]` + `lru_tick_`，`apply_ubatch` 中 `seq_add` 时 `seq_touch`；只读 `get_lru_evictable_seq(keep, n_keep)` 返回有 cells 且不在 keep 里的最久未用 seq。
- `llama_memory_i`（`src/llama-memory.h`）：虚方法默认 -1；`llama_kv_cache_iswa` override 委托 `kv_base`。
- 公共 API `llama_memory_get_lru_seq(mem, keep, n_keep)`（`include/llama.h` + `src/llama-context.cpp`）。
- `--kv-evict-lru`（`common/common.h` + `common/arg.cpp`，默认关）。server 消费者在 `server-context.cpp` 的 `get_available_slot`（gated，当前为可观测日志）。

设计正确性：纯 attention 模型返回有效 LRU seq；hybrid/recurrent 模型（如本仓测试模型 Qwen3.5-9B，`llama_memory_hybrid`）返回 -1，因为 recurrent 记忆不可中途淘汰，避免误淘汰破坏状态。

## 接口与依赖

- 被 `server-context.cpp` 的 `prompt_save`/slot 处理调用。
- `--cache-ram` 控制总容量；`--cache-idle-slots`（默认开）会清空 idle slot KV 使 within-slot 复用失效，观测 radix/LRU 需 `--no-cache-idle-slots`。
- 大容量 arena 预留依赖 Linux overcommit；严格 overcommit(=2) 下 `resize` 减半降级。

## 测试

- `tests/test-cache-arena.cpp`：`server_kv_arena` 单元测试（basic/reuse/coalescing/oversize/clear），1000 次 alloc churn 下 `used` 稳定 ≤ capacity。
- `tests/test-cache-radix.cpp`：`server_radix_index_impl` 单元测试（exact/no-match、query 比 entry 长、共享前缀 entry 更长、最小候选选择、`f_keep≥0.25` 阈值、erase+refcount+prune、clear、空 entry、churn），用轻量 `test_entry` + `std::vector<llama_token>`，无 server/llama 链接。

## 相关文档

- [`features/server-prompt-cache/index.md`](../../features/server-prompt-cache/index.md) - 业务能力视图。
- [`features/changelog/2026-06-27/server-prompt-cache-radix-lru-arena.md`](../../features/changelog/2026-06-27/server-prompt-cache-radix-lru-arena.md) - 改造记录。
