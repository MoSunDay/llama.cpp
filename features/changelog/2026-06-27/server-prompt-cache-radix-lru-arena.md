Commit: 3fc4e105279105106b08a133a4e3e483116e621f (含本地未提交改动)

# server prompt cache 改造为 radix 前缀树 + 真 LRU + 无碎片 arena

## Context

`llama-server` 的 RAM 提示缓存 `server_prompt_cache`（`--cache-ram`，跨请求 KV 复用以避免重复 prefill）原实现有三个问题：
- 匹配为 O(条目数 × 长度) 的线性 LCP 扫描（`get_common_prefix`），高并发下随缓存条目增长而退化。
- 淘汰为 FIFO（`states.pop_front()`），`load()` 命中即消费（`states.erase` + `std::move`），非真 LRU，无法跨请求共享同一份前缀。
- 每条目用独立 `std::vector<uint8_t>` 存序列化 KV，反复 `resize/clear/shrink_to_fit` 造成堆碎片。

需求：配置最大 KV cache 大小（复用 `--cache-ram`）、真 LRU 过期、无内存碎片、匹配用 radix 前缀树。

## Change Summary

新增两个自包含组件，并重构 `server_prompt_cache` 内部（公共 API `alloc/load/update/size/n_tokens` 与构造签名保持兼容，server 层调用点几乎不动）：

- **`tools/server/server-cache-arena.h` — `server_kv_arena`**：单个连续 host buffer（`new[]` 大块走 glibc mmap，按页延迟提交，RSS 按需增长、上限为容量）+ best-fit free-list + 邻块合并。镜像 `ggml_dyn_tallocr`（`ggml-alloc.c`）但用于裸 host 字节、不耦合 tensor。`alloc` 先查 free-list 再切 committed tail；`free` 回收并合并。注意：`resize` **不**预填 free-list（free-list 与零基 tail 重叠会导致 offset 0 重复分配——已修复的 bug）。
- **`tools/server/server-cache-radix.{h,cpp}` — `server_radix_index`**：逐 token 的 trie，沿 entry 每个节点做 refcount（`n_entries`）并在终点标 entry。`longest_prefix` 走查询 token 到分叉处得 LCP 深度，再在子树 bounded DFS（`kDfsBudget=256`）按 `f_keep≥0.25` 取最优 entry。O(命中前缀长)。语义等价旧线性扫描但更快；支持"entry 比共享前缀长"的常见场景（共享系统前缀）。
- **`tools/server/server-task.{h,cpp}`**：
  - `server_prompt_data`：`std::vector<uint8_t> main/drft` → 非拥有 arena 切片（`uint8_t* + size`），消除 per-entry 堆分配。
  - `server_prompt`：加 `uint64_t last_access`（LRU 访问时间戳）。
  - `server_prompt_cache`：持有 `arena_` + `radix_` + `tick_`；构造按 `--cache-ram` 预留 arena（`-1`→16GB 虚拟上限，lazy commit；严格 overcommit 下 `resize` 减半重试到 256MB 下限）。`alloc` 用 arena 切片（满则 `evict_one` LRU 重试）；`load` **读后保留**（真 LRU，blob 只读共享，多请求复用同一前缀），命中刷新 `last_access`；`update`/`evict_one` 按 `last_access` 最小者淘汰并 `release_blob`（arena free + radix erase）。

## Impact Surface

- `tools/server/server-context.cpp`：`prompt_save` 中 `cur->data.main.data()` → `cur->data.main`（现为裸指针）。
- `tools/server/CMakeLists.txt`：新增 `server-cache-radix.cpp` 到 `server-context` 静态库。
- 行为：缓存默认仍由 `--cache-ram` 开关（`0`=禁用，与旧一致）。`cache_prompt=false`（per-request）仍禁用前缀复用。
- 已知限制（未在本工作流修复）：hybrid/recurrent 模型 `common_context_can_seq_rm()==FULL`（见 `common/common.cpp:1459`、PR #13194），slot 前缀处理后的 `common_context_seq_rm`（`server-context.cpp:3379`）无法部分截断 → 退化全量重 prefill。复用量因此受限于 checkpoint 恢复点（实测复用 ~699/~1215 tok）。彻底修复需核心层 block 共享（`seq_cp`，见 Workstream C）。

## 验证（node02，Qwen3.5-9B-Q4_K_M，RTX 4060 Ti）

- 编译：fork（commit 3fc4e10）CUDA 构建 llama-server 成功（sm_89）。
- radix 跨请求复用：共享前缀 + 不同后缀，`--cache-idle-slots`（slot 清空），req2/3/4 复用 699 tok、prefill 1215→516；cache OFF 全量 1215。
- LRU + arena：`--cache-ram 300`，25 个不同 prompt → 17 次 LRU 淘汰，arena `used` bound 在 ~117 MiB ≤ 300（修复前会增长到 410+ MiB 失控）。
- 真 LRU：`load` 读后保留，多请求共享同一缓存条目。
- 4 路并发：共享前缀两波，wave2 复用 2832 tok、prefill 4896→2064（降 58%）。
- 单元测试 `tests/test-cache-arena.cpp`（`server_kv_arena`）：1000 次 alloc churn 下 `used` 稳定 ≤ capacity、无重叠、合并正确、clear/oversize 正确——全通过。
- 复跑验证（当前二进制，默认配置 `--cache-ram 1024 --no-cache-idle-slots`）：req1 冷启 cached=0；req2/req3 共享系统前缀 cached=456/473（96% 复用），且 req2/req3 命中数相同 = 读后保留（真 LRU 共享 blob）生效；slot `n_prompt_tokens_processed=17`（=473−456）。回归为零、复用正常。

### 修复记录
- arena `resize` bug：最初同时预填 free-list `[{0,cap}]` 且 `committed_=0`，二者重叠导致 offset 0 被重复分配、`used` 失控增长。改为 free-list 初始为空、alloc 一律从 committed tail 切、free 回收到 free-list。另修 `resize` 的 256MiB 下限导致小容量（单测用 1024 字节）不分配的问题：改为先直接尝试请求容量、失败才减半。

## Notes / Compatibility

- 公共 API 未改（`server_prompt_cache` 接口不变）；改动集中在 server 层，未动核心 `src/llama-kv-cache.*`。
- arena 大容量预留依赖 Linux overcommit（默认 0/1）；严格 overcommit(=2) 下 `resize` 减半降级。
- 相关未完成：Workstream A（truncate 全量重 prefill）与 C（核心层 block 共享 + LRU evict hook + pin + `include/llama.h` 公共 API）未实施，二者交织——彻底提升 hybrid 模型复用率需 C。

## Workstream C（核心层 LRU 淘汰基座，加法/默认关闭）

为跨序列 LRU 淘汰打地基，按"加法 + 默认关闭 + 可测"的安全方式落地：

- `src/llama-kv-cells.h`/`llama-kv-cache.{h,cpp}`：`llama_kv_cache` 加 per-seq `seq_last_use_[]` + `lru_tick_`，`apply_ubatch` 中 `seq_add` 时 `seq_touch`；新增只读 `get_lru_evictable_seq(keep, n_keep)`（返回有 cells 且不在 `keep` 里的最久未用 seq，无则 -1）。**加法、不改变分配/淘汰行为。**
- `src/llama-memory.h`：`llama_memory_i` 加虚方法 `get_lru_evictable_seq`（默认 -1）。
- `src/llama-kv-cache-iswa.{h,cpp}`：override 委托到 `kv_base`。
- `include/llama.h` + `src/llama-context.cpp`：公共 API `llama_memory_get_lru_seq(mem, keep, n_keep)`。
- `common/common.h` + `common/arg.cpp`：`--kv-evict-lru`/`--no-kv-evict-lru`（默认关）+ 环境变量 `LLAMA_ARG_KV_EVICT_LRU`。
- `tools/server/server-context.cpp`：`get_available_slot` 内、gated 打开时查询 LRU 候选（当前为可观测日志；实际 idle 清理仍由 `--cache-idle-slots` 驱动）。

### 设计正确性
- 纯 attention 模型（`llama_kv_cache` / `llama_kv_cache_iswa`）：API 返回有效 LRU seq（已 override）。
- hybrid/recurrent 模型（`llama_memory_hybrid`）：API 返回 -1 —— **这是正确且安全的行为**，recurrent 记忆不可中途淘汰，故对 hybrid 不提供候选，避免误淘汰破坏状态。本仓测试模型 Qwen3.5-9B 即 hybrid，故观测到 -1。
- 默认关闭 → 现有行为 100% 不变；回归（共享前缀复用 1428→516、4 路并发 wave2 降 58%）与 C 前完全一致。

### 后续（达到完整生产级 C 的剩余工作）
在 hybrid 模型上获得 block 级复用需：核心 `llama_memory_hybrid` 的 attention/recurrent 解耦（可单独淘汰 attention cell 而保持 recurrent）+ server 端把 `--kv-evict-lru` 从"观测"接为"实际淘汰调用"（`get_available_slot`/ctx_shift 路径，保护 active slot）。这是独立的较大改造，需跨 dense/hybrid/SWA 的回归矩阵。
