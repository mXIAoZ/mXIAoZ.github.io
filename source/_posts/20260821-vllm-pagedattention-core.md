---
title: "vLLM 深度解析（二）：PagedAttention 核心机制"
date: 2026/08/21
tags:
  - vLLM
  - PagedAttention
  - FlashAttention
  - Prefix Caching
  - Chunked Prefill
  - Sampling
description: "从 Attention 的 IO 瓶颈出发，沿 vLLM v0.11.0 源码解释 PagedAttention、Prefix Caching、Chunked Prefill 与 Sampling。"
categories:
  - AI Infra
  - vLLM
mathjax: true
---

# vLLM 深度解析（二）：PagedAttention 核心机制

## 1. 引言

vLLM 的核心贡献之一，是把操作系统分页思想引入 KV Cache 管理：模型仍然看到逻辑连续的 token 序列，但这些 token 的 K/V 可以分散存放在不同物理块中，再由 `block_table` 恢复逻辑顺序。这套机制降低了连续预留造成的浪费，也为公共前缀共享和长 Prompt 分段执行提供了统一的地址基础。

<!--more-->

自回归生成每前进一步，都要读取历史 token 的 K/V，并把当前 token 的 K/V 追加到缓存。忽略量化、对齐和并行切分后，每个 token 的 KV Cache 大小近似为：

$$
\text{KV bytes/token}
=2\times N_{layers}\times N_{kv\_heads}\times d_{head}
\times \text{bytes(dtype)}
$$

例如，32 层、8 个 KV heads、`head_dim=128`、BF16 时，每个 token 约占 128 KiB；单个 32K 上下文约占 4 GiB。这个数值只对应上述模型形状，不能直接推广到其他模型。

若为每个请求预留一段最大长度的连续空间，会同时出现三类问题：

- **过度预留**：多数请求不会生成到长度上限；
- **外部碎片**：长短请求交错结束后，留下难以复用的连续空洞；
- **增长与共享困难**：输出长度未知，连续扩容昂贵，相同前缀也难以自然共享同一份 KV。

PagedAttention 解决的是 KV 的**存储、寻址和生命周期**；FlashAttention 解决的是 Attention 计算过程的**片上分块与 HBM 流量**。二者处在不同抽象层，可以同时使用。本文基于 vLLM `v0.11.0` 源码，依次讨论：

1. FlashAttention 为什么需要 online softmax；
2. PagedAttention 如何完成逻辑块到物理块的映射；
3. CUDA kernel 如何读取分页 K/V；
4. Prefix Caching、Chunked Prefill 如何复用这套地址机制；
5. Attention 结束后，不同 Sampling 策略如何影响性能。

> backend 选择、CUDA Graph 支持和 PagedAttention V1/V2 启发式都可能随版本变化。本文的源码路径与条件均限定在 vLLM `v0.11.0`。

## 2. FlashAttention：理解基础

### 2.1 为什么需要 FlashAttention？

多 token self-attention，特别是 Prefill，通常写成：

$$
S=QK^T,\qquad P=\operatorname{softmax}(S),\qquad O=PV
$$

如果先完整计算 `S`、写回 HBM，再读出计算 `P`；随后再次写回、读出 `P` 计算 `O`，两个 $n\times n$ 中间矩阵会产生大量 HBM 流量，临时显存也随序列长度按 $O(n^2)$ 增长。

这里必须区分 Prefill 与单 token Decode：Prefill 有多行 Query，完整 score 矩阵可能很大；Decode 通常只有一行 Query，单步 score 和 KV 读取随当前上下文近似按 $O(n)$ 增长。FlashAttention 的收益大小因此取决于 Query 长度、head size、dtype、GPU 和 kernel 实现，不能用固定倍率概括。

FlashAttention 的关键不是减少数学运算，而是使用 IO-aware tiling：

| 实现 | 完整 `S/P` 是否写入 HBM | 跨 tile 保留的状态 | 数学结果 |
|---|---|---|---|
| 直接实现 | 是 | 完整中间矩阵 | exact attention |
| FlashAttention | 否 | 每行的 `m`、`l` 与输出累加器 | exact attention |

它可能增加少量重计算，却显著减少大矩阵的 HBM 往返。计算吞吐与数据搬运速度的差距随硬件和 shape 改变，应以 profiler 中的带宽、算力利用率和 kernel 时间判断，而不是套用一个固定数字。

### 2.2 Online Softmax 算法

设已经处理过一部分 K/V tiles，并保存状态 $(m,l,A)$：

- `m`：当前行已经见过的最大 score；
- `l`：在最大值基准下的指数和；
- `A`：尚未除以 `l` 的输出累加器。

对新的 score tile $S_j$ 和 Value tile $V_j$，稳定合并公式为：

$$
m'=\max(m,\max S_j)
$$

$$
l'=e^{m-m'}l+\sum e^{S_j-m'}
$$

$$
A'=e^{m-m'}A+\sum e^{S_j-m'}V_j,
\qquad O=A'/l'
$$

每次更新都先把旧状态缩放到新的最大值基准，因此不会因为 score 很大而直接计算 `exp(score)`。遍历完全部 K/V tiles 后，得到的结果与普通 softmax 等价。

![FlashAttention Tile 流程](/images/vllm-pagedattention-core/01_flashattention_tile_flow.svg)

一个常见的逻辑映射是 `CTA(request, query_head, query_tile)`：一个请求会产生多个 CTA；一个 CTA 只能完整驻留在一个 SM；SM 又会动态执行来自不同请求的 CTA。request、CUDA block 和 SM 不是一一对应关系。

Tile 结束后，当前 `S/P/K/V` tile buffer 可以复用，但 HBM 中的 K/V 数据没有被删除。真正跨 tile 存活的是 Query tile 与 `(m,l,A)`。`Br/Bc` 还会同时受到以下因素限制：

- Q、K、V、score、输出累加器所需的片上容量；
- FP32 softmax 状态与输出累加器；
- 双缓冲、流水线和矩阵指令的对齐要求；
- register pressure、shared memory/VMEM bank 与 occupancy。

## 3. PagedAttention：虚拟内存管理

### 3.1 Block Table 机制

PagedAttention 把 KV Cache 切成固定 token 数的物理块，只在序列增长时按需分配。设 block size 为 `B`，请求内 token 位置为 `t`：

$$
logical\_block=\left\lfloor\frac{t}{B}\right\rfloor,
\qquad offset=t\bmod B
$$

$$
physical\_block=block\_table[request][logical\_block]
$$

在最简单的单 KV Cache group、单 DCP rank 情况下：

$$
slot=physical\_block\times B+offset
$$

![PagedAttention 地址翻译](/images/vllm-pagedattention-core/01_address_translation.svg)

图中 `B=16`、`block_table=[7,2,11]`。token 20 位于逻辑块 1、块内 offset 4，因此映射到物理块 2 的 offset 4，简化 slot 为 36。物理块 ID 不需要连续。

vLLM `v0.11.0` 中，写地址计算位于 `BlockTable.compute_slot_mapping`：

```python
block_table_indices = (
    req_indices * max_num_blocks_per_req
    + positions // block_size
)
block_numbers = block_table.ravel()[block_table_indices]
block_offsets = positions % block_size
slot_mapping = block_numbers * block_size + block_offsets
```

| 元数据 | 回答的问题 | 粒度 |
|---|---|---|
| `block_table` | 第 `i` 个逻辑块在哪个物理块？ | 按请求维护，随 KV block 分配增长 |
| `slot_mapping` | 本轮新 token 的 K/V 写入哪个 slot？ | 按本轮 flattened token batch 构造 |

简化成一句话：`slot_mapping` 负责写新增 K/V，`block_table` 负责读历史 K/V。

![单层 Attention 中的 KV 读写契约](/images/vllm-pagedattention-core/02_runtime_loop.svg)

执行链可以概括为：

```text
Scheduler.schedule
  → SchedulerOutput.num_scheduled_tokens
  → KVCacheManager.allocate_slots
  → GpuModelRunner._prepare_inputs
  → block_table / slot_mapping / query_start_loc / seq_lens
  → Attention backend
```

分页不会消除所有浪费：每个活跃序列的尾块仍可能未填满，Block Table 也需要内存。Block size 是尾块浪费、映射表大小与 kernel 访问效率之间的折中。

### 3.2 CUDA Kernel 实现细节

vLLM 可以选择 FlashAttention、FlashInfer、Triton 等 backend；下面分析的是 custom PagedAttention decode kernel。它非常适合观察 Block Table 如何进入 CUDA，但源码中存在这条路径，不代表任意 NVIDIA V1 配置都会默认使用它。

```text
PagedAttention.forward_decode
  → vllm._custom_ops.paged_attention_v1 / v2
  → torch.ops._C.paged_attention_v1 / v2
  → csrc/torch_bindings.cpp
  → paged_attention_v1.cu / paged_attention_v2.cu
  → attention_kernels.cuh
```

下面的代码按执行链截取，省略模板声明、部分外层循环和 Block Sparse 分支；有些片段从循环中间开始或结束，不能独立编译。每段后的说明负责补足数据布局、线程协作和缺失上下文。

#### 3.2.1 Q、K/V Cache 的物理布局

```  
	const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
```

K Cache 形状为 `[num_blocks, num_kv_heads, head_size/x, block_size, x]`，V Cache 为 `[num_blocks, num_kv_heads, head_size, block_size]`。其中：

- `x = 16 / sizeof(cache_t)`，K 最内层 `x` 个元素合计 16 bytes；
- thread group 根据 `scalar_t` 和 `THREAD_GROUP_SIZE` 决定计算向量 `VEC_SIZE`；
- `offset1/offset2` 把计算向量映射到 K Cache 的最内层布局；
- FP8 KV Cache 需要先加载量化向量，再按 scale 转换为计算向量。

K/V 排布不同来自 kernel 的访问模式：QK 点积更适合按 head dimension 向量化读取 K；`P·V` 则沿 token 权重累加每个输出维度。

#### 3.2.2 协作加载 Query

```
  const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  __shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];

#pragma unroll
  for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD;
       i += NUM_THREAD_GROUPS) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    q_vecs[thread_group_offset][i] =
        *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
  }
  __syncthreads();  // TODO(naed90): possible speedup if this is replaced with a
```

`q_ptr` 定位 `(sequence, query head)`。CTA 内多个 thread groups 分工填充同一份 shared `q_vecs`；组内线程通过 `thread_group_offset` 加载不同 head-dimension 片段，`__syncthreads()` 后由整个 CTA 重用。代码末尾停在一个未完整展示的 TODO 注释处，完整同步优化需查看链接源码。

#### 3.2.3 定位 Block Table

```
  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
```

`seq_idx * max_num_blocks_per_seq` 选择当前请求的 Block Table 行。逻辑 `block_idx` 到物理块的读取发生在后续循环：

```cpp
const int64_t physical_block_number =
    static_cast<int64_t>(block_table[block_idx]);
```

这就是 `block_table[request][logical_block] → physical_block` 在 CUDA 中的落点。

#### 3.2.4 分页读取 K 并计算 QK

```
      for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        const cache_t* k_ptr =
            k_cache + physical_block_number * kv_block_stride +
            kv_head_idx * kv_head_stride + physical_block_offset * x;
        const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
        const int offset1 = (vec_idx * VEC_SIZE) / x;
        const int offset2 = (vec_idx * VEC_SIZE) % x;

        if constexpr (KV_DTYPE == Fp8KVCacheDataType::kAuto) {
          k_vecs[j] = *reinterpret_cast<const K_vec*>(
              k_ptr + offset1 * BLOCK_SIZE * x + offset2);
        } else {
          // Vector conversion from Quant_vec to K_vec.
          Quant_vec k_vec_quant = *reinterpret_cast<const Quant_vec*>(
              k_ptr + offset1 * BLOCK_SIZE * x + offset2);
          k_vecs[j] = fp8::scaled_convert<K_vec, Quant_vec, KV_DTYPE>(
              k_vec_quant, *k_scale);
        }
      }

      // Compute dot product.
      // This includes a reduction across the threads in the same thread group.
      float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(
                             q_vecs[thread_group_offset], k_vecs);
      // Add the ALiBi bias if slopes are given.
      qk += (alibi_slope != 0) ? alibi_slope * (token_idx - seq_len + 1) : 0;

      if (thread_group_offset == 0) {
        // Store the partial reductions to shared memory.
        // NOTE(woosuk): It is required to zero out the masked logits.
        const bool mask = token_idx >= seq_len;
        logits[token_idx - start_token_idx] = mask ? 0.f : qk;
        // Update the max value.
        qk_max = mask ? qk_max : fmaxf(qk_max, qk);
      }

  // Perform reduction across the threads in the same warp to get the
  // max qk value for each "warp" (not across the thread block yet).
  // The 0-th thread of each thread group already has its max qk value.
#pragma unroll
```

关键步骤是：

1. `physical_block_number`、`kv_head_idx` 和 `physical_block_offset` 确定 K 的物理地址；
2. `KV_DTYPE == kAuto` 时直接加载；其他分支把 FP8 `Quant_vec` 按 `k_scale` 转成计算向量；
3. `Qk_dot` 在线程组内完成点积归约；
4. ALiBi bias 在缩放后的 QK 上累加；
5. 超出 `seq_len` 的尾块位置写 0，且不更新最大值。

片段末尾的 `#pragma unroll` 属于随后 max reduction 的循环提示。

#### 3.2.5 Max Reduction

```
  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
// Broadcast the max qk value to all threads.
  qk_max = VLLM_SHFL_SYNC(qk_max, 0);
```

该片段只展示跨 warp 阶段读取 `red_smem` 与广播结果。完整流程是：thread group 局部最大值 → warp 内 shuffle reduction → warp leader 写 shared memory → 跨 warp reduction → 广播 block-wide `qk_max`。V1 的处理范围是完整序列，V2 是当前 partition。

#### 3.2.6 稳定 Softmax

```
  // Get the sum of the exp values.
  float exp_sum = 0.f;
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

  // Compute softmax.
  const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  __syncthreads();
```

先计算 `exp(logit - qk_max)`，再用 `block_sum` 汇总指数和，最后乘 `1/(exp_sum+1e-6)` 完成归一化。这里会保存当前处理范围内的全部 logits，再统一做 max、sum-exp 和归一化；它不是 FlashAttention 按 K/V tile 持续维护 `(m,l,A)` 的实现。

#### 3.2.7 保存 V2 Partition 统计量

```
    float* max_logits_ptr = max_logits +
                            seq_idx * num_heads * max_num_partitions +
                            head_idx * max_num_partitions + partition_idx;
    *max_logits_ptr = qk_max;
    float* exp_sums_ptr = exp_sums + seq_idx * num_heads * max_num_partitions +
                          head_idx * max_num_partitions + partition_idx;
    *exp_sums_ptr = exp_sum;
```

真实源码在 `USE_PARTITIONING && thread_idx == 0` 条件内写入。每个 partition 保存 `qk_max` 与 `exp_sum`，供第二个 kernel 做跨 partition 稳定归并；V1 不需要这组临时 Tensor。

#### 3.2.8 读取 V 并累加输出

```
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE) {
        const int offset = row_idx * BLOCK_SIZE + physical_block_offset;
        V_vec v_vec;

        if (block_idx == num_seq_blocks - 1) {
          // NOTE(woosuk): When v_vec contains the tokens that are out of the
          // context, we should explicitly zero out the values since they may
          // contain NaNs. See
          // https://github.com/vllm-project/vllm/issues/641#issuecomment-1682544472
          scalar_t* v_vec_ptr = reinterpret_cast<scalar_t*>(&v_vec);
#pragma unroll
          for (int j = 0; j < V_VEC_SIZE; j++) {
            v_vec_ptr[j] = token_idx + j < seq_len ? v_vec_ptr[j] : zero_value;
          }
        }
        accs[i] += dot(logits_vec, v_vec);
      }
    }
```

这一段关注尾块保护与 `dot(logits_vec, v_vec)`。在它之前，kernel 已经：

- 根据 `physical_block_number` 与 KV head 定位 V block；
- 构造当前 token 范围的 `logits_vec`；
- 按 `KV_DTYPE` 直接加载 V，或完成 FP8 scale conversion。

最后一个 KV block 可能未填满，因此上下文之外的 V 元素必须显式清零。否则未使用 slot 中的脏数据或 NaN 可能污染累加结果。

#### 3.2.9 V2 跨 Partition 归并

> 代码第一行 `paged_attention_v2_reduce_kernel:` 是用于标识片段的伪标签，不是合法 C++；整个代码块从真实 reduce kernel 的中段开始，不能独立编译。下面只解释跨 partition 的归并逻辑。

```
paged_attention_v2_reduce_kernel:

  // Broadcast the max value to all threads.
  max_logit = VLLM_SHFL_SYNC(max_logit, 0);

  // Load rescaled exp sums to shared memory.
  float* shared_exp_sums =
      reinterpret_cast<float*>(shared_mem + sizeof(float) * num_partitions);
  const float* exp_sums_ptr = exp_sums +
                              seq_idx * num_heads * max_num_partitions +
                              head_idx * max_num_partitions;
  float global_exp_sum = 0.0f;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    float l = shared_max_logits[i];
    float rescaled_exp_sum = exp_sums_ptr[i] * expf(l - max_logit);
    global_exp_sum += rescaled_exp_sum;
    shared_exp_sums[i] = rescaled_exp_sum;
  }
  __syncthreads();
  global_exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], global_exp_sum);
  const float inv_global_exp_sum = __fdividef(1.0f, global_exp_sum + 1e-6f);

```

该片段从最大值广播后的中间位置开始。完整 reduce kernel 会先求所有 partitions 的全局最大值，再把局部指数和缩放到同一基准：

$$
m=\max_j m_j,
\qquad L=\sum_j l_j e^{m_j-m}
$$

$$
O=\sum_j O_j\frac{l_j e^{m_j-m}}{L}
$$

最后还会按相同权重合并 `tmp_out`。这属于跨 partition 的稳定归并，不是 FlashAttention 跨 K/V tile 的完整 online-softmax 流程。

### 3.3 V1 vs V2 对比

> **V1 Engine** 是 `vllm/v1/...` 的引擎架构；**PagedAttention V1/V2** 是 custom decode kernel 的两个执行方案，两组名称没有版本对应关系。

![PagedAttention V1 与 V2](/images/vllm-pagedattention-core/03_v1_v2_partitions.svg)

| 维度 | PagedAttention V1 | PagedAttention V2 |
|---|---|---|
| CTA grid | `(num_heads, num_seqs, 1)` | 增加 partition 维 |
| 单个 CTA 的上下文范围 | 完整序列 | 一个 compute partition |
| 临时结果 | 直接输出 | `tmp_out`、`max_logits`、`exp_sums` |
| 第二个 kernel | 不需要 | `paged_attention_v2_reduce_kernel` |

vLLM `v0.11.0` 默认 partition size 为 512 tokens。V2 通过增加 CUDA blocks 提高长序列或小 batch 下的并行度，但需要额外临时 Tensor 和一次 reduce kernel。

该版本的选择条件是：

```python
use_v1 = (
    max_seq_len <= 8192
    and (max_num_partitions == 1 or num_seqs * num_heads > 512)
)
```

满足条件时使用 V1，否则使用 V2。这是 v0.11.0 的启发式，不是 PagedAttention 的通用定义。

## 4. Prefix Caching：减少重复计算

### 4.1 核心机制

PagedAttention 解决“一个逻辑块放在哪里”；Prefix Caching 进一步解决“不同请求能否共同引用已经计算好的同一物理块”。

![Prefix Cache 共享物理块](/images/vllm-pagedattention-core/04_prefix_sharing.svg)

图中共享成立的前提是：相同完整 block 已由较早请求计算并提交。vLLM `v0.11.0` 在新 block 插入 cache 时不做去重，因此两个请求在同一轮首次生成相同完整 block，并不保证立即指向同一个 physical block；后续请求通过 cache lookup 才可能复用已经缓存的实例。完整边界见 [`block_pool.py:L30-L39`](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/core/block_pool.py#L30-L39)。

vLLM 为完整 token block 构造链式 hash：

$$
H_i=H(H_{i-1},\ token\_ids_i,\ extra\_keys_i)
$$

> 以下教学片段省略了完整函数签名、类型上下文和函数定义末尾的冒号，不能直接运行；vLLM `v0.11.0` 的完整实现见 [`kv_cache_utils.py:L547-L573`](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/core/kv_cache_utils.py#L547-L573)。

```
def hash_block_tokens()
    if not parent_block_hash:
        parent_block_hash = NONE_HASH

    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    return BlockHash(
        hash_function(
            (parent_block_hash, curr_block_token_ids_tuple, extra_keys)))
```

`parent_block_hash` 把 causal prefix 纳入当前块身份。即使中间某一块 token 完全相同，只要前面的上下文不同，就不能复用相同 KV；第 `i` 个块 miss 后，后续 hash 链也不能作为连续前缀继续命中。`extra_keys` 还可能包含多模态内容标识、LoRA ID 或 cache salt。

只有完整且可确认的 token block 才适合作为普通 Prefix Cache block。未填满尾块、可能被 speculative decoding 拒绝的 token，都不能提前提交为稳定 cache。

命中一个仍在 free queue 中的 block 时，`touch()` 会先把它移出淘汰候选队列，再增加引用计数：

```
    def touch(self, blocks: tuple[list[KVCacheBlock], ...]) -> None:
        for blocks_per_group in blocks:
            for block in blocks_per_group:
                # ref_cnt=0 means this block is in the free list (i.e. eviction
                # candidate), so remove it.
                if block.ref_cnt == 0 and not block.is_null:
                    self.free_block_queue.remove(block)
                block.ref_cnt += 1
```

请求释放引用时，`free_blocks()` 递减计数，并把归零且非 null 的 block 追加到 free queue：

```
    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        blocks_list = list(ordered_blocks)
        for block in blocks_list:
            block.ref_cnt -= 1
        self.free_block_queue.append_n([
            block for block in blocks_list
            if block.ref_cnt == 0 and not block.is_null
        ])
```

`ref_cnt=0` 不代表数据立即清零，只代表 block 可以被重新分配或淘汰；在尚未覆盖前，如果 hash 再次命中，`touch()` 可以把它取回。free queue 的次序与 block 的最近使用状态共同形成 LRU 风格的淘汰基础。

即使 Prompt 全部命中，也必须保留产生 logits 所需的计算。vLLM 至少需要重算最后一个 token；受 block 对齐限制，某些路径还会回退一个完整块。

### 4.2 实测数据（探索性观测）

下面是一组探索性端到端观测：

| 配置 | 第一轮（截图标注“冷启动”） | 第二轮（截图标注“热缓存”） | 本配置第一轮 / 第二轮（加速比） | 第二轮相对 OFF |
|---|---:|---:|---:|---:|
| Prefix Cache OFF | 11.55 s | 11.53 s | 1.00× | 基准 |
| Prefix Cache ON | 7.83 s | 7.66 s | 1.02× | 约 1.51× |

<details>
<summary>查看 Prefix Cache 运行截图</summary>

<img src="/images/vllm-pagedattention-core/03_prefix_cache_off.png" alt="Prefix Cache OFF 测试截图">

<img src="/images/vllm-pagedattention-core/04_prefix_cache_on.png" alt="Prefix Cache ON 测试截图">

</details>

这组数据没有记录 `num_cached_tokens`、稳定 TTFT、模型 warmup 和完整运行参数，因此只能说明当时的 ON 配置端到端时间更短，不能把全部差异归因于 Prefix Cache。

更可靠的实验应固定 Prompt 与输出长度，用无关请求完成模型 warmup，再执行 cold → repeated-prefix 对照；同时记录命中 token 数，并以 `max_tokens=1` 的 TTFT 为主要指标，尽量隔离后续 Decode。

### 4.3 适用场景

| 场景 | Prefix Cache 价值 | 关键条件 |
|---|---|---|
| 多轮对话 | 复用历史对话前缀 | 新一轮输入必须以前一轮完整历史开头 |
| 批量任务 | 复用模板或固定指令 | 公共前缀至少形成一个完整 block |
| 相同 system prompt | 降低重复 Prefill | hash 的 extra keys 必须兼容 |
| 每个请求前缀都不同 | 通常收益很小 | 几乎没有连续 prefix hit |

Prefix Caching 的收益取决于命中长度、block size、cache 压力和到达顺序；“开关已启用”不等于当前请求一定命中。

## 5. Chunked Prefill：混合调度

### 5.1 问题：长 Prompt 的饥饿

Prefill 需要处理全部 Prompt tokens。若一个 32K Prompt 一次性占用整轮 token budget 和 GPU 时间，已经进入 Decode 的短请求就可能长时间拿不到下一个 token，表现为 TTFT/ITL 尾延迟恶化。

Chunked Prefill 把长 Prompt 拆成多个 scheduler iterations。每轮只调度一段 Prompt tokens，使同轮尚未消耗的共享 budget 有机会容纳其他请求；是否能容纳、以及实际剩余多少，取决于请求处理顺序、总 token budget 和 `long_prefill_token_threshold`。它优化的是服务公平性与 head-of-line blocking，不会减少长 Prompt 自身必须完成的 Attention 计算量。

### 5.2 Chunked Prefill 方案

vLLM V1 Scheduler 不维护两条互斥的 Prefill/Decode phase；它在一份共享 token budget 内为每个请求计算 `num_new_tokens`：

```
num_new_tokens = (request.num_tokens_with_spec +
                  request.num_output_placeholders -
                  request.num_computed_tokens)
if (0 < self.scheduler_config.long_prefill_token_threshold <
        num_new_tokens):
    num_new_tokens = (
        self.scheduler_config.long_prefill_token_threshold)
num_new_tokens = min(num_new_tokens, token_budget)

# Make sure the input position does not exceed the max model len.
# This is necessary when using spec decoding.
num_new_tokens = min(
    num_new_tokens,
    self.max_model_len - 1 - request.num_computed_tokens)
```

代码中依次考虑：请求尚未计算的 token、`long_prefill_token_threshold`、当前 `token_budget`，以及 speculative decoding 下不能超过 `max_model_len-1` 的位置限制。

当请求成功进入本轮调度后，消耗的 token 数从共享 budget 扣除：

```
token_budget -= num_new_tokens
```

![Chunked Prefill 与 Decode 混合调度](/images/vllm-pagedattention-core/05_mixed_schedule.svg)

例如一轮中两个 Decode 请求各得到 1 token，长 Prompt 得到 4-token chunk：

```text
num_scheduled_tokens = [1, 1, 4]
flattened tokens      = [D0, D1, P0, P1, P2, P3]
query_start_loc       = [0, 1, 2, 6]
```

相邻 `query_start_loc` 的差就是本轮各请求的 `q_len`。vLLM `v0.11.0` 中，真正根据 `q_len` 划分计算形态的是 [`split_decodes_and_prefills`](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/attention/backends/utils.py#L721-L771)：

```python
query_lens = query_start_loc[1:] - query_start_loc[:-1]
if require_uniform:
    is_prefill = query_lens != query_lens[0]
else:
    is_prefill = query_lens > decode_threshold
```

默认 `decode_threshold=1`，所以普通路径中 `q_len <= 1` 会被归为 decode-like；如果 `require_uniform=True`，则只有与 batch 首个 Query 长度相同的连续请求留在 decode 侧。

同一文件中的另一个 helper 会在模型执行前按本轮调度 token 数重排 batch，把 decode-like 请求移到前面。下面是它的局部逻辑：

> 该局部片段省略了 `decodes/prefills` 初始化、batch move 和 `return`，不能直接运行；完整实现见 [`utils.py:L774-L834`](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/attention/backends/utils.py#L774-L834)。

```
def reorder_batch_to_split_decodes_and_prefills(
    input_batch: "InputBatch",
    scheduler_output: "SchedulerOutput",
    decode_threshold: int = 1,
) -> bool:
    """
    Reorders the batch to split into prefill and decode requests; places all
    requests with <= decode_threshold tokens at the front of the batch.
    
    Returns:
        True if the batch was modified, False otherwise.
    """

    for i, req_id in enumerate(input_batch.req_ids):
        num_tokens = scheduler_output.num_scheduled_tokens[req_id]
        if num_tokens <= decode_threshold:
            decodes.append(i)
            num_decode_tokens += num_tokens
        else:
            prefills.append(i)
            num_prefill_tokens += num_tokens
```

展示的 `reorder_batch_to_split_decodes_and_prefills` 直接读取 `scheduler_output.num_scheduled_tokens`，并不读取 `query_start_loc`；`q_len` 判断属于前面的 `split_decodes_and_prefills`。两者的默认 threshold 都是 1，但 backend 可以覆盖。这里的 decode/prefill 只描述 attention backend 眼中的本轮计算形态，不是 Scheduler 为请求保存的永久状态。

同一 scheduler/model batch 也不保证进入同一个 Attention kernel：

- FlashAttention 可以用一次 varlen kernel 处理混合 Query 长度；
- FlashInfer 可以在同一个 layer forward 内分别调用 Prefill 与 Decode wrapper；
- 其他 backend 也可能先重排，再走不同 kernel。

Paged KV 让分段调度真正高效：前一个 chunk 的 K/V 留在已分配物理块中；下一轮只增量写入新 K/V，并通过 Block Table 读取完整历史，不需要拼接整段缓存。

### 5.3 实测效果（探索性观测）

截图中的实验汇总如下。ON 组显式设置 `max_num_batched_tokens=2048`、`long_prefill_token_threshold=1024`；OFF 组的 token budget 为 131072。

| 实验 | 配置 | Long TTFT | Long 总延迟 | Short TTFT median | Short TTFT max | Short latency median | Short latency max | 现象 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 实验 1 | Chunked Prefill ON，budget 2048，threshold 1024 | 25.89 s | 26.55 s | 0.415 s | 0.688 s | 0.416 s | 0.691 s | 短请求基本都能在 Long Prefill 期间插入执行 |
| 实验 2 | Chunked Prefill OFF，budget 131072 | 26.74 s | 27.45 s | 25.597 s | 26.437 s | 25.602 s | 26.441 s | 短请求基本被 Long Prefill 阻塞到 Long 首 token 附近 |

截图给出的逐指标对比保留了更多小数位：

| 指标 | Chunked ON + threshold 1024 | Chunked OFF | 截图中的改善幅度 |
|---|---:|---:|---|
| Short TTFT median | 0.415 s | 25.597 s | 约 61.7× 更低 |
| Short TTFT max | 0.688 s | 26.437 s | 约 38.4× 更低 |
| Short latency median | 0.416 s | 25.602 s | 约 61.5× 更低 |
| Short latency max | 0.691 s | 26.441 s | 约 38.3× 更低 |
| Long TTFT | 25.895 s | 26.738 s | ON 略快约 0.84 s |
| Long 总延迟 | 26.555 s | 27.452 s | ON 略快约 0.90 s |

按请求时间线统计，短请求首 token 的分布为：

| 实验 | Short 请求首 token 分布 | Long first token 时间 | 结论 |
|---|---|---:|---|
| Chunked ON + threshold 1024 | `short-0` 至 `short-19` 都在约 0.95–2.59 s 拿到首 token | 26.04 s | 短请求成功穿插到 Long Prefill 中 |
| Chunked OFF | 除 `short-0` 外，大部分 Short 请求在约 26.85–26.96 s 拿到首 token | 26.87 s | 短请求被 Long Prefill 阻塞，直到 Long Prefill 结束 |

<details>
<summary>查看 Chunked Prefill 延迟截图</summary>

<img src="/images/vllm-pagedattention-core/05_chunked_prefill_latency.png" alt="Chunked Prefill 延迟截图">

</details>

两组观测相差约 61.7×，但实验同时修改了 Chunked Prefill 标签和 scheduler token budget，而且没有记录 engine 版本与 effective config，不能当作 Chunked Prefill 单变量加速比。

还有一个明确的版本边界：vLLM `v0.11.0` 在常见 GPU/TPU 的 V1 非 pooling 路径中会启用 Chunked Prefill；POWER、ARM 和 s390x CPU 是平台例外，V1 会再次将其禁用。因此表中的 OFF 标签不能直接视为当前常见 GPU V1 路径的合法对照，它可能来自其他 engine、版本或例外平台。

Chunk 越小并不必然越好：

- 小 chunk 通常改善短请求 TTFT/ITL 和调度公平性；
- 过小会增加 scheduler iterations、kernel launch 和长 Prompt 完成时间；
- 大 chunk 提高单轮计算量，却可能重新放大短请求尾延迟。

应固定真实到达时间线，扫描 `max_num_batched_tokens` 与 `long_prefill_token_threshold`，同时报告短请求 TTFT/ITL P50/P95/P99 和长请求吞吐。

## 6. Sampling：从 Logits 到 Token

### 6.1 常见采样方法

最后一层 hidden state 经 LM Head 投影为词表 logits，Sampler 才开始工作：

```text
Transformer → LM Head → logits → Sampler → sampled_token_ids
                                               ↓
                          更新请求、检查停止条件、进入下一轮
```

Greedy Sampling 直接选择最大 logit：

```
next_token = torch.argmax(logits, dim=-1)
```

Temperature、Top-k 和 Top-p 的最小参数表示如下：

```
logits = logits / temperature
```

这里的 `temperature=0` 是 Greedy 的接口语义，不是实际除数。vLLM `v0.11.0` 把 `<1e-5` 视为 Greedy；混合 batch 在缩放前会把这些行的 temperature 临时替换为 `1.0`，随后再用 `torch.where` 选择 Greedy 结果，因此不会执行 `logits / 0`。实现见 [`sampler.py:L127-L195`](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/sample/sampler.py#L127-L195)。

```
top_k = 5
```

```
top_p = 0.9
```

| 方法 | 适用场景 | 输出特点 | 性能影响 |
|---|---|---|---|
| Greedy，`temperature=0` | 代码、数学、分类、结构化抽取 | 确定性最高，直接选择最大 logit | 最低，主要是一次 `argmax` |
| Temperature | 通用对话、创作 | 调整整体分布的尖锐程度 | 需要缩放、softmax 和随机采样，比 Greedy 稍慢 |
| Top-k | 希望候选数量固定 | 只保留概率最高的 `k` 个 token | 中等；Top-k-only 不需要全词表排序 |
| Top-p | 对话、开放式生成 | 保留累计概率达到 `p` 的动态候选集 | 原生实现较贵，需要对整个词表排序 |
| Min-p | 高温创作、抑制低概率长尾 token | 相对于最高概率动态过滤 | 需要 softmax、max 和 mask，但不需要排序 |
| Beam Search | 翻译、短序列搜索、候选评分 | 同时保留多个高分路径 | 很高；计算量和 KV Cache 大致随 beam width 增长 |
| `n>1` 独立采样 | 需要多个创意候选 | 生成多条随机序列 | Decode 计算和 KV Cache 近似随 `n` 增长 |

参数没有脱离任务的全局最优值。确定性任务通常从 Greedy 开始；开放式对话常用适中的 Temperature + Top-p；高温导致长尾候选过多时，可以评估 Min-p。

### 6.2 性能影响

采样通常小于大模型 forward 的成本，但在大 batch、大词表、小模型或请求大量 `logprobs` 时可能成为明显开销。

vLLM `v0.11.0` 的 batch 路径有三种：

- **全 Greedy**：计算 `argmax` 后提前返回；
- **全 Random**：跳过 Greedy，只计算随机路径；
- **Greedy/Random 混合**：两条结果都计算，再用 `torch.where` 按请求选择。

不同过滤器的主要额外工作也不同：

| 路径 | 主要工作 | 性能边界 |
|---|---|---|
| Greedy | 全词表 `argmax` | 通常最轻，但仍扫描 vocab |
| Temperature | 逐元素缩放 | 后续仍需概率化和随机采样 |
| Native Top-k only | `topk` | 不必完整排序全词表 |
| Native Top-p | sort、softmax、cumsum、scatter | 大词表时更明显 |
| Min-p | 全词表 softmax、amax、阈值 mask，之后仍随机采样 | 高温或大词表时会增加概率化与逐元素处理开销 |
| FlashInfer rejection sampling | 避免 native 全排序 | 有版本、seed、logprobs mode、环境变量限制，并包含 CPU–GPU 同步 |

`logprobs=N` 常比小幅调整 `top_p` 更值得关注：默认 raw-logprobs 路径会计算 FP32 `log_softmax`，随后还要做 top-k、sampled-token gather 和 rank 计算。

Beam Search 和 `n>1` 都会增加活跃请求数、模型 forward 工作量和峰值 KV 占用。实际倍率受 Prefix Cache、候选提前结束和 scheduler batching 影响，不能简单写成严格的 `W×` 或 `n×`。

<details>
<summary>查看 Sampling 方法截图</summary>

<img src="/images/vllm-pagedattention-core/06_sampling_methods.png" alt="Sampling 方法截图">

</details>

## 7. 长上下文综合优化

### 7.1 32K Tokens 场景

长上下文没有单一“最佳开关”。首先要区分负载：

- **单文档问答**：Prompt 很长，但公共前缀较少。主要矛盾是 Prefill 计算、KV 容量和短请求排队；Paged KV + 合理 token budget 更重要。
- **多轮对话**：每轮都携带之前的完整历史。若历史形成稳定完整 blocks，Prefix Cache 可以复用大量 causal prefix；Paged KV 负责继续追加新 K/V。

以前述 32 层、8 KV heads、`head_dim=128`、BF16 为例，32K KV Cache 约 4 GiB。并发容量还要考虑模型权重、activation workspace、CUDA Graph、临时 partition buffers 和 allocator 对齐，不能只用 HBM 总容量除以 4 GiB。

| 场景 | 首要矛盾 | 重点机制 | 关键指标 |
|---|---|---|---|
| 单个长文档 | Prefill 计算量、KV 容量 | Paged KV、Chunked Prefill | TTFT、吞吐、可用 KV blocks、抢占 |
| 公共模板批处理 | 重复前缀计算 | Prefix Cache | 命中 token、TTFT、block 淘汰 |
| 多轮对话 | 历史前缀重复、尾部持续增长 | Prefix Cache + Paged KV | 命中长度、ITL、端到端延迟 |
| 长 Prefill + 短 Decode | head-of-line blocking | Chunked Prefill | 短请求 TTFT/ITL P95/P99、长请求吞吐 |

### 7.2 优化组合效果（探索性观测）

| 配置 | 单文档 | 多轮对话 | 截图中的显存 | 单文档 / 多轮相对 Baseline |
|---|---:|---:|---:|---:|
| Baseline | 16.39 s | 29.69 s | 0.00 GB | 1.00× / 1.00× |
| Chunked Prefill，budget 2048 | 16.36 s | 29.76 s | 0.00 GB | 1.00× / 1.00× |
| Prefix Cache | 16.44 s | 13.28 s | 0.00 GB | 1.00× / 2.24× |
| Prefix Cache + Chunked，budget 2048 | 16.41 s | 13.35 s | 0.00 GB | 1.00× / 2.22× |
| Prefix Cache + Chunked，budget 4096 | 16.44 s | 13.27 s | 0.00 GB | 1.00× / 2.24× |

<details>
<summary>查看长上下文组合截图</summary>

<img src="/images/vllm-pagedattention-core/07_long_context_combo.png" alt="长上下文组合截图">

</details>

观测方向与机制预期兼容：若该次单文档输入没有可复用前缀，几组配置差异很小是合理现象；多轮对话在 Prefix Cache 标签下时间更短。但实验缺少 `num_cached_tokens`、控制变量和完整运行配置，不能据此确认实际命中，也不能把约 2.24× 的差异当作正式加速比。截图中的显存列为 `0.00 GB`，同样不能据此得出显存节省结论。

### 7.3 最佳实践

先确定 SLA，再回放真实 Prompt/输出长度和到达分布。建议按以下顺序调优：

1. **验证地址与容量**：确认 block size、KV dtype、可用 KV blocks、抢占和重计算；
2. **验证 Prefix Cache**：记录 `num_cached_tokens`，区分启用与实际命中；
3. **扫描 token budget**：逐步调整 `max_num_batched_tokens`，观察 TTFT/ITL 与吞吐；
4. **限制单个长 Prefill**：必要时设置 `long_prefill_token_threshold`，避免一个请求吞掉大部分预算；
5. **控制 batch 上限**：`max_num_seqs` 只是序列数上限，实际并发仍受 token budget 与 KV 容量约束；
6. **单变量实验**：固定 engine 版本、backend、模型、输入和到达时间线，每次只改一个参数。

正式报告至少应包含：

- 模型、dtype、KV dtype、GPU、vLLM commit、Attention backend 与完整启动参数；
- Prompt tokens、公共前缀 tokens、输出 tokens、并发和到达时间线；
- TTFT、ITL、端到端延迟和吞吐的 P50/P95/P99；
- Prefix Cache 命中 token、GPU KV blocks、抢占/重计算次数；
- cold start、warmup、重复次数与测量区间。

PagedAttention 的核心不是把 Attention 公式改成“分页公式”，而是建立稳定的 KV 地址抽象：

```text
逻辑 token 位置
  → logical block + offset
  → block_table
  → physical block
  → K/V 地址
```

在这层抽象之上，FlashAttention 优化 IO，Prefix Cache 复用公共前缀，Chunked Prefill 改善调度公平性，Sampling 决定 logits 之后的候选过滤与选择。只有把存储、计算、调度和采样分层观察，才能正确解释长上下文服务的容量、吞吐与尾延迟。

## 参考资料

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [NVIDIA Ampere GPU Architecture Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/)
- [V1 Scheduler：统一 token budget](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/core/sched/scheduler.py)
- [EngineArgs：V1 Chunked Prefill 版本边界](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/engine/arg_utils.py)
- [GPU Model Runner：ragged batch 与 Attention metadata](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/worker/gpu_model_runner.py)
- [BlockTable：slot_mapping 计算](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/worker/block_table.py)
- [Attention backend utilities：q_len 与 decode_threshold](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/attention/backends/utils.py)
- [Prefix Cache hash chain](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/core/kv_cache_utils.py)
- [Block Pool：引用计数与 free queue](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/core/block_pool.py)
- [Custom PagedAttention wrapper 与 V1/V2 选择](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/attention/ops/paged_attn.py)
- [Custom PagedAttention CUDA kernel](https://github.com/vllm-project/vllm/blob/v0.11.0/csrc/attention/attention_kernels.cuh)
- [V1 Sampler](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/sample/sampler.py)
- [Top-k/Top-p sampler](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/sample/ops/topk_topp_sampler.py)
