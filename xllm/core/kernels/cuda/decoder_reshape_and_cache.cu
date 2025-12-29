/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/cuda.h>

#include "cuda_ops_api.h"
#include "utils.h"

namespace {

// 融合的 decoder reshape and cache kernel
// 将 proj_k 和 proj_v 复制到 unshared_k_cache 和 unshared_v_cache 的指定位置
// 输入:
//   proj_k: [batch_size, beam_size, kv_heads, head_dim]
//   proj_v: [batch_size, beam_size, kv_heads, head_dim]
//   unshared_k_cache: [max_num_request, beam_size, max_decode_step, kv_heads, head_dim]
//   unshared_v_cache: [max_num_request, beam_size, max_decode_step, kv_heads, head_dim]
//   block_table: [batch_size] - 每个 batch 对应的 block_id
//   step: 当前 decode step
template <typename scalar_t>
__global__ void decoder_reshape_and_cache_kernel(
    const scalar_t* __restrict__ proj_k,           // [batch_size, beam_size, kv_heads, head_dim]
    const scalar_t* __restrict__ proj_v,           // [batch_size, beam_size, kv_heads, head_dim]
    scalar_t* __restrict__ unshared_k_cache,       // [max_num_request, beam_size, max_decode_step, kv_heads, head_dim]
    scalar_t* __restrict__ unshared_v_cache,       // [max_num_request, beam_size, max_decode_step, kv_heads, head_dim]
    const int64_t* __restrict__ block_table,       // [batch_size]
    const int64_t batch_size,
    const int64_t beam_size,
    const int64_t kv_heads,
    const int64_t head_dim,
    const int64_t max_decode_step,
    const int64_t max_num_request,
    const uint32_t step) {
  
  // 每个 thread 处理一个 (batch, beam, kv_head, head_dim) 的元素
  // 或者每个 thread 处理一个 (batch, beam, kv_head) 的完整 head_dim（更高效）
  
  // 计算全局索引
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total_elements = batch_size * beam_size * kv_heads;
  
  if (idx >= total_elements) {
    return;
  }
  
  // 分解索引
  const int64_t batch_idx = idx / (beam_size * kv_heads);
  const int64_t remaining = idx % (beam_size * kv_heads);
  const int64_t beam_idx = remaining / kv_heads;
  const int64_t kv_head_idx = remaining % kv_heads;
  
  // 获取对应的 block_id
  const int64_t block_id = block_table[batch_idx];
  
  // 边界检查
  if (block_id < 0 || block_id >= max_num_request) {
    return;
  }
  
  // 计算源和目标的基址索引
  // proj_k[batch_idx, beam_idx, kv_head_idx, :] 的基址
  const int64_t src_base = ((batch_idx * beam_size + beam_idx) * kv_heads + kv_head_idx) * head_dim;
  
  // unshared_k_cache[block_id, beam_idx, step, kv_head_idx, :] 的基址
  const int64_t dst_k_base = (((block_id * beam_size + beam_idx) * max_decode_step + step) * kv_heads + kv_head_idx) * head_dim;
  const int64_t dst_v_base = (((block_id * beam_size + beam_idx) * max_decode_step + step) * kv_heads + kv_head_idx) * head_dim;
  
  // 复制整个 head_dim（向量化复制，更高效）
  for (int64_t d = 0; d < head_dim; ++d) {
    unshared_k_cache[dst_k_base + d] = proj_k[src_base + d];
    unshared_v_cache[dst_v_base + d] = proj_v[src_base + d];
  }
}

} // namespace

namespace xllm::kernel::cuda {

void decoder_reshape_and_cache(torch::Tensor proj_k,
                                torch::Tensor proj_v,
                                torch::Tensor unshared_k_cache,
                                torch::Tensor unshared_v_cache,
                                torch::Tensor block_table,
                                uint32_t step) {
  // 输入检查
  TORCH_CHECK(proj_k.dim() == 4, "proj_k must be 4-dimensional");
  TORCH_CHECK(proj_v.dim() == 4, "proj_v must be 4-dimensional");
  TORCH_CHECK(unshared_k_cache.dim() == 5, "unshared_k_cache must be 5-dimensional");
  TORCH_CHECK(unshared_v_cache.dim() == 5, "unshared_v_cache must be 5-dimensional");
  TORCH_CHECK(block_table.dim() == 2, "block_table must be 2-dimensional");
  TORCH_CHECK(block_table.size(1) == 1, "block_table second dim must be 1");
  
  const int64_t batch_size = proj_k.size(0);
  const int64_t beam_size = proj_k.size(1);
  const int64_t kv_heads = proj_k.size(2);
  const int64_t head_dim = proj_k.size(3);
  const int64_t max_num_request = unshared_k_cache.size(0);
  const int64_t max_decode_step = unshared_k_cache.size(2);
  
  // 形状兼容性检查
  TORCH_CHECK(proj_v.sizes() == proj_k.sizes(), "proj_v and proj_k must have same shape");
  TORCH_CHECK(block_table.size(0) == batch_size, "block_table size must match batch_size");
  TORCH_CHECK(step >= 0 && step < max_decode_step, "step must be in valid range");
  TORCH_CHECK(unshared_k_cache.size(1) == beam_size, "unshared_k_cache beam_size mismatch");
  TORCH_CHECK(unshared_k_cache.size(3) == kv_heads, "unshared_k_cache kv_heads mismatch");
  TORCH_CHECK(unshared_k_cache.size(4) == head_dim, "unshared_k_cache head_dim mismatch");
  TORCH_CHECK(unshared_v_cache.sizes() == unshared_k_cache.sizes(), 
              "unshared_v_cache and unshared_k_cache must have same shape");
  
  const at::cuda::OptionalCUDAGuard device_guard(device_of(proj_k));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  
  // 准备 block_table（只取第一列，转换为 int64_t）
  torch::Tensor block_table_flat = block_table.select(1, 0).to(torch::kInt64);
  
  // Launch kernel
  const int64_t total_elements = batch_size * beam_size * kv_heads;
  const int threads_per_block = 256;
  const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
  
  DISPATCH_FLOATING_TYPES(proj_k.scalar_type(), "decoder_reshape_and_cache_kernel", [&] {
    decoder_reshape_and_cache_kernel<scalar_t><<<blocks, threads_per_block, 0, stream>>>(
        proj_k.data_ptr<scalar_t>(),
        proj_v.data_ptr<scalar_t>(),
        unshared_k_cache.data_ptr<scalar_t>(),
        unshared_v_cache.data_ptr<scalar_t>(),
        block_table_flat.data_ptr<int64_t>(),
        batch_size,
        beam_size,
        kv_heads,
        head_dim,
        max_decode_step,
        max_num_request,
        step);
  });
  
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace xllm::kernel::cuda

