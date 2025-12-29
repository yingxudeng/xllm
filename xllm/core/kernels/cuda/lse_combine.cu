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
#include <cmath>

#include "cuda_ops_api.h"
#include "utils.h"

namespace {

// 融合的 LSE combine kernel
// 每个 thread 处理一个 (batch_idx, head_idx) 的完整 head_dim
// 输入:
//   shared_o: [B, H, D] - shared attention output
//   shared_lse: [B, H, 1] - shared log-sum-exp
//   unshared_o: [B, H, D] - unshared attention output
//   unshared_lse: [B, H, 1] - unshared log-sum-exp
// 输出:
//   output: [B, H, D] - combined output
template <typename scalar_t>
__global__ void lse_combine_kernel(
    scalar_t* __restrict__ output,           // [B, H, D]
    const scalar_t* __restrict__ shared_o,   // [B, H, D]
    const float* __restrict__ shared_lse,    // [B, H, 1], always FP32
    const scalar_t* __restrict__ unshared_o, // [B, H, D]
    const float* __restrict__ unshared_lse,  // [B, H, 1], always FP32
    const int64_t B,    // batch_size * beam_size
    const int64_t H,    // num_heads
    const int64_t D) {  // head_dim
  
  // 每个 thread 处理一个 (batch_idx, head_idx) 的完整 head_dim
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total_elements = B * H;
  
  if (idx >= total_elements) {
    return;
  }
  
  const int64_t batch_idx = idx / H;
  const int64_t head_idx = idx % H;
  
  // 计算 LSE 相关的值（融合所有中间计算）
  const float shared_lse_val = static_cast<float>(shared_lse[idx]);
  const float unshared_lse_val = static_cast<float>(unshared_lse[idx]);
  
  // 1. 计算 element-wise 最大 LSE
  const float li_max = fmaxf(shared_lse_val, unshared_lse_val);
  
  // 2. 计算以 2 为底的指数差
  const float exp_li = exp2f(shared_lse_val - li_max);
  const float exp_lij = exp2f(unshared_lse_val - li_max);
  
  // 3. 计算合并后的新 LSE
  const float li_new = li_max + log2f(exp_li + exp_lij);
  
  // 4. 计算归一化权重
  const float wi = exp2f(shared_lse_val - li_new);
  const float wij = exp2f(unshared_lse_val - li_new);
  
  // 5. 加权合并输出（每个 thread 处理完整的 head_dim）
  const int64_t base_idx = idx * D;
  for (int64_t d = 0; d < D; ++d) {
    const float shared_val = static_cast<float>(shared_o[base_idx + d]);
    const float unshared_val = static_cast<float>(unshared_o[base_idx + d]);
    const float combined = wi * shared_val + wij * unshared_val;
    output[base_idx + d] = static_cast<scalar_t>(combined);
  }
}

} // namespace

namespace xllm::kernel::cuda {

void lse_combine(torch::Tensor output,
                 torch::Tensor shared_o,
                 torch::Tensor shared_lse,
                 torch::Tensor unshared_o,
                 torch::Tensor unshared_lse) {
  // 输入检查
  TORCH_CHECK(shared_o.dim() == 3, "shared_o must be 3D [B, H, D]");
  TORCH_CHECK(unshared_o.dim() == 3, "unshared_o must be 3D [B, H, D]");
  TORCH_CHECK(shared_lse.dim() == 3, "shared_lse must be 3D [B, H, 1]");
  TORCH_CHECK(unshared_lse.dim() == 3, "unshared_lse must be 3D [B, H, 1]");
  
  const int64_t B = shared_o.size(0);
  const int64_t H = shared_o.size(1);
  const int64_t D = shared_o.size(2);
  
  TORCH_CHECK(shared_o.sizes() == unshared_o.sizes(), 
              "shared_o and unshared_o must have same shape");
  TORCH_CHECK(shared_lse.scalar_type() == torch::kFloat32,
              "shared_lse must be float32");
  TORCH_CHECK(unshared_lse.scalar_type() == torch::kFloat32,
              "unshared_lse must be float32");
  TORCH_CHECK(shared_lse.size(0) == B && shared_lse.size(1) == H && shared_lse.size(2) == 1,
              "shared_lse shape mismatch");
  TORCH_CHECK(unshared_lse.size(0) == B && unshared_lse.size(1) == H && unshared_lse.size(2) == 1,
              "unshared_lse shape mismatch");
  
  // 确保 output 的形状和类型正确
  if (!output.defined() || output.sizes() != shared_o.sizes()) {
    output = torch::empty_like(shared_o);
  }
  
  const at::cuda::OptionalCUDAGuard device_guard(device_of(shared_o));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  
  // Launch kernel
  const int64_t total_elements = B * H;
  const int threads_per_block = 256;
  const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
  
  DISPATCH_FLOATING_TYPES(shared_o.scalar_type(), "lse_combine_kernel", [&] {
    lse_combine_kernel<scalar_t><<<blocks, threads_per_block, 0, stream>>>(
        output.data_ptr<scalar_t>(),
        shared_o.data_ptr<scalar_t>(),
        shared_lse.data_ptr<float>(),
        unshared_o.data_ptr<scalar_t>(),
        unshared_lse.data_ptr<float>(),
        B, H, D);
  });
  
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace xllm::kernel::cuda
