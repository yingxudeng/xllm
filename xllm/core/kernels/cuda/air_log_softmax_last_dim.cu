/* Copyright 2025 The xLLM Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://github.com/jd-opensource/xllm/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================*/

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cub/block/block_reduce.cuh>
#include <limits>

#include "air_log_softmax_last_dim.h"
#include "utils.h"

namespace xllm::kernel::cuda {

namespace {

template <typename scalar_t, int kThreads>
__global__ void log_softmax_last_dim_kernel(const scalar_t* __restrict__ input,
                                            const float* __restrict__ temps,
                                            bool has_temps,
                                            float* __restrict__ output,
                                            int32_t k,
                                            int64_t stride) {
  const int32_t row = static_cast<int32_t>(blockIdx.x);
  const int64_t base = static_cast<int64_t>(row) * stride;

  float inv_temp = 1.0f;
  if (has_temps) {
    float t = temps[row];
    if (t == 0.0f) {
      t = 1.0f;
    }
    inv_temp = 1.0f / t;
  }

  using BlockReduce = cub::BlockReduce<float, kThreads>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;

  float thread_max = -std::numeric_limits<float>::infinity();
  for (int32_t col = threadIdx.x; col < k; col += blockDim.x) {
    float x = static_cast<float>(input[base + col]) * inv_temp;
    thread_max = x > thread_max ? x : thread_max;
  }
  float row_max = BlockReduce(reduce_storage).Reduce(thread_max, cub::Max());
  __syncthreads();

  float thread_sum = 0.0f;
  for (int32_t col = threadIdx.x; col < k; col += blockDim.x) {
    float x = static_cast<float>(input[base + col]) * inv_temp;
    thread_sum += expf(x - row_max);
  }
  float row_sum = BlockReduce(reduce_storage).Sum(thread_sum);
  __syncthreads();

  float log_denom = logf(row_sum) + row_max;
  for (int32_t col = threadIdx.x; col < k; col += blockDim.x) {
    float x = static_cast<float>(input[base + col]) * inv_temp;
    output[base + col] = x - log_denom;
  }
}

}  // namespace

torch::Tensor air_log_softmax_last_dim(const torch::Tensor& input,
                                       const torch::Tensor& temperatures) {
  TORCH_CHECK(input.is_cuda(), "air_log_softmax_last_dim: input must be CUDA");
  TORCH_CHECK(input.dim() == 2,
              "air_log_softmax_last_dim: input must be 2D [B, K]");

  const int64_t batch64 = input.size(0);
  const int64_t k64 = input.size(1);
  TORCH_CHECK(batch64 >= 0 && batch64 <= INT32_MAX,
              "air_log_softmax_last_dim: batch too large");
  TORCH_CHECK(k64 > 0 && k64 <= INT32_MAX,
              "air_log_softmax_last_dim: k too large");

  const int32_t batch = static_cast<int32_t>(batch64);
  const int32_t k = static_cast<int32_t>(k64);

  if (batch == 0) {
    return torch::empty(
        {batch64, k64},
        torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));
  }

  bool has_temps = temperatures.defined();
  torch::Tensor temps = temperatures;
  if (has_temps) {
    TORCH_CHECK(temps.is_cuda(),
                "air_log_softmax_last_dim: temperatures must be CUDA");
    TORCH_CHECK(temps.dim() == 1,
                "air_log_softmax_last_dim: temperatures must be 1D [B]");
    TORCH_CHECK(temps.size(0) == batch64,
                "air_log_softmax_last_dim: temperatures size mismatch");
    TORCH_CHECK(temps.scalar_type() == torch::kFloat32,
                "air_log_softmax_last_dim: temperatures must be float32");
    temps = temps.contiguous();
  }

  c10::cuda::CUDAGuard device_guard(input.device());
  auto in = input.contiguous();
  auto out = torch::empty(
      {batch64, k64},
      torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));

  const int64_t stride = k64;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  constexpr int kThreads = 256;
  dim3 grid(batch);
  dim3 block(std::min<int32_t>(kThreads, 1024));

  DISPATCH_FLOATING_TYPES(in.scalar_type(), "air_log_softmax_last_dim", [&] {
    const scalar_t* in_ptr = in.data_ptr<scalar_t>();
    const float* t_ptr = has_temps ? temps.data_ptr<float>() : nullptr;
    float* out_ptr = out.data_ptr<float>();
    log_softmax_last_dim_kernel<scalar_t, kThreads><<<grid, block, 0, stream>>>(
        in_ptr, t_ptr, has_temps, out_ptr, k, stride);
  });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace xllm::kernel::cuda
