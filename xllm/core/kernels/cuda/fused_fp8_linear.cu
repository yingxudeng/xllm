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
 * ===========================================================================*/

// clang-format off
#include "fused_fp8_linear.cuh"
#include "fp8_quant_utils.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
// clang-format on

namespace xllm {
namespace kernel {
namespace cuda {

// =============================================================================
// Utility functions
// =============================================================================

// Convert various types to float
// Note: at::Half and at::BFloat16 are ATen types with implicit conversion to
// float.
template <typename T>
__device__ __forceinline__ float to_float(T val) {
  return static_cast<float>(val);
}

// Convert float to various types
// Note: at::Half and at::BFloat16 can be constructed from float via
// static_cast.
template <typename T>
__device__ __forceinline__ T from_float(float val) {
  return static_cast<T>(val);
}

// Convert input value to FP8 with scaling (on-the-fly quantization)
__device__ __forceinline__ c10::Float8_e4m3fn quantize_to_fp8(float val,
                                                              float inv_scale) {
  float scaled = val * inv_scale;
  // Clamp to FP8 E4M3 range [-448, 448]
  scaled = fmaxf(-kFp8E4m3Max, fminf(scaled, kFp8E4m3Max));
#ifdef ENABLE_FP8
  return fp8::vec_conversion<c10::Float8_e4m3fn, float>(scaled);
#else
  return static_cast<c10::Float8_e4m3fn>(scaled);
#endif
}

// Convert FP8 to float
__device__ __forceinline__ float fp8_to_float(c10::Float8_e4m3fn val) {
  return static_cast<float>(val);
}

// =============================================================================
// Fused FP8 GEMV Kernel (M=1, optimized for decode)
// =============================================================================

template <typename InputT, typename OutputT, int BLOCK_DIM_X = 256>
__global__ void fused_fp8_gemv_kernel_impl(
    OutputT* __restrict__ output,                   // [1, N]
    const InputT* __restrict__ input,               // [1, K]
    const c10::Float8_e4m3fn* __restrict__ weight,  // [N, K]
    const float* __restrict__ input_scale,          // [1]
    const float* __restrict__ weight_scale,         // [1]
    const OutputT* __restrict__ bias,               // [N] or nullptr
    int N,
    int K) {
  // Each block computes one or more output elements
  // Use warp-level reduction for efficiency

  const int tid = threadIdx.x;
  const int n_idx = blockIdx.x;

  if (n_idx >= N) return;

  const float inv_input_scale = 1.0f / (*input_scale);
  const float combined_scale = (*input_scale) * (*weight_scale);

  // Pointer to the weight row for this output
  const c10::Float8_e4m3fn* weight_row = weight + n_idx * K;

  // Accumulate dot product
  float acc = 0.0f;

  // Vectorized load and compute
  constexpr int VEC_SIZE = 4;
  const int vec_k = K / VEC_SIZE;
  const int rem_k = K % VEC_SIZE;

  // Process vectorized elements
  for (int k = tid; k < vec_k; k += BLOCK_DIM_X) {
    const int base_k = k * VEC_SIZE;

// Load input values
#pragma unroll
    for (int v = 0; v < VEC_SIZE; ++v) {
      float in_val = to_float(input[base_k + v]);
      // On-the-fly quantization: convert input to FP8 equivalent value
      float quantized_in = roundf(in_val * inv_input_scale);
      quantized_in = fmaxf(-kFp8E4m3Max, fminf(quantized_in, kFp8E4m3Max));

      // Load FP8 weight and convert to float
      float w_val = fp8_to_float(weight_row[base_k + v]);

      // Accumulate
      acc += quantized_in * w_val;
    }
  }

  // Process remaining elements
  const int rem_start = vec_k * VEC_SIZE;
  for (int k = tid; k < rem_k; k += BLOCK_DIM_X) {
    float in_val = to_float(input[rem_start + k]);
    float quantized_in = roundf(in_val * inv_input_scale);
    quantized_in = fmaxf(-kFp8E4m3Max, fminf(quantized_in, kFp8E4m3Max));
    float w_val = fp8_to_float(weight_row[rem_start + k]);
    acc += quantized_in * w_val;
  }

  // Warp reduction
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xffffffff, acc, offset);
  }

  // Block reduction using shared memory
  __shared__ float shared_acc[32];  // One per warp
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;

  if (lane_id == 0) {
    shared_acc[warp_id] = acc;
  }
  __syncthreads();

  // Final reduction by first warp
  if (warp_id == 0) {
    acc = (lane_id < (BLOCK_DIM_X / 32)) ? shared_acc[lane_id] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    if (lane_id == 0) {
      // Apply combined scale
      float result = acc * combined_scale;

      // Add bias if present
      if (bias != nullptr) {
        result += to_float(bias[n_idx]);
      }

      output[n_idx] = from_float<OutputT>(result);
    }
  }
}

// =============================================================================
// Fused FP8 GEMM Kernel (small M, e.g., 2-64)
// =============================================================================

template <typename InputT,
          typename OutputT,
          int BLOCK_M = 16,
          int BLOCK_N = 64,
          int BLOCK_K = 64,
          int THREADS = 64>
__global__ void fused_fp8_gemm_kernel_impl(
    OutputT* __restrict__ output,                   // [M, N]
    const InputT* __restrict__ input,               // [M, K]
    const c10::Float8_e4m3fn* __restrict__ weight,  // [N, K]
    const float* __restrict__ input_scale,          // [1]
    const float* __restrict__ weight_scale,         // [1]
    const OutputT* __restrict__ bias,               // [N] or nullptr
    int M,
    int N,
    int K) {
  // Block computes BLOCK_M x BLOCK_N tile of output
  // Each thread handles BLOCK_N / THREADS columns

  static_assert(BLOCK_N % THREADS == 0, "BLOCK_N must be divisible by THREADS");
  constexpr int N_PER_THREAD = BLOCK_N / THREADS;

  const int bm = blockIdx.y;
  const int bn = blockIdx.x;

  const int m_start = bm * BLOCK_M;
  const int n_start = bn * BLOCK_N;

  // Thread indices within block
  const int tid = threadIdx.x;

  // Shared memory for input tile (quantized on-the-fly)
  __shared__ float smem_input[BLOCK_M][BLOCK_K];
  __shared__ float smem_weight[BLOCK_N][BLOCK_K];

  // Accumulator registers - use compile-time constant for array size
  float acc[BLOCK_M][N_PER_THREAD];
#pragma unroll
  for (int i = 0; i < BLOCK_M; ++i) {
#pragma unroll
    for (int j = 0; j < N_PER_THREAD; ++j) {
      acc[i][j] = 0.0f;
    }
  }

  const float inv_input_scale = 1.0f / (*input_scale);
  const float combined_scale = (*input_scale) * (*weight_scale);

  // Iterate over K dimension
  for (int k_start = 0; k_start < K; k_start += BLOCK_K) {
    // Load input tile with on-the-fly quantization
    for (int idx = tid; idx < BLOCK_M * BLOCK_K; idx += THREADS) {
      const int m_local = idx / BLOCK_K;
      const int k_local = idx % BLOCK_K;
      const int m_global = m_start + m_local;
      const int k_global = k_start + k_local;

      float val = 0.0f;
      if (m_global < M && k_global < K) {
        float in_val = to_float(input[m_global * K + k_global]);
        // On-the-fly quantization to FP8 equivalent
        val = roundf(in_val * inv_input_scale);
        val = fmaxf(-kFp8E4m3Max, fminf(val, kFp8E4m3Max));
      }
      smem_input[m_local][k_local] = val;
    }

    // Load weight tile (already in FP8, convert to float)
    for (int idx = tid; idx < BLOCK_N * BLOCK_K; idx += THREADS) {
      const int n_local = idx / BLOCK_K;
      const int k_local = idx % BLOCK_K;
      const int n_global = n_start + n_local;
      const int k_global = k_start + k_local;

      float val = 0.0f;
      if (n_global < N && k_global < K) {
        val = fp8_to_float(weight[n_global * K + k_global]);
      }
      smem_weight[n_local][k_local] = val;
    }

    __syncthreads();

    // Compute partial products
    const int n_local_start = tid * N_PER_THREAD;

#pragma unroll
    for (int m_local = 0; m_local < BLOCK_M; ++m_local) {
#pragma unroll
      for (int n_offset = 0; n_offset < N_PER_THREAD; ++n_offset) {
        const int n_local = n_local_start + n_offset;
#pragma unroll
        for (int k_local = 0; k_local < BLOCK_K; ++k_local) {
          acc[m_local][n_offset] +=
              smem_input[m_local][k_local] * smem_weight[n_local][k_local];
        }
      }
    }

    __syncthreads();
  }

  // Write output with scale and bias
  const int n_local_start = tid * N_PER_THREAD;

#pragma unroll
  for (int m_local = 0; m_local < BLOCK_M; ++m_local) {
    const int m_global = m_start + m_local;
    if (m_global >= M) continue;

#pragma unroll
    for (int n_offset = 0; n_offset < N_PER_THREAD; ++n_offset) {
      const int n_global = n_start + n_local_start + n_offset;
      if (n_global >= N) continue;

      float result = acc[m_local][n_offset] * combined_scale;

      if (bias != nullptr) {
        result += to_float(bias[n_global]);
      }

      output[m_global * N + n_global] = from_float<OutputT>(result);
    }
  }
}

// =============================================================================
// Dispatch functions
// =============================================================================

template <typename InputT, typename OutputT>
void launch_fused_fp8_gemv(OutputT* output,
                           const InputT* input,
                           const c10::Float8_e4m3fn* weight,
                           const float* input_scale,
                           const float* weight_scale,
                           const OutputT* bias,
                           int N,
                           int K,
                           cudaStream_t stream) {
  constexpr int BLOCK_DIM = 256;
  dim3 grid(N);
  dim3 block(BLOCK_DIM);

  fused_fp8_gemv_kernel_impl<InputT, OutputT, BLOCK_DIM>
      <<<grid, block, 0, stream>>>(
          output, input, weight, input_scale, weight_scale, bias, N, K);
}

template <typename InputT, typename OutputT>
void launch_fused_fp8_gemm(OutputT* output,
                           const InputT* input,
                           const c10::Float8_e4m3fn* weight,
                           const float* input_scale,
                           const float* weight_scale,
                           const OutputT* bias,
                           int M,
                           int N,
                           int K,
                           cudaStream_t stream) {
  constexpr int BLOCK_M = 16;
  constexpr int BLOCK_N = 64;
  constexpr int BLOCK_K = 64;
  constexpr int THREADS = 64;

  dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
  dim3 block(THREADS);

  fused_fp8_gemm_kernel_impl<InputT,
                             OutputT,
                             BLOCK_M,
                             BLOCK_N,
                             BLOCK_K,
                             THREADS><<<grid, block, 0, stream>>>(
      output, input, weight, input_scale, weight_scale, bias, M, N, K);
}

// =============================================================================
// Public API
// =============================================================================

torch::Tensor fused_fp8_linear(const torch::Tensor& input,         // [M, K]
                               const torch::Tensor& weight,        // [N, K] FP8
                               const torch::Tensor& input_scale,   // [1]
                               const torch::Tensor& weight_scale,  // [1]
                               const std::optional<torch::Tensor>& bias,
                               at::ScalarType output_dtype) {
  // Input validation
  TORCH_CHECK(input.dim() == 2, "Input must be 2D tensor [M, K]");
  TORCH_CHECK(weight.dim() == 2, "Weight must be 2D tensor [N, K]");
  TORCH_CHECK(weight.dtype() == torch::kFloat8_e4m3fn,
              "Weight must be FP8 E4M3 format");
  TORCH_CHECK(input.size(1) == weight.size(1),
              "Input K dimension must match weight K dimension");
  TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");
  TORCH_CHECK(input_scale.numel() == 1,
              "input_scale must be scalar (per-tensor)");
  TORCH_CHECK(weight_scale.numel() == 1,
              "weight_scale must be scalar (per-tensor)");

  const int64_t M = input.size(0);
  const int64_t N = weight.size(0);
  const int64_t K = input.size(1);

  // Allocate output tensor
  auto output = torch::empty({M, N}, input.options().dtype(output_dtype));

  // Get bias pointer if present
  const void* bias_ptr = nullptr;
  if (bias.has_value() && bias.value().defined()) {
    TORCH_CHECK(bias.value().numel() == N, "Bias must have N elements");
    TORCH_CHECK(bias.value().dtype() == output_dtype,
                "Bias dtype must match output dtype");
    bias_ptr = bias.value().data_ptr();
  }

  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(input));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Dispatch based on input/output dtype
  if (input.dtype() == torch::kBFloat16) {
    if (output_dtype == torch::kBFloat16) {
      if (M == 1) {
        launch_fused_fp8_gemv<at::BFloat16, at::BFloat16>(
            static_cast<at::BFloat16*>(output.data_ptr()),
            static_cast<const at::BFloat16*>(input.data_ptr()),
            static_cast<const c10::Float8_e4m3fn*>(weight.data_ptr()),
            static_cast<const float*>(input_scale.data_ptr()),
            static_cast<const float*>(weight_scale.data_ptr()),
            static_cast<const at::BFloat16*>(bias_ptr),
            N,
            K,
            stream);
      } else {
        launch_fused_fp8_gemm<at::BFloat16, at::BFloat16>(
            static_cast<at::BFloat16*>(output.data_ptr()),
            static_cast<const at::BFloat16*>(input.data_ptr()),
            static_cast<const c10::Float8_e4m3fn*>(weight.data_ptr()),
            static_cast<const float*>(input_scale.data_ptr()),
            static_cast<const float*>(weight_scale.data_ptr()),
            static_cast<const at::BFloat16*>(bias_ptr),
            M,
            N,
            K,
            stream);
      }
    } else {
      TORCH_CHECK(output_dtype == torch::kFloat16,
                  "Output dtype must be BFloat16 or Float16");
      if (M == 1) {
        launch_fused_fp8_gemv<at::BFloat16, at::Half>(
            static_cast<at::Half*>(output.data_ptr()),
            static_cast<const at::BFloat16*>(input.data_ptr()),
            static_cast<const c10::Float8_e4m3fn*>(weight.data_ptr()),
            static_cast<const float*>(input_scale.data_ptr()),
            static_cast<const float*>(weight_scale.data_ptr()),
            static_cast<const at::Half*>(bias_ptr),
            N,
            K,
            stream);
      } else {
        launch_fused_fp8_gemm<at::BFloat16, at::Half>(
            static_cast<at::Half*>(output.data_ptr()),
            static_cast<const at::BFloat16*>(input.data_ptr()),
            static_cast<const c10::Float8_e4m3fn*>(weight.data_ptr()),
            static_cast<const float*>(input_scale.data_ptr()),
            static_cast<const float*>(weight_scale.data_ptr()),
            static_cast<const at::Half*>(bias_ptr),
            M,
            N,
            K,
            stream);
      }
    }
  } else if (input.dtype() == torch::kFloat16) {
    if (output_dtype == torch::kFloat16) {
      if (M == 1) {
        launch_fused_fp8_gemv<at::Half, at::Half>(
            static_cast<at::Half*>(output.data_ptr()),
            static_cast<const at::Half*>(input.data_ptr()),
            static_cast<const c10::Float8_e4m3fn*>(weight.data_ptr()),
            static_cast<const float*>(input_scale.data_ptr()),
            static_cast<const float*>(weight_scale.data_ptr()),
            static_cast<const at::Half*>(bias_ptr),
            N,
            K,
            stream);
      } else {
        launch_fused_fp8_gemm<at::Half, at::Half>(
            static_cast<at::Half*>(output.data_ptr()),
            static_cast<const at::Half*>(input.data_ptr()),
            static_cast<const c10::Float8_e4m3fn*>(weight.data_ptr()),
            static_cast<const float*>(input_scale.data_ptr()),
            static_cast<const float*>(weight_scale.data_ptr()),
            static_cast<const at::Half*>(bias_ptr),
            M,
            N,
            K,
            stream);
      }
    } else {
      TORCH_CHECK(output_dtype == torch::kBFloat16,
                  "Output dtype must be BFloat16 or Float16");
      if (M == 1) {
        launch_fused_fp8_gemv<at::Half, at::BFloat16>(
            static_cast<at::BFloat16*>(output.data_ptr()),
            static_cast<const at::Half*>(input.data_ptr()),
            static_cast<const c10::Float8_e4m3fn*>(weight.data_ptr()),
            static_cast<const float*>(input_scale.data_ptr()),
            static_cast<const float*>(weight_scale.data_ptr()),
            static_cast<const at::BFloat16*>(bias_ptr),
            N,
            K,
            stream);
      } else {
        launch_fused_fp8_gemm<at::Half, at::BFloat16>(
            static_cast<at::BFloat16*>(output.data_ptr()),
            static_cast<const at::Half*>(input.data_ptr()),
            static_cast<const c10::Float8_e4m3fn*>(weight.data_ptr()),
            static_cast<const float*>(input_scale.data_ptr()),
            static_cast<const float*>(weight_scale.data_ptr()),
            static_cast<const at::BFloat16*>(bias_ptr),
            M,
            N,
            K,
            stream);
      }
    }
  } else {
    TORCH_CHECK(false, "Input dtype must be BFloat16 or Float16");
  }

  return output;
}

// Threshold for using fused kernel vs CUTLASS
// For M > threshold, CUTLASS is generally faster due to better tiling
constexpr int FUSED_KERNEL_M_THRESHOLD = 64;

bool should_use_fused_kernel(int M, int N, int K) {
  // Use fused kernel for small M (decode scenario)
  // For larger M, CUTLASS's optimized tiling is more efficient
  return M <= FUSED_KERNEL_M_THRESHOLD;
}

}  // namespace cuda
}  // namespace kernel
}  // namespace xllm
