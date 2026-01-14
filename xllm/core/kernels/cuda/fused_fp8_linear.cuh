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

#pragma once

#include <c10/util/Float8_e4m3fn.h>

#include <cstdint>

namespace xllm {
namespace kernel {
namespace cuda {

// Maximum value for FP8 E4M3 format
constexpr float kFp8E4m3Max = 448.0f;

/**
 * Fused FP8 quantization + scaled GEMM kernel for small batch sizes.
 * Optimized for decode scenario where M is small (typically 1-64).
 *
 * Performs: output = (quantize(input, input_scale) @ weight.T) * input_scale *
 * weight_scale
 *
 * This kernel fuses quantization with GEMM to:
 * 1. Eliminate intermediate FP8 tensor allocation
 * 2. Reduce global memory bandwidth by ~50% for A matrix
 * 3. Reduce kernel launch overhead
 *
 * @tparam InputT Input tensor element type (half or bfloat16)
 * @tparam OutputT Output tensor element type (half or bfloat16)
 * @tparam BLOCK_M Tile size in M dimension
 * @tparam BLOCK_N Tile size in N dimension
 * @tparam BLOCK_K Tile size in K dimension
 */
template <typename InputT,
          typename OutputT,
          int BLOCK_M = 16,
          int BLOCK_N = 64,
          int BLOCK_K = 64>
__global__ void fused_fp8_gemm_kernel(
    OutputT* __restrict__ output,                   // [M, N]
    const InputT* __restrict__ input,               // [M, K]
    const c10::Float8_e4m3fn* __restrict__ weight,  // [N, K] in FP8
    const float* __restrict__ input_scale,          // [1] per-tensor scale
    const float* __restrict__ weight_scale,         // [1] per-tensor scale
    const OutputT* __restrict__ bias,               // [N] or nullptr
    int M,
    int N,
    int K);

/**
 * Fused FP8 GEMV kernel for single token decode (M=1).
 * Highly optimized for the most common decode case.
 *
 * @tparam InputT Input tensor element type
 * @tparam OutputT Output tensor element type
 */
template <typename InputT, typename OutputT>
__global__ void fused_fp8_gemv_kernel(
    OutputT* __restrict__ output,                   // [1, N]
    const InputT* __restrict__ input,               // [1, K]
    const c10::Float8_e4m3fn* __restrict__ weight,  // [N, K]
    const float* __restrict__ input_scale,          // [1]
    const float* __restrict__ weight_scale,         // [1]
    const OutputT* __restrict__ bias,               // [N] or nullptr
    int N,
    int K);

}  // namespace cuda
}  // namespace kernel
}  // namespace xllm
