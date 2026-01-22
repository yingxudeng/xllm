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

// ref to:
// https://github.com/vllm-project/vllm/blob/main/csrc/quantization/cutlass_w8a8/c3x/scaled_mm_sm120_fp8_dispatch.cuh

#pragma once

// clang-format off
#include "cutlass_gemm_caller.cuh"
#include "scaled_mm.cuh"
#include "cutlass_extensions/common.hpp"
#include "cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp"
// clang-format on

// SM120 (Blackwell) FP8 GEMM kernel configurations.
// Uses swap_ab technique for small M and wave efficiency optimization for tile selection.

namespace xllm {
namespace kernel {
namespace cuda {

using c3x::cutlass_gemm_caller;

// =============================================================================
// Dispatch Configuration Constants
// =============================================================================
struct SM120DispatchConfig {
  // M dimension thresholds for dispatch decisions:
  // - M <= kSmallM: use 128x32 tile with swap_ab
  // - M <= kLargeM: use 128x64 tile, swap_ab depends on N/K size
  // - M > kLargeM: use wave efficiency to select 128x64 vs 128x128
  static constexpr uint32_t kSmallM = 16;
  static constexpr uint32_t kLargeM = 128;

  // N/K thresholds for swap_ab decision in medium M range (kSmallM < M <= kLargeM)
  static constexpr uint32_t kLargeN = 8192;
  static constexpr uint32_t kLargeK = 4096;

  // Wave efficiency margin: prefer 128x128 when its efficiency is within
  // this margin of 128x64's efficiency (128x128 has better arithmetic intensity)
  static constexpr float kEfficiencyMargin = 0.05f;

  // Default SM count fallback when cudaDeviceGetAttribute fails
  static constexpr uint32_t kDefaultSMCount = 128;
};

// =============================================================================
// SM120 FP8 GEMM Kernel with swap_ab Support
// =============================================================================
// swap_ab transforms C[M,N] = A[M,K] * B[K,N] into C^T[N,M] = B^T * A^T,
// which improves efficiency for small M by turning it into small N.
template <typename ElementAB_,
          typename ElementD_,
          template <typename, typename, typename> typename Epilogue_,
          typename TileShape,
          typename ClusterShape,
          typename KernelSchedule,
          typename EpilogueSchedule,
          bool swap_ab_ = false>
struct cutlass_3x_gemm_sm120_fp8 {
  // Type aliases
  using ElementAB = ElementAB_;
  using ElementC = ElementD_;
  using ElementD = ElementD_;
  using ElementAcc =
      std::conditional_t<std::is_same_v<ElementAB, int8_t>, int32_t, float>;

  using Epilogue = Epilogue_<ElementAcc, ElementD, TileShape>;
  using EVTCompute = typename Epilogue::EVTCompute;

  // Alignment constants
  static constexpr int AlignmentAB =
      128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int AlignmentCD =
      128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr bool swap_ab = swap_ab_;

  // Layout definitions
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::RowMajor;
  using LayoutC = LayoutD;

  using LayoutA_T = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
  using LayoutB_T = typename cutlass::layout::LayoutTranspose<LayoutB>::type;
  using LayoutC_T = typename cutlass::layout::LayoutTranspose<LayoutC>::type;
  using LayoutD_T = typename cutlass::layout::LayoutTranspose<LayoutD>::type;

  // Collective epilogue with conditional layout swap
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm120,
          cutlass::arch::OpClassTensorOp,
          TileShape,
          ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAcc,
          float,
          ElementC,
          conditional_t<swap_ab, LayoutC_T, LayoutC>,
          AlignmentCD,
          ElementD,
          conditional_t<swap_ab, LayoutD_T, LayoutD>,
          AlignmentCD,
          EpilogueSchedule,
          EVTCompute>::CollectiveOp;

  static constexpr size_t CEStorageSize =
      sizeof(typename CollectiveEpilogue::SharedStorage);
  using Stages = typename cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(CEStorageSize)>;

  // Collective mainloop with conditional operand swap
  using CollectiveMainloop = conditional_t<
      swap_ab,
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
          ElementAB, LayoutB_T, AlignmentAB,
          ElementAB, LayoutA_T, AlignmentAB,
          ElementAcc, TileShape, ClusterShape, Stages,
          KernelSchedule>::CollectiveOp,
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
          ElementAB, LayoutA, AlignmentAB,
          ElementAB, LayoutB, AlignmentAB,
          ElementAcc, TileShape, ClusterShape, Stages,
          KernelSchedule>::CollectiveOp>;

  // Final kernel definition
  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
                                           CollectiveMainloop,
                                           CollectiveEpilogue,
                                           void>;
};

// =============================================================================
// SM120 FP8 Generic Configuration Template
// =============================================================================
// Parameterizes TileShape and swap_ab to eliminate code duplication.
// Note: SM120 FP8 requires TileM >= 128, ClusterShape must be 1x1x1.
template <typename InType,
          typename OutType,
          bool EnableBias,
          int TileM_,
          int TileN_,
          int TileK_,
          bool SwapAB_>
struct sm120_fp8_config_generic {
  static_assert(std::is_same_v<InType, cutlass::float_e4m3_t>,
                "InType must be float_e4m3_t");

  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using TileShape =
      Shape<cute::Int<TileM_>, cute::Int<TileN_>, cute::Int<TileK_>>;
  using ClusterShape = Shape<_1, _1, _1>;

  static constexpr bool SwapAB = SwapAB_;

  // Epilogue selection: bias+swap -> ColumnBias, bias -> RowBias, none -> NoOp
  using Cutlass3xGemm = conditional_t<
      EnableBias,
      conditional_t<
          SwapAB,
          cutlass_3x_gemm_sm120_fp8<InType, OutType, c3x::ScaledEpilogueColumnBias,
                                    TileShape, ClusterShape, KernelSchedule,
                                    EpilogueSchedule, SwapAB>,
          cutlass_3x_gemm_sm120_fp8<InType, OutType, c3x::ScaledEpilogueBias,
                                    TileShape, ClusterShape, KernelSchedule,
                                    EpilogueSchedule, SwapAB>>,
      cutlass_3x_gemm_sm120_fp8<InType, OutType, c3x::ScaledEpilogue,
                                TileShape, ClusterShape, KernelSchedule,
                                EpilogueSchedule, SwapAB>>;
};

// =============================================================================
// SM120 FP8 Configuration Type Aliases
// =============================================================================
// Four configurations covering all problem shapes:
// - Default (128x128x128): Large M with good wave efficiency
// - Tile128x64: Medium M or better wave efficiency than 128x128
// - Tile128x64Swap: Small-medium M with large N/K
// - Tile128x32Swap: Very small M (<=16)

template <typename InType, typename OutType, bool EnableBias>
using sm120_fp8_config_default =
    sm120_fp8_config_generic<InType, OutType, EnableBias, 128, 128, 128, false>;

template <typename InType, typename OutType, bool EnableBias>
using sm120_fp8_config_tile_128x64 =
    sm120_fp8_config_generic<InType, OutType, EnableBias, 128, 64, 128, false>;

template <typename InType, typename OutType, bool EnableBias>
using sm120_fp8_config_tile_128x64_swap =
    sm120_fp8_config_generic<InType, OutType, EnableBias, 128, 64, 128, true>;

template <typename InType, typename OutType, bool EnableBias>
using sm120_fp8_config_tile_128x32_swap =
    sm120_fp8_config_generic<InType, OutType, EnableBias, 128, 32, 128, true>;

// =============================================================================
// Wave Efficiency Utilities
// =============================================================================
// Wave efficiency = actual_tiles / (num_waves * num_sms), in range [0, 1].
// Higher efficiency means better SM utilization.
inline float compute_wave_efficiency(uint32_t m, uint32_t n,
                                     uint32_t tile_m, uint32_t tile_n,
                                     uint32_t num_sms) {
  uint32_t tiles_m = (m + tile_m - 1) / tile_m;
  uint32_t tiles_n = (n + tile_n - 1) / tile_n;
  uint32_t total_tiles = tiles_m * tiles_n;
  uint32_t num_waves = (total_tiles + num_sms - 1) / num_sms;
  return static_cast<float>(total_tiles) /
         static_cast<float>(num_waves * num_sms);
}

// Select between 128x64 and 128x128 tiles based on wave efficiency.
// Prefers 128x128 when efficiencies are similar (better arithmetic intensity).
inline bool should_use_tile_128x128(uint32_t m, uint32_t n, uint32_t num_sms) {
  float eff_128x128 = compute_wave_efficiency(m, n, 128, 128, num_sms);
  float eff_128x64 = compute_wave_efficiency(m, n, 128, 64, num_sms);
  return eff_128x128 >= eff_128x64 - SM120DispatchConfig::kEfficiencyMargin;
}

// =============================================================================
// SM120 FP8 GEMM Caller with swap_ab Support
// =============================================================================
template <typename Gemm, typename... EpilogueArgs>
void cutlass_gemm_caller_sm120_fp8(torch::Tensor& out,
                                   torch::Tensor const& a,
                                   torch::Tensor const& b,
                                   EpilogueArgs&&... epilogue_params) {
  constexpr bool swap_ab = Gemm::swap_ab;
  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;
  using GemmKernel = typename Gemm::GemmKernel;
  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;

  const int32_t m = a.size(0);
  const int32_t n = b.size(1);
  const int32_t k = a.size(1);

  // Problem shape: swap M and N dimensions when using swap_ab
  auto prob_shape = swap_ab ? cute::make_shape(n, m, k, 1)
                            : cute::make_shape(m, n, k, 1);

  auto a_stride =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  auto b_stride =
      cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  auto c_stride = cutlass::make_cute_packed_stride(
      StrideC{}, swap_ab ? cute::make_shape(n, m, 1) : cute::make_shape(m, n, 1));

  auto* a_ptr = static_cast<ElementAB*>(a.data_ptr());
  auto* b_ptr = static_cast<ElementAB*>(b.data_ptr());
  auto* c_ptr = static_cast<ElementD*>(out.data_ptr());

  // Swap operands A and B when swap_ab is enabled
  typename GemmKernel::MainloopArguments mainloop_args =
      swap_ab ? typename GemmKernel::MainloopArguments{b_ptr, b_stride,
                                                       a_ptr, a_stride}
              : typename GemmKernel::MainloopArguments{a_ptr, a_stride,
                                                       b_ptr, b_stride};

  typename GemmKernel::EpilogueArguments epilogue_args{
      Gemm::Epilogue::prepare_args(
          std::forward<EpilogueArgs>(epilogue_params)...),
      c_ptr, c_stride, c_ptr, c_stride};

  c3x::cutlass_gemm_caller<GemmKernel>(a.device(), prob_shape, mainloop_args,
                                       epilogue_args);
}

// =============================================================================
// SM120 FP8 Dispatch: Select Optimal Config Based on Problem Shape
// =============================================================================
// Strategy:
// 1. Small M (<=16): swap_ab + 128x32 tile
// 2. Medium M (17-128) with large N/K: swap_ab + 128x64 tile
// 3. Medium M (17-128) with small N/K: 128x64 tile (no swap)
// 4. Large M (>128): Select 128x64 vs 128x128 based on wave efficiency

namespace detail {

// Get cached SM count (thread-safe initialization)
inline uint32_t get_cached_sm_count() {
  static uint32_t num_sms = []() {
    uint32_t count = static_cast<uint32_t>(get_device_sm_count());
    return count > 0 ? count : SM120DispatchConfig::kDefaultSMCount;
  }();
  return num_sms;
}

// Check if swap_ab should be used for given M, N, K
inline bool should_use_swap_ab(uint32_t m, uint32_t n, uint32_t k) {
  if (m <= SM120DispatchConfig::kSmallM) return true;
  if (m <= SM120DispatchConfig::kLargeM) {
    return n >= SM120DispatchConfig::kLargeN ||
           k >= SM120DispatchConfig::kLargeK;
  }
  return false;
}

}  // namespace detail

template <typename InType,
          typename OutType,
          bool EnableBias,
          typename... EpilogueArgs>
inline void cutlass_gemm_sm120_fp8_dispatch(torch::Tensor& out,
                                            torch::Tensor const& a,
                                            torch::Tensor const& b,
                                            torch::Tensor const& a_scales,
                                            torch::Tensor const& b_scales,
                                            EpilogueArgs&&... args) {
  static_assert(std::is_same_v<InType, cutlass::float_e4m3_t>);
  TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn);

  // Kernel configurations
  using GemmDefault =
      typename sm120_fp8_config_default<InType, OutType, EnableBias>::Cutlass3xGemm;
  using Gemm128x64 =
      typename sm120_fp8_config_tile_128x64<InType, OutType, EnableBias>::Cutlass3xGemm;
  using Gemm128x64Swap =
      typename sm120_fp8_config_tile_128x64_swap<InType, OutType, EnableBias>::Cutlass3xGemm;
  using Gemm128x32Swap =
      typename sm120_fp8_config_tile_128x32_swap<InType, OutType, EnableBias>::Cutlass3xGemm;

  const uint32_t m = a.size(0);
  const uint32_t n = b.size(1);
  const uint32_t k = a.size(1);

  // Case 1: Very small M - use 128x32 with swap_ab
  if (m <= SM120DispatchConfig::kSmallM) {
    return cutlass_gemm_caller_sm120_fp8<Gemm128x32Swap>(
        out, a, b, b_scales, a_scales, std::forward<EpilogueArgs>(args)...);
  }

  // Case 2: Small-medium M - decide swap_ab based on N/K
  if (m <= SM120DispatchConfig::kLargeM) {
    if (detail::should_use_swap_ab(m, n, k)) {
      return cutlass_gemm_caller_sm120_fp8<Gemm128x64Swap>(
          out, a, b, b_scales, a_scales, std::forward<EpilogueArgs>(args)...);
    }
    return cutlass_gemm_caller_sm120_fp8<Gemm128x64>(
        out, a, b, a_scales, b_scales, std::forward<EpilogueArgs>(args)...);
  }

  // Case 3: Large M - select tile based on wave efficiency
  const uint32_t num_sms = detail::get_cached_sm_count();
  if (should_use_tile_128x128(m, n, num_sms)) {
    return cutlass_gemm_caller_sm120_fp8<GemmDefault>(
        out, a, b, a_scales, b_scales, std::forward<EpilogueArgs>(args)...);
  }
  return cutlass_gemm_caller_sm120_fp8<Gemm128x64>(
      out, a, b, a_scales, b_scales, std::forward<EpilogueArgs>(args)...);
}

// =============================================================================
// SM120 FP8 Scaled MM Entry Point
// =============================================================================
template <bool EnableBias, typename... EpilogueArgs>
void cutlass_scaled_mm_sm120_fp8_epilogue(torch::Tensor& out,
                                          torch::Tensor const& a,
                                          torch::Tensor const& b,
                                          torch::Tensor const& a_scales,
                                          torch::Tensor const& b_scales,
                                          EpilogueArgs&&... epilogue_args) {
  TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn);

  if (out.dtype() == torch::kBFloat16) {
    cutlass_gemm_sm120_fp8_dispatch<cutlass::float_e4m3_t, cutlass::bfloat16_t,
                                    EnableBias>(
        out, a, b, a_scales, b_scales,
        std::forward<EpilogueArgs>(epilogue_args)...);
  } else {
    TORCH_CHECK(out.dtype() == torch::kFloat16);
    cutlass_gemm_sm120_fp8_dispatch<cutlass::float_e4m3_t, cutlass::half_t,
                                    EnableBias>(
        out, a, b, a_scales, b_scales,
        std::forward<EpilogueArgs>(epilogue_args)...);
  }
}

}  // namespace cuda
}  // namespace kernel
}  // namespace xllm