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

/**
 * SM120 (Blackwell) FP8 GEMM config inference utilities.
 *
 * Benchmark-only: used by cutlass_scaled_mm_sm120_benchmark and
 * cutlass_scaled_mm_sm120_config_analysis. Must match the dispatch logic in
 * scaled_mm_sm120_fp8_dispatch.cuh exactly.
 */

#pragma once

#include <cstdint>
#include <string>

#include "cutlass_extensions/common.hpp"

namespace xllm {
namespace kernel {
namespace cuda {

// Constants aligned with SM120DispatchConfig in
// scaled_mm_sm120_fp8_dispatch.cuh
namespace sm120_config {

constexpr int64_t kLargeN = 8192;
constexpr int64_t kLargeK = 4096;
constexpr float kEfficiencyMargin = 0.05f;
constexpr int kDefaultSMCount = 128;

}  // namespace sm120_config

/**
 * Compute wave efficiency for a given tile configuration.
 * Wave efficiency = actual_tiles / (num_waves * num_sms), in range [0, 1].
 */
inline float compute_wave_efficiency(int64_t m,
                                     int64_t n,
                                     int tile_m,
                                     int tile_n,
                                     int num_sms) {
  int tiles_m = static_cast<int>((m + tile_m - 1) / tile_m);
  int tiles_n = static_cast<int>((n + tile_n - 1) / tile_n);
  int total_tiles = tiles_m * tiles_n;
  int num_waves = (total_tiles + num_sms - 1) / num_sms;
  return static_cast<float>(total_tiles) /
         static_cast<float>(num_waves * num_sms);
}

/**
 * Get config name from (M, N, K). Matches dispatch logic in
 * scaled_mm_sm120_fp8_dispatch.cuh.
 *
 * Strategy:
 * 1. M <= 16: 128x32_swap
 * 2. M <= 128: 128x64_swap if N >= 8192 or K >= 4096, else 128x64
 * 3. M > 128: 128x128 if eff_128x128 >= eff_128x64 - 0.05, else 128x64
 *
 * @param num_sms If 0, uses get_device_sm_count() (or kDefaultSMCount
 * fallback).
 */
inline std::string get_sm120_config_name(int64_t m,
                                         int64_t n,
                                         int64_t k,
                                         int num_sms = 0) {
  if (num_sms <= 0) {
    num_sms = xllm::kernel::cuda::get_device_sm_count();
    if (num_sms <= 0) num_sms = sm120_config::kDefaultSMCount;
  }

  if (m <= 16) {
    return "128x32_swap";
  }
  if (m <= 128) {
    if (n >= sm120_config::kLargeN || k >= sm120_config::kLargeK) {
      return "128x64_swap";
    }
    return "128x64";
  }
  // M > 128: wave efficiency based selection
  float eff_128x128 = compute_wave_efficiency(m, n, 128, 128, num_sms);
  float eff_128x64 = compute_wave_efficiency(m, n, 128, 64, num_sms);
  if (eff_128x128 >= eff_128x64 - sm120_config::kEfficiencyMargin) {
    return "128x128";
  }
  return "128x64";
}

/**
 * Get TileShape description for a config name.
 * Format: MxNxK, matching Shape<_M, _N, _K> in dispatch.cuh.
 */
inline std::string get_sm120_tile_shape_desc(const std::string& config_name) {
  if (config_name == "128x32_swap") {
    return "128x32x128";
  }
  if (config_name == "128x64_swap" || config_name == "128x64") {
    return "128x64x128";
  }
  return "128x128x128";  // 128x128
}

/**
 * Check if config uses swap_ab.
 */
inline bool is_swap_ab_config(const std::string& config_name) {
  return config_name.find("_swap") != std::string::npos;
}

}  // namespace cuda
}  // namespace kernel
}  // namespace xllm
