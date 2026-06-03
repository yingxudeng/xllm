/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "framework/kv_cache/kv_cache_utils.h"

#include <gtest/gtest.h>

#include "common/global_flags.h"
#include "common/options.h"

namespace xllm {
namespace {

// Padding slot reserved by LinearStateSlotPool (id 0). Kept in sync with the
// kPaddingLinearStateBlocks constant in kv_cache_utils.cpp.
constexpr int64_t kPadding = 2;

// A budget large enough that the balanced max_blocks cap never binds in the
// concurrency-sized cases below.
constexpr int64_t kLargeCacheBytes = 64LL << 30;  // 64 GiB
constexpr int64_t kLinearSlotSize = 2LL << 20;    // 2 MiB per slot
constexpr int64_t kNumLinearLayers = 12;
constexpr int64_t kNumFullLayers = 12;
constexpr int64_t kFullBlockBytes = 1LL << 20;  // 1 MiB per full block

class LinearStateBlocksTest : public ::testing::Test {
 protected:
  void SetUp() override {
    saved_max_concurrent_ = FLAGS_max_concurrent_requests;
  }
  void TearDown() override {
    FLAGS_max_concurrent_requests = saved_max_concurrent_;
  }

  int64_t saved_max_concurrent_ = 0;
};

}  // namespace

// Without prefix caching the pool is sized to the live slots bounded by the
// rate limiter: max_concurrent_requests + padding.
TEST_F(LinearStateBlocksTest, NoPrefixCacheSizesToConcurrency) {
  FLAGS_max_concurrent_requests = 200;
  // max_linear_state_cache_slots defaults to 0.
  LinearStateCacheOptions options;

  const int64_t blocks =
      calculate_linear_state_blocks(kLargeCacheBytes,
                                    kNumLinearLayers,
                                    kLinearSlotSize,
                                    kNumFullLayers,
                                    kFullBlockBytes,
                                    options,
                                    /*enable_prefix_cache=*/false);

  EXPECT_EQ(blocks, 200 + kPadding);
}

// With prefix caching the auto memory-ratio path is used, which on a large
// budget allocates far more blocks than the concurrency-based sizing.
TEST_F(LinearStateBlocksTest, PrefixCacheUsesMemoryRatio) {
  FLAGS_max_concurrent_requests = 200;
  LinearStateCacheOptions options;

  const int64_t blocks =
      calculate_linear_state_blocks(kLargeCacheBytes,
                                    kNumLinearLayers,
                                    kLinearSlotSize,
                                    kNumFullLayers,
                                    kFullBlockBytes,
                                    options,
                                    /*enable_prefix_cache=*/true);

  EXPECT_GT(blocks, 200 + kPadding);
}

// An explicit max_linear_state_cache_slots overrides both paths regardless of
// the prefix-cache flag.
TEST_F(LinearStateBlocksTest, ExplicitSlotsOverrideConcurrency) {
  FLAGS_max_concurrent_requests = 200;
  LinearStateCacheOptions options;
  options.max_linear_state_cache_slots(32);

  for (bool enable_prefix_cache : {false, true}) {
    const int64_t blocks = calculate_linear_state_blocks(kLargeCacheBytes,
                                                         kNumLinearLayers,
                                                         kLinearSlotSize,
                                                         kNumFullLayers,
                                                         kFullBlockBytes,
                                                         options,
                                                         enable_prefix_cache);
    EXPECT_EQ(blocks, 32 + kPadding);
  }
}

// When the budget cannot fit the requested concurrency, the balanced max_blocks
// cap binds so full-attention KV cache is not starved.
TEST_F(LinearStateBlocksTest, NoPrefixCacheCappedByBudget) {
  FLAGS_max_concurrent_requests = 1000000;  // absurdly high
  LinearStateCacheOptions options;

  const int64_t blocks =
      calculate_linear_state_blocks(kLargeCacheBytes,
                                    kNumLinearLayers,
                                    kLinearSlotSize,
                                    kNumFullLayers,
                                    kFullBlockBytes,
                                    options,
                                    /*enable_prefix_cache=*/false);

  EXPECT_LT(blocks, FLAGS_max_concurrent_requests + kPadding);
  EXPECT_GE(blocks, kPadding);
}

// max_concurrent_requests == 0 (unlimited) mirrors the full-attention
// single-block pool sizing: it collapses to the padding floor.
TEST_F(LinearStateBlocksTest, UnlimitedConcurrencyFallsBackToPadding) {
  FLAGS_max_concurrent_requests = 0;
  LinearStateCacheOptions options;

  const int64_t blocks =
      calculate_linear_state_blocks(kLargeCacheBytes,
                                    kNumLinearLayers,
                                    kLinearSlotSize,
                                    kNumFullLayers,
                                    kFullBlockBytes,
                                    options,
                                    /*enable_prefix_cache=*/false);

  EXPECT_EQ(blocks, kPadding);
}

}  // namespace xllm
