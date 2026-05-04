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

#include <limits>

namespace xllm {
namespace {

TEST(KVCacheUtilsTest, AutoLinearStateBlocksAreMemoryRatioDerived) {
  LinearStateCacheOptions options;
  options.linear_state_full_kv_memory_ratio(1.0);

  EXPECT_EQ(calculate_linear_state_blocks(/*cache_size_in_bytes=*/10000,
                                          /*num_linear_attention_layers=*/1,
                                          /*linear_slot_size=*/100,
                                          /*num_full_attention_layers=*/1,
                                          /*full_attention_block_size=*/100,
                                          options),
            50);
}

TEST(KVCacheUtilsTest, AutoLinearStateBlocksUseRatioWhenSlotsUnset) {
  LinearStateCacheOptions options;
  options.max_linear_state_cache_slots(0).linear_state_full_kv_memory_ratio(
      0.5);

  EXPECT_EQ(calculate_linear_state_blocks(/*cache_size_in_bytes=*/12000,
                                          /*num_linear_attention_layers=*/1,
                                          /*linear_slot_size=*/100,
                                          /*num_full_attention_layers=*/1,
                                          /*full_attention_block_size=*/100,
                                          options),
            40);
}

TEST(KVCacheUtilsTest, ExplicitLinearStateBlocksUseSlotCapacity) {
  LinearStateCacheOptions options;
  options.max_linear_state_cache_slots(12);

  EXPECT_EQ(calculate_linear_state_blocks(/*cache_size_in_bytes=*/10000,
                                          /*num_linear_attention_layers=*/1,
                                          /*linear_slot_size=*/100,
                                          /*num_full_attention_layers=*/1,
                                          /*full_attention_block_size=*/100,
                                          options),
            14);
}

TEST(KVCacheUtilsTest, ExplicitLinearStateBlocksAreBoundedByKvBudget) {
  LinearStateCacheOptions options;
  options.max_linear_state_cache_slots(1024);

  EXPECT_EQ(calculate_linear_state_blocks(/*cache_size_in_bytes=*/10000,
                                          /*num_linear_attention_layers=*/1,
                                          /*linear_slot_size=*/100,
                                          /*num_full_attention_layers=*/1,
                                          /*full_attention_block_size=*/100,
                                          options),
            51);
}

TEST(KVCacheUtilsTest, AutoLinearStateBlocksAreBoundedByKvBudget) {
  LinearStateCacheOptions options;
  options.linear_state_full_kv_memory_ratio(1.0);

  EXPECT_EQ(calculate_linear_state_blocks(/*cache_size_in_bytes=*/10000,
                                          /*num_linear_attention_layers=*/1,
                                          /*linear_slot_size=*/100,
                                          /*num_full_attention_layers=*/1,
                                          /*full_attention_block_size=*/100,
                                          options),
            50);
}

TEST(KVCacheUtilsTest, LinearStateCacheRatioMustBeFinite) {
  LinearStateCacheOptions options;
  options.linear_state_full_kv_memory_ratio(
      std::numeric_limits<double>::infinity());

  EXPECT_DEATH(validate_linear_state_cache_options(options), "must be finite");
}

TEST(KVCacheUtilsTest, MaxLinearStateCacheSlotsMustBeNonNegative) {
  LinearStateCacheOptions options;
  options.max_linear_state_cache_slots(-1);

  EXPECT_DEATH(validate_linear_state_cache_options(options),
               "must be greater than or equal to 0");
}

TEST(KVCacheUtilsTest, AutoLinearStateBlocksHandlesVeryLargeFiniteRatio) {
  LinearStateCacheOptions options;
  options.linear_state_full_kv_memory_ratio(std::numeric_limits<double>::max());

  EXPECT_EQ(calculate_linear_state_blocks(/*cache_size_in_bytes=*/10000,
                                          /*num_linear_attention_layers=*/1,
                                          /*linear_slot_size=*/100,
                                          /*num_full_attention_layers=*/1,
                                          /*full_attention_block_size=*/100,
                                          options),
            51);
}

}  // namespace
}  // namespace xllm
