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

namespace xllm {
namespace {

TEST(KVCacheUtilsTest, AutoLinearStateBlocksAreDerivedFromKvBudget) {
  LinearStateCacheOptions options;

  EXPECT_EQ(calculate_linear_state_blocks(/*cache_size_in_bytes=*/10000,
                                          /*num_linear_attention_layers=*/1,
                                          /*linear_slot_size=*/100,
                                          /*num_full_attention_layers=*/1,
                                          /*full_attention_block_size=*/100,
                                          options),
            47);
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

TEST(KVCacheUtilsTest, ExplicitLinearStateBlocksRejectOverBudgetRequest) {
  LinearStateCacheOptions options;
  options.max_linear_state_cache_slots(1024);

  EXPECT_DEATH(calculate_linear_state_blocks(/*cache_size_in_bytes=*/10000,
                                             /*num_linear_attention_layers=*/1,
                                             /*linear_slot_size=*/100,
                                             /*num_full_attention_layers=*/1,
                                             /*full_attention_block_size=*/100,
                                             options),
               "configured KV cache budget");
}

TEST(KVCacheUtilsTest, AutoLinearStateBlocksAreBoundedByKvBudget) {
  LinearStateCacheOptions options;

  EXPECT_EQ(calculate_linear_state_blocks(/*cache_size_in_bytes=*/10000,
                                          /*num_linear_attention_layers=*/1,
                                          /*linear_slot_size=*/100,
                                          /*num_full_attention_layers=*/1,
                                          /*full_attention_block_size=*/100,
                                          options),
            47);
}

TEST(KVCacheUtilsTest, LinearStateLiveSlotsLeaveCheckpointSlots) {
  EXPECT_EQ(calculate_linear_state_live_slots(
                /*num_linear_state_blocks=*/10, /*max_running_requests=*/4),
            5);
  EXPECT_EQ(calculate_linear_state_live_slots(
                /*num_linear_state_blocks=*/10, /*max_running_requests=*/32),
            8);
}

TEST(KVCacheUtilsTest, LinearStateLiveSlotsKeepMinimumCapacity) {
  EXPECT_EQ(calculate_linear_state_live_slots(
                /*num_linear_state_blocks=*/0, /*max_running_requests=*/4),
            0);
  EXPECT_EQ(calculate_linear_state_live_slots(
                /*num_linear_state_blocks=*/2, /*max_running_requests=*/4),
            2);
}

TEST(KVCacheUtilsTest, MaxLinearStateCacheSlotsMustBeNonNegative) {
  LinearStateCacheOptions options;
  options.max_linear_state_cache_slots(-1);

  EXPECT_DEATH(validate_linear_state_cache_options(options),
               "must be greater than or equal to 0");
}

}  // namespace
}  // namespace xllm
