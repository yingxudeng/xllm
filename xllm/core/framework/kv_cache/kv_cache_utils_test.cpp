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

TEST(KVCacheUtilsTest, CompatLinearStateBlocksFollowMaxSeqs) {
  EXPECT_EQ(calculate_linear_state_blocks(/*max_seqs_per_batch=*/4,
                                          /*cache_size_in_bytes=*/10000,
                                          /*num_linear_attention_layers=*/1,
                                          /*linear_slot_size=*/10,
                                          /*num_full_attention_layers=*/1,
                                          /*full_attention_block_size=*/100),
            6);
}

TEST(KVCacheUtilsTest, AutoLinearStateBlocksAreMemoryRatioDerived) {
  LinearStateCacheOptions options;
  options.policy(LinearStateCachePolicy::AUTO)
      .linear_state_full_kv_memory_ratio(1.0);

  EXPECT_EQ(calculate_linear_state_blocks(/*max_seqs_per_batch=*/1,
                                          /*cache_size_in_bytes=*/10000,
                                          /*num_linear_attention_layers=*/1,
                                          /*linear_slot_size=*/100,
                                          /*num_full_attention_layers=*/1,
                                          /*full_attention_block_size=*/100,
                                          options),
            50);
}

TEST(KVCacheUtilsTest, AutoLinearStateBlocksDoNotDependOnMaxSeqs) {
  LinearStateCacheOptions options;
  options.policy(LinearStateCachePolicy::AUTO)
      .linear_state_full_kv_memory_ratio(1.0);

  EXPECT_EQ(calculate_linear_state_blocks(/*max_seqs_per_batch=*/1,
                                          /*cache_size_in_bytes=*/10000,
                                          /*num_linear_attention_layers=*/1,
                                          /*linear_slot_size=*/100,
                                          /*num_full_attention_layers=*/1,
                                          /*full_attention_block_size=*/100,
                                          options),
            calculate_linear_state_blocks(/*max_seqs_per_batch=*/1024,
                                          /*cache_size_in_bytes=*/10000,
                                          /*num_linear_attention_layers=*/1,
                                          /*linear_slot_size=*/100,
                                          /*num_full_attention_layers=*/1,
                                          /*full_attention_block_size=*/100,
                                          options));
}

TEST(KVCacheUtilsTest, FixedLinearStateBlocksUseExplicitSlotCapacity) {
  LinearStateCacheOptions options;
  options.policy(LinearStateCachePolicy::FIXED)
      .max_linear_state_cache_slots(12);

  EXPECT_EQ(calculate_linear_state_blocks(/*max_seqs_per_batch=*/1,
                                          /*cache_size_in_bytes=*/10000,
                                          /*num_linear_attention_layers=*/1,
                                          /*linear_slot_size=*/100,
                                          /*num_full_attention_layers=*/1,
                                          /*full_attention_block_size=*/100,
                                          options),
            14);
}

TEST(KVCacheUtilsTest, FixedLinearStateBlocksAreBoundedByKvBudget) {
  LinearStateCacheOptions options;
  options.policy(LinearStateCachePolicy::FIXED)
      .max_linear_state_cache_slots(1024);

  EXPECT_EQ(calculate_linear_state_blocks(/*max_seqs_per_batch=*/1,
                                          /*cache_size_in_bytes=*/10000,
                                          /*num_linear_attention_layers=*/1,
                                          /*linear_slot_size=*/100,
                                          /*num_full_attention_layers=*/1,
                                          /*full_attention_block_size=*/100,
                                          options),
            51);
}

TEST(KVCacheUtilsTest, LinearStateBlocksPreserveMinimumFullKvBlocks) {
  LinearStateCacheOptions options;
  options.policy(LinearStateCachePolicy::AUTO)
      .linear_state_full_kv_memory_ratio(1.0)
      .min_full_kv_cache_blocks(80);

  EXPECT_EQ(calculate_linear_state_blocks(/*max_seqs_per_batch=*/1,
                                          /*cache_size_in_bytes=*/10000,
                                          /*num_linear_attention_layers=*/1,
                                          /*linear_slot_size=*/100,
                                          /*num_full_attention_layers=*/1,
                                          /*full_attention_block_size=*/100,
                                          options),
            20);
}

TEST(KVCacheUtilsTest, CompatLinearStateBlocksPreserveMinimumFullKvBlocks) {
  LinearStateCacheOptions options;
  options.policy(LinearStateCachePolicy::COMPAT).min_full_kv_cache_blocks(80);

  EXPECT_EQ(calculate_linear_state_blocks(/*max_seqs_per_batch=*/1024,
                                          /*cache_size_in_bytes=*/10000,
                                          /*num_linear_attention_layers=*/1,
                                          /*linear_slot_size=*/100,
                                          /*num_full_attention_layers=*/1,
                                          /*full_attention_block_size=*/100,
                                          options),
            20);
}

TEST(KVCacheUtilsTest, MinFullKvBlocksMustFitWithLinearStatePadding) {
  LinearStateCacheOptions options;
  options.policy(LinearStateCachePolicy::AUTO)
      .linear_state_full_kv_memory_ratio(1.0)
      .min_full_kv_cache_blocks(99);

  EXPECT_DEATH(calculate_linear_state_blocks(
                   /*max_seqs_per_batch=*/1,
                   /*cache_size_in_bytes=*/10000,
                   /*num_linear_attention_layers=*/1,
                   /*linear_slot_size=*/100,
                   /*num_full_attention_layers=*/1,
                   /*full_attention_block_size=*/100,
                   options),
               "min_full_kv_cache_blocks cannot be preserved");
}

TEST(KVCacheUtilsTest, LinearStateCacheRatioMustBeFinite) {
  LinearStateCacheOptions options;
  options.linear_state_full_kv_memory_ratio(
      std::numeric_limits<double>::infinity());

  EXPECT_DEATH(validate_linear_state_cache_options(options), "must be finite");
}

TEST(KVCacheUtilsTest, AutoLinearStateBlocksHandlesVeryLargeFiniteRatio) {
  LinearStateCacheOptions options;
  options.policy(LinearStateCachePolicy::AUTO)
      .linear_state_full_kv_memory_ratio(std::numeric_limits<double>::max());

  EXPECT_EQ(calculate_linear_state_blocks(/*max_seqs_per_batch=*/1,
                                          /*cache_size_in_bytes=*/10000,
                                          /*num_linear_attention_layers=*/1,
                                          /*linear_slot_size=*/100,
                                          /*num_full_attention_layers=*/1,
                                          /*full_attention_block_size=*/100,
                                          options),
            51);
}

}  // namespace
}  // namespace xllm
