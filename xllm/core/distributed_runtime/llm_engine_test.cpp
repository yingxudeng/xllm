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

#include "llm_engine.h"

#include <gtest/gtest.h>

namespace xllm {

TEST(LLMEngineTest, LinearStateCacheSlotsFollowSequenceBudget) {
  EXPECT_EQ(LLMEngine::get_linear_state_cache_slots(1), 3);
  EXPECT_EQ(LLMEngine::get_linear_state_cache_slots(256), 258);
}

TEST(LLMEngineTest, LinearConvCacheShapeUsesSequenceSlots) {
  const int64_t seq_slots = 258;
  const auto shape =
      LLMEngine::build_linear_conv_cache_shape(seq_slots,
                                               /*linear_key_head_dim=*/128,
                                               /*n_local_linear_k_heads=*/4,
                                               /*n_local_linear_v_heads=*/2,
                                               /*linear_conv_kernel_dim=*/4);
  ASSERT_EQ(shape.size(), 3);
  EXPECT_EQ(shape[0], seq_slots);
  EXPECT_EQ(shape[1], 128 * 4 * 2 + 128 * 2);
  EXPECT_EQ(shape[2], 3);
}

TEST(LLMEngineTest, LinearSsmCacheShapeUsesSequenceSlots) {
  const int64_t seq_slots = 258;
  const auto shape =
      LLMEngine::build_linear_ssm_cache_shape(seq_slots,
                                              /*n_local_linear_v_heads=*/2,
                                              /*linear_key_head_dim=*/128,
                                              /*linear_value_head_dim=*/64);
  ASSERT_EQ(shape.size(), 4);
  EXPECT_EQ(shape[0], seq_slots);
  EXPECT_EQ(shape[1], 2);
  EXPECT_EQ(shape[2], 128);
  EXPECT_EQ(shape[3], 64);
}

}  // namespace xllm
