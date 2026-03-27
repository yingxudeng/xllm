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

#include "qwen3_gated_delta_net_base.h"

#include <gtest/gtest.h>

namespace xllm::layer {

TEST(Qwen3GatedDeltaNetBaseTest, UsesEmbeddingIdsWhenAvailable) {
  AttentionMetadata metadata;
  metadata.block_table =
      torch::tensor({{9, 10}, {12, 13}}, torch::dtype(torch::kInt32));
  ModelInputParams input_params;
  input_params.embedding_ids = {1, 7};

  const auto indices = Qwen3GatedDeltaNetBaseImpl::build_linear_state_indices(
      metadata, input_params, torch::kCPU);

  ASSERT_EQ(indices.sizes().size(), 1);
  ASSERT_EQ(indices.size(0), 2);
  EXPECT_EQ(indices[0].item<int64_t>(), 1);
  EXPECT_EQ(indices[1].item<int64_t>(), 7);
}

TEST(Qwen3GatedDeltaNetBaseTest, FallsBackToFirstBlockIdWhenNoEmbeddingIds) {
  AttentionMetadata metadata;
  metadata.block_table =
      torch::tensor({{9, 10}, {12, 13}}, torch::dtype(torch::kInt32));
  ModelInputParams input_params;

  const auto indices = Qwen3GatedDeltaNetBaseImpl::build_linear_state_indices(
      metadata, input_params, torch::kCPU);

  ASSERT_EQ(indices.sizes().size(), 1);
  ASSERT_EQ(indices.size(0), 2);
  EXPECT_EQ(indices[0].item<int64_t>(), 9);
  EXPECT_EQ(indices[1].item<int64_t>(), 12);
}

TEST(Qwen3GatedDeltaNetBaseTest, ThrowsOnEmbeddingIdSizeMismatch) {
  AttentionMetadata metadata;
  metadata.block_table =
      torch::tensor({{9, 10}, {12, 13}}, torch::dtype(torch::kInt32));
  ModelInputParams input_params;
  input_params.embedding_ids = {3};

  EXPECT_THROW(Qwen3GatedDeltaNetBaseImpl::build_linear_state_indices(
                   metadata, input_params, torch::kCPU),
               c10::Error);
}

}  // namespace xllm::layer
