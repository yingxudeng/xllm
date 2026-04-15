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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "framework/batch/batch_forward_type.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "layers/common/attention_metadata.h"
#include "layers/common/attention_metadata_builder.h"

namespace xllm::layer {
namespace {

#if defined(USE_NPU)
ModelArgs create_gdn_model_args() {
  ModelArgs model_args;
  model_args.layer_types({"linear_attention"});
  return model_args;
}

TEST(GatedDeltaNetMetadataTest,
     MixedBatchBuildsPackedPrefillAndDecodeMetadata) {
  ModelArgs model_args = create_gdn_model_args();
  ModelInputParams input_params;
  input_params.batch_forward_type = BatchForwardType::MIXED;
  input_params.num_sequences = 4;
  input_params.q_max_seq_len = 9;
  input_params.kv_max_seq_len = 16;
  input_params.q_seq_lens_vec = {9, 1, 1, 4};
  input_params.kv_seq_lens_vec = {9, 8, 16, 8};
  input_params.q_seq_lens = torch::tensor({9, 1, 1, 4}, torch::kInt32);
  input_params.kv_seq_lens = torch::tensor({9, 8, 16, 8}, torch::kInt32);
  input_params.block_tables = torch::tensor(
      {{1, 2, 3}, {4, 5, 6}, {8, 9, 10}, {13, 14, 15}}, torch::kInt32);
  input_params.kv_cache_tokens_nums_host = {0, 7, 15, 4};

  AttentionMetadata metadata =
      AttentionMetadataBuilder::build(input_params, model_args);
  ASSERT_TRUE(metadata.gated_delta_net_metadata.has_value());

  const AttentionMetadata::GatedDeltaNetMetadata& gdn_metadata =
      metadata.gated_delta_net_metadata.value();
  EXPECT_EQ(gdn_metadata.num_non_spec_prefills, 2);
  EXPECT_EQ(gdn_metadata.num_non_spec_prefill_tokens, 13);
  EXPECT_EQ(gdn_metadata.num_decodes, 2);
  EXPECT_EQ(gdn_metadata.max_non_spec_query_len, 9);

  EXPECT_TRUE(
      torch::equal(gdn_metadata.non_spec_prefill_token_indices.cpu(),
                   torch::tensor({0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14},
                                 torch::kInt64)));
  EXPECT_TRUE(torch::equal(gdn_metadata.decode_token_indices.cpu(),
                           torch::tensor({9, 10}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(gdn_metadata.non_spec_query_start_loc_host,
                           torch::tensor({0, 9, 13}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(gdn_metadata.non_spec_state_indices.cpu(),
                           torch::tensor({1, 13}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(gdn_metadata.decode_state_indices.cpu(),
                           torch::tensor({4, 8}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(gdn_metadata.has_initial_state.cpu(),
                           torch::tensor({false, true}, torch::kBool)));
}

TEST(GatedDeltaNetMetadataTest, PurePrefillBuildsVarlenMetadata) {
  ModelArgs model_args = create_gdn_model_args();
  ModelInputParams input_params;
  input_params.batch_forward_type = BatchForwardType::PREFILL;
  input_params.num_sequences = 2;
  input_params.q_max_seq_len = 5;
  input_params.kv_max_seq_len = 9;
  input_params.q_seq_lens_vec = {5, 3};
  input_params.kv_seq_lens_vec = {5, 9};
  input_params.q_seq_lens = torch::tensor({5, 3}, torch::kInt32);
  input_params.kv_seq_lens = torch::tensor({5, 9}, torch::kInt32);
  input_params.block_tables = torch::tensor({{2, 3}, {7, 8}}, torch::kInt32);
  input_params.kv_cache_tokens_nums_host = {0, 6};

  AttentionMetadata metadata =
      AttentionMetadataBuilder::build(input_params, model_args);
  ASSERT_TRUE(metadata.gated_delta_net_metadata.has_value());

  const AttentionMetadata::GatedDeltaNetMetadata& gdn_metadata =
      metadata.gated_delta_net_metadata.value();
  EXPECT_EQ(gdn_metadata.num_non_spec_prefills, 2);
  EXPECT_EQ(gdn_metadata.num_non_spec_prefill_tokens, 8);
  EXPECT_EQ(gdn_metadata.num_decodes, 0);
  EXPECT_EQ(gdn_metadata.max_non_spec_query_len, 5);

  EXPECT_TRUE(
      torch::equal(gdn_metadata.non_spec_prefill_token_indices.cpu(),
                   torch::tensor({0, 1, 2, 3, 4, 5, 6, 7}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(gdn_metadata.non_spec_query_start_loc_host,
                           torch::tensor({0, 5, 8}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(gdn_metadata.non_spec_state_indices.cpu(),
                           torch::tensor({2, 7}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(gdn_metadata.has_initial_state.cpu(),
                           torch::tensor({false, true}, torch::kBool)));
}

TEST(GatedDeltaNetMetadataTest, DecodeOnlyBuildsDecodeMetadata) {
  ModelArgs model_args = create_gdn_model_args();
  ModelInputParams input_params;
  input_params.batch_forward_type = BatchForwardType::DECODE;
  input_params.num_sequences = 3;
  input_params.q_max_seq_len = 1;
  input_params.kv_max_seq_len = 12;
  input_params.q_seq_lens_vec = {1, 1, 1};
  input_params.kv_seq_lens_vec = {4, 7, 12};
  input_params.q_seq_lens = torch::tensor({1, 1, 1}, torch::kInt32);
  input_params.kv_seq_lens = torch::tensor({4, 7, 12}, torch::kInt32);
  input_params.block_tables =
      torch::tensor({{3, 4}, {9, 10}, {12, 13}}, torch::kInt32);
  input_params.kv_cache_tokens_nums_host = {3, 6, 11};

  AttentionMetadata metadata =
      AttentionMetadataBuilder::build(input_params, model_args);
  ASSERT_TRUE(metadata.gated_delta_net_metadata.has_value());

  const AttentionMetadata::GatedDeltaNetMetadata& gdn_metadata =
      metadata.gated_delta_net_metadata.value();
  EXPECT_EQ(gdn_metadata.num_non_spec_prefills, 0);
  EXPECT_EQ(gdn_metadata.num_non_spec_prefill_tokens, 0);
  EXPECT_EQ(gdn_metadata.num_decodes, 3);
  EXPECT_EQ(gdn_metadata.max_non_spec_query_len, 0);
  EXPECT_TRUE(torch::equal(gdn_metadata.non_spec_prefill_token_indices.cpu(),
                           torch::tensor({}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(gdn_metadata.decode_token_indices.cpu(),
                           torch::tensor({0, 1, 2}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(gdn_metadata.non_spec_query_start_loc_host,
                           torch::tensor({0}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(gdn_metadata.decode_state_indices.cpu(),
                           torch::tensor({3, 9, 12}, torch::kInt32)));
}

TEST(GatedDeltaNetMetadataTest, ZeroLengthRowsAreSkippedFromMetadata) {
  ModelArgs model_args = create_gdn_model_args();
  ModelInputParams input_params;
  input_params.batch_forward_type = BatchForwardType::MIXED;
  input_params.num_sequences = 4;
  input_params.q_max_seq_len = 6;
  input_params.kv_max_seq_len = 10;
  input_params.q_seq_lens_vec = {1, 0, 6, 0};
  input_params.kv_seq_lens_vec = {5, 5, 10, 10};
  input_params.q_seq_lens = torch::tensor({1, 0, 6, 0}, torch::kInt32);
  input_params.kv_seq_lens = torch::tensor({5, 5, 10, 10}, torch::kInt32);
  input_params.block_tables =
      torch::tensor({{2, 3}, {4, 5}, {7, 8}, {9, 10}}, torch::kInt32);
  input_params.kv_cache_tokens_nums_host = {4, 4, 9, 9};

  AttentionMetadata metadata =
      AttentionMetadataBuilder::build(input_params, model_args);
  ASSERT_TRUE(metadata.gated_delta_net_metadata.has_value());

  const AttentionMetadata::GatedDeltaNetMetadata& gdn_metadata =
      metadata.gated_delta_net_metadata.value();
  EXPECT_EQ(gdn_metadata.num_non_spec_prefills, 1);
  EXPECT_EQ(gdn_metadata.num_non_spec_prefill_tokens, 6);
  EXPECT_EQ(gdn_metadata.num_decodes, 1);
  EXPECT_TRUE(torch::equal(gdn_metadata.non_spec_prefill_token_indices.cpu(),
                           torch::tensor({1, 2, 3, 4, 5, 6}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(gdn_metadata.decode_token_indices.cpu(),
                           torch::tensor({0}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(gdn_metadata.non_spec_query_start_loc_host,
                           torch::tensor({0, 6}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(gdn_metadata.non_spec_state_indices.cpu(),
                           torch::tensor({7}, torch::kInt32)));
}
#endif

}  // namespace
}  // namespace xllm::layer
