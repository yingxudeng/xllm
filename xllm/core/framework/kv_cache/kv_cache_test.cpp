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

#include "kv_cache.h"

#include <gtest/gtest.h>

namespace xllm {
namespace {

TEST(KVCacheTest, SwapBlocksCopiesLinearAttentionCaches) {
  torch::Tensor conv_cache = torch::arange(12, torch::kFloat32).view({3, 4});
  torch::Tensor ssm_cache = torch::arange(24, torch::kFloat32).view({3, 2, 4});
  KVCache kv_cache(torch::empty({0}), torch::empty({0}), conv_cache, ssm_cache);

  torch::Tensor original_conv = conv_cache.clone();
  torch::Tensor original_ssm = ssm_cache.clone();
  torch::Tensor src_blocks = torch::tensor({0, 2}, torch::kLong);
  torch::Tensor dst_blocks = torch::tensor({1, 0}, torch::kLong);

  kv_cache.swap_blocks(src_blocks, dst_blocks);

  EXPECT_TRUE(torch::equal(kv_cache.get_conv_cache()[1], original_conv[0]));
  EXPECT_TRUE(torch::equal(kv_cache.get_conv_cache()[0], original_conv[2]));
  EXPECT_TRUE(torch::equal(kv_cache.get_ssm_cache()[1], original_ssm[0]));
  EXPECT_TRUE(torch::equal(kv_cache.get_ssm_cache()[0], original_ssm[2]));
}

TEST(KVCacheTest, SwapBlocksCopiesAllDefinedCaches) {
  torch::Tensor key_cache = torch::arange(24, torch::kFloat32).view({3, 2, 4});
  torch::Tensor value_cache =
      torch::arange(24, torch::kFloat32).view({3, 2, 4}) + 100.0f;
  torch::Tensor index_cache = torch::arange(12, torch::kFloat32).view({3, 4});
  torch::Tensor key_scale = torch::arange(6, torch::kFloat32).view({3, 2});
  torch::Tensor value_scale =
      torch::arange(6, torch::kFloat32).view({3, 2}) + 10.0f;
  KVCache kv_cache(key_cache, value_cache, index_cache, key_scale, value_scale);

  torch::Tensor original_key = key_cache.clone();
  torch::Tensor original_value = value_cache.clone();
  torch::Tensor original_index = index_cache.clone();
  torch::Tensor original_key_scale = key_scale.clone();
  torch::Tensor original_value_scale = value_scale.clone();
  torch::Tensor src_blocks = torch::tensor({2}, torch::kLong);
  torch::Tensor dst_blocks = torch::tensor({0}, torch::kLong);

  kv_cache.swap_blocks(src_blocks, dst_blocks);

  EXPECT_TRUE(torch::equal(kv_cache.get_k_cache()[0], original_key[2]));
  EXPECT_TRUE(torch::equal(kv_cache.get_v_cache()[0], original_value[2]));
  EXPECT_TRUE(torch::equal(kv_cache.get_index_cache()[0], original_index[2]));
  ASSERT_TRUE(kv_cache.get_k_cache_scale().has_value());
  ASSERT_TRUE(kv_cache.get_v_cache_scale().has_value());
  EXPECT_TRUE(torch::equal(kv_cache.get_k_cache_scale().value()[0],
                           original_key_scale[2]));
  EXPECT_TRUE(torch::equal(kv_cache.get_v_cache_scale().value()[0],
                           original_value_scale[2]));
}

}  // namespace
}  // namespace xllm
