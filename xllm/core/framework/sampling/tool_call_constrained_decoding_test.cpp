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

#include "tool_call_constrained_decoding.h"

#include <gtest/gtest.h>

#include <unordered_map>

namespace xllm {
namespace {

class CharTokenizer final : public Tokenizer {
 public:
  bool encode(const std::string_view& text,
              std::vector<int32_t>* ids,
              bool /*add_special_tokens*/) const override {
    ids->clear();
    for (char ch : text) {
      ids->push_back(static_cast<unsigned char>(ch));
    }
    return true;
  }

  std::string decode(const Slice<int32_t>& ids,
                     bool /*skip_special_tokens*/) const override {
    std::string out;
    out.reserve(ids.size());
    for (auto id : ids) {
      out.push_back(static_cast<char>(id));
    }
    return out;
  }

  size_t vocab_size() const override { return 256; }
};

TEST(ToolCallConstrainedDecodingTest, BlocksEmptyRequiredParametersObject) {
  CharTokenizer tokenizer;
  ToolCallConstrainedDecoding decoding(tokenizer,
                                       /*vocab_size=*/256,
                                       torch::kFloat32,
                                       torch::kCPU,
                                       {ToolCallConstraintMode::REQUIRED},
                                       {{"SkuCommentInfo"}},
                                       {{R"({
        "type":"function",
        "function":{
          "name":"SkuCommentInfo",
          "parameters":{
            "type":"object",
            "properties":{
              "sku_ids":{
                "anyOf":[
                  {"type":"array","items":{"type":"string"}},
                  {"type":"string","enum":["ALL"]}
                ]
              }
            },
            "required":["sku_ids"]
          }
        }
      })"}});

  ASSERT_TRUE(decoding.build_mask_cache());

  std::vector<int32_t> generated;
  ASSERT_TRUE(tokenizer.encode(
      R"([{"name":"SkuCommentInfo","parameters":{)", &generated, false));
  auto mask = decoding.generate_mask({generated});
  ASSERT_TRUE(mask.defined());

  EXPECT_EQ(mask.index({0, static_cast<int>('}')}).item<float>(), -10000.0f);
  EXPECT_EQ(mask.index({0, static_cast<int>('"')}).item<float>(), 0.0f);
}

TEST(ToolCallConstrainedDecodingTest, AllowsSchemaDrivenValuePrefixes) {
  CharTokenizer tokenizer;
  ToolCallConstrainedDecoding decoding(tokenizer,
                                       /*vocab_size=*/256,
                                       torch::kFloat32,
                                       torch::kCPU,
                                       {ToolCallConstraintMode::REQUIRED},
                                       {{"SkuCommentInfo"}},
                                       {{R"({
        "type":"function",
        "function":{
          "name":"SkuCommentInfo",
          "parameters":{
            "type":"object",
            "properties":{
              "sku_ids":{
                "anyOf":[
                  {"type":"array","items":{"type":"string"}},
                  {"type":"string","enum":["ALL"]}
                ]
              }
            },
            "required":["sku_ids"]
          }
        }
      })"}});

  ASSERT_TRUE(decoding.build_mask_cache());

  std::vector<int32_t> generated;
  ASSERT_TRUE(
      tokenizer.encode(R"([{"name":"SkuCommentInfo","parameters":{"sku_ids":)",
                       &generated,
                       false));
  auto mask = decoding.generate_mask({generated});
  ASSERT_TRUE(mask.defined());

  EXPECT_EQ(mask.index({0, static_cast<int>('[')}).item<float>(), 0.0f);
  EXPECT_EQ(mask.index({0, static_cast<int>('"')}).item<float>(), 0.0f);
  EXPECT_EQ(mask.index({0, static_cast<int>('}')}).item<float>(), -10000.0f);
}

}  // namespace
}  // namespace xllm
