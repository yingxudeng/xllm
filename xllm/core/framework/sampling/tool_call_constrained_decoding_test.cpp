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

TEST(ToolCallConstrainedDecodingTest, BlocksTrailingCommaWhenNoMoreProperties) {
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
              "sku_ids":{"type":"string","enum":["ALL"]}
            },
            "required":["sku_ids"]
          }
        }
      })"}});

  ASSERT_TRUE(decoding.build_mask_cache());

  std::vector<int32_t> generated;
  ASSERT_TRUE(tokenizer.encode(
      R"([{"name":"SkuCommentInfo","parameters":{"sku_ids":"ALL")",
      &generated,
      false));
  auto mask = decoding.generate_mask({generated});
  ASSERT_TRUE(mask.defined());

  EXPECT_EQ(mask.index({0, static_cast<int>('}')}).item<float>(), 0.0f);
  EXPECT_EQ(mask.index({0, static_cast<int>(',')}).item<float>(), -10000.0f);
}

TEST(ToolCallConstrainedDecodingTest,
     AllowsOptionalPropertyBeforeRemainingRequiredProperty) {
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
              "query_type":{
                "type":"array",
                "items":{"type":"string","enum":["detail"]}
              },
              "sku_ids":{
                "type":"array",
                "items":{"type":"string"}
              }
            },
            "required":["sku_ids"]
          }
        }
      })"}});

  ASSERT_TRUE(decoding.build_mask_cache());

  std::vector<int32_t> generated;
  ASSERT_TRUE(tokenizer.encode(
      R"([{"name":"SkuCommentInfo","parameters":{"query_type":["detail"])",
      &generated,
      false));
  auto mask = decoding.generate_mask({generated});
  ASSERT_TRUE(mask.defined());

  EXPECT_EQ(mask.index({0, static_cast<int>(',')}).item<float>(), 0.0f);
  EXPECT_EQ(mask.index({0, static_cast<int>('}')}).item<float>(), -10000.0f);
}

TEST(ToolCallConstrainedDecodingTest, NamedModeLimitsArrayToSingleToolCall) {
  CharTokenizer tokenizer;
  ToolCallConstrainedDecoding decoding(tokenizer,
                                       /*vocab_size=*/256,
                                       torch::kFloat32,
                                       torch::kCPU,
                                       {ToolCallConstraintMode::NAMED},
                                       {{"SkuCommentInfo"}},
                                       {{R"({
        "type":"function",
        "function":{
          "name":"SkuCommentInfo",
          "parameters":{
            "type":"object",
            "properties":{
              "sku_ids":{"type":"string","enum":["ALL"]}
            },
            "required":["sku_ids"]
          }
        }
      })"}});

  ASSERT_TRUE(decoding.build_mask_cache());

  std::vector<int32_t> generated;
  ASSERT_TRUE(tokenizer.encode(
      R"([{"name":"SkuCommentInfo","parameters":{"sku_ids":"ALL"}})",
      &generated,
      false));
  auto mask = decoding.generate_mask({generated});
  ASSERT_TRUE(mask.defined());

  EXPECT_EQ(mask.index({0, static_cast<int>(']')}).item<float>(), 0.0f);
  EXPECT_EQ(mask.index({0, static_cast<int>(',')}).item<float>(), -10000.0f);
}

TEST(ToolCallConstrainedDecodingTest, AllowsEmptyOptionalObjectValue) {
  CharTokenizer tokenizer;
  ToolCallConstrainedDecoding decoding(tokenizer,
                                       /*vocab_size=*/256,
                                       torch::kFloat32,
                                       torch::kCPU,
                                       {ToolCallConstraintMode::REQUIRED},
                                       {{"Ping"}},
                                       {{R"({
        "type":"function",
        "function":{
          "name":"Ping",
          "parameters":{
            "type":"object",
            "properties":{
              "metadata":{
                "type":"object",
                "properties":{
                  "region":{"type":"string"}
                }
              }
            },
            "required":["metadata"]
          }
        }
      })"}});

  ASSERT_TRUE(decoding.build_mask_cache());

  std::vector<int32_t> generated;
  ASSERT_TRUE(tokenizer.encode(
      R"([{"name":"Ping","parameters":{"metadata":{)", &generated, false));
  auto mask = decoding.generate_mask({generated});
  ASSERT_TRUE(mask.defined());

  EXPECT_EQ(mask.index({0, static_cast<int>('}')}).item<float>(), 0.0f);
  EXPECT_EQ(mask.index({0, static_cast<int>('"')}).item<float>(), 0.0f);
}

TEST(ToolCallConstrainedDecodingTest, AllowsEmptyOptionalArrayValue) {
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
              "sku_ids":{"type":"string","enum":["ALL"]},
              "query_type":{
                "type":"array",
                "items":{"type":"string","enum":["detail","good_comment"]}
              }
            },
            "required":["sku_ids"]
          }
        }
      })"}});

  ASSERT_TRUE(decoding.build_mask_cache());

  std::vector<int32_t> generated;
  ASSERT_TRUE(tokenizer.encode(
      R"([{"name":"SkuCommentInfo","parameters":{"sku_ids":"ALL","query_type":[)",
      &generated,
      false));
  auto mask = decoding.generate_mask({generated});
  ASSERT_TRUE(mask.defined());

  EXPECT_EQ(mask.index({0, static_cast<int>(']')}).item<float>(), 0.0f);
  EXPECT_EQ(mask.index({0, static_cast<int>('"')}).item<float>(), 0.0f);
}

TEST(ToolCallConstrainedDecodingTest, ResolvesLocalDefsRefs) {
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
            "$defs":{
              "SkuIds":{
                "anyOf":[
                  {"type":"array","items":{"type":"string"}},
                  {"type":"string","enum":["ALL"]}
                ]
              }
            },
            "properties":{
              "sku_ids":{"$ref":"#/$defs/SkuIds"}
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
}

TEST(ToolCallConstrainedDecodingTest, SupportsTypeUnionWithNull) {
  CharTokenizer tokenizer;
  ToolCallConstrainedDecoding decoding(tokenizer,
                                       /*vocab_size=*/256,
                                       torch::kFloat32,
                                       torch::kCPU,
                                       {ToolCallConstraintMode::REQUIRED},
                                       {{"NullableEcho"}},
                                       {{R"({
        "type":"function",
        "function":{
          "name":"NullableEcho",
          "parameters":{
            "type":"object",
            "properties":{
              "value":{"type":["string","null"]}
            },
            "required":["value"]
          }
        }
      })"}});

  ASSERT_TRUE(decoding.build_mask_cache());

  std::vector<int32_t> generated;
  ASSERT_TRUE(tokenizer.encode(
      R"([{"name":"NullableEcho","parameters":{"value":)", &generated, false));
  auto mask = decoding.generate_mask({generated});
  ASSERT_TRUE(mask.defined());

  EXPECT_EQ(mask.index({0, static_cast<int>('"')}).item<float>(), 0.0f);
  EXPECT_EQ(mask.index({0, static_cast<int>('n')}).item<float>(), 0.0f);
}

}  // namespace
}  // namespace xllm
