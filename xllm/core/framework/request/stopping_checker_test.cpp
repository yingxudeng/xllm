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

#include "stopping_checker.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace xllm {
namespace {

function_call::JsonTool make_tool(const std::string& name,
                                  const nlohmann::json& parameters) {
  function_call::JsonTool tool;
  tool.type = "function";
  tool.function.name = name;
  tool.function.parameters = parameters;
  return tool;
}

class MockTokenizer final : public Tokenizer {
 public:
  explicit MockTokenizer(std::unordered_map<int32_t, std::string> token_map)
      : token_map_(std::move(token_map)) {}

  std::string decode(const Slice<int32_t>& ids,
                     bool /*skip_special_tokens*/) const override {
    std::string out;
    for (auto id : ids) {
      auto it = token_map_.find(id);
      if (it != token_map_.end()) {
        out += it->second;
      }
    }
    return out;
  }

 private:
  std::unordered_map<int32_t, std::string> token_map_;
};

TEST(StoppingCheckerTest, DetectsCompletedRequiredToolCallJson) {
  StoppingChecker checker(/*max_generated_tokens=*/128,
                          /*max_context_len=*/0,
                          /*eos_token=*/-1,
                          /*ignore_eos=*/false,
                          /*stop_tokens=*/{},
                          /*stop_sequences=*/{});
  checker.set_tool_call_constraint(
      std::make_shared<MockTokenizer>(std::unordered_map<int32_t, std::string>{
          {101, "["},
          {102,
           "{\"name\":\"get_weather\",\"parameters\":{\"city\":\"Beijing\"}}"},
          {103, "]"},
      }),
      ToolCallConstraintMode::REQUIRED,
      {make_tool("get_weather", nlohmann::json::parse(R"({
            "type":"object",
            "properties":{"city":{"type":"string"}},
            "required":["city"]
          })"))});

  std::vector<int32_t> token_ids = {1, 101, 102, 103};
  EXPECT_EQ(checker.check(token_ids, /*num_prompt_tokens=*/1),
            FinishReason::FUNCTION_CALL);
}

TEST(StoppingCheckerTest, RejectsUnknownToolInCompletedJson) {
  StoppingChecker checker(/*max_generated_tokens=*/128,
                          /*max_context_len=*/0,
                          /*eos_token=*/-1,
                          /*ignore_eos=*/false,
                          /*stop_tokens=*/{},
                          /*stop_sequences=*/{});
  checker.set_tool_call_constraint(
      std::make_shared<MockTokenizer>(std::unordered_map<int32_t, std::string>{
          {201, "["},
          {202, "{\"name\":\"reply\",\"parameters\":{\"text\":\"hello\"}}"},
          {203, "]"},
      }),
      ToolCallConstraintMode::REQUIRED,
      {make_tool("get_weather", nlohmann::json::parse(R"({
            "type":"object",
            "properties":{"city":{"type":"string"}},
            "required":["city"]
          })"))});

  std::vector<int32_t> token_ids = {1, 201, 202, 203};
  EXPECT_EQ(checker.check(token_ids, /*num_prompt_tokens=*/1),
            FinishReason::NONE);
}

TEST(StoppingCheckerTest, RejectsMissingRequiredArgumentsInCompletedJson) {
  StoppingChecker checker(/*max_generated_tokens=*/128,
                          /*max_context_len=*/0,
                          /*eos_token=*/-1,
                          /*ignore_eos=*/false,
                          /*stop_tokens=*/{},
                          /*stop_sequences=*/{});
  checker.set_tool_call_constraint(
      std::make_shared<MockTokenizer>(std::unordered_map<int32_t, std::string>{
          {301, "["},
          {302, "{\"name\":\"SkuCommentInfo\",\"parameters\":{}}"},
          {303, "]"},
      }),
      ToolCallConstraintMode::REQUIRED,
      {make_tool("SkuCommentInfo", nlohmann::json::parse(R"({
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
          })"))});

  std::vector<int32_t> token_ids = {1, 301, 302, 303};
  EXPECT_EQ(checker.check(token_ids, /*num_prompt_tokens=*/1),
            FinishReason::NONE);
}

}  // namespace
}  // namespace xllm
