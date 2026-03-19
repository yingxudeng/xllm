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

#include "function_call_parser.h"

#include <gtest/gtest.h>

#include "tool_choice_constraint_utils.h"

namespace xllm::function_call {

TEST(FunctionCallParserTest, EmptyParserDefaultsToQwenParserForQwen3) {
  EXPECT_EQ(FunctionCallParser::get_parser_auto("", "qwen3"), "qwen25");
  EXPECT_EQ(FunctionCallParser::get_parser_auto("", "qwen2"), "qwen25");
}

TEST(FunctionCallParserTest, EmptyParserStaysDisabledForUnsupportedModels) {
  EXPECT_TRUE(FunctionCallParser::get_parser_auto("", "glm4_moe").empty());
}

TEST(FunctionCallParserTest, RequiredToolChoiceUsesJsonArrayParser) {
  JsonTool tool;
  tool.type = "function";
  tool.function.name = "get_weather";
  tool.function.parameters = nlohmann::json::object();

  EXPECT_EQ(resolve_tool_call_parser_for_choice({tool}, "required", "qwen25"),
            "json_array");
}

TEST(FunctionCallParserTest, JsonArrayParserRepairsMissingRequiredArguments) {
  JsonTool tool;
  tool.type = "function";
  tool.function.name = "SkuCommentInfo";
  tool.function.parameters = nlohmann::json::parse(R"({
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
  })");

  FunctionCallParser parser({tool}, "json_array");
  auto [normal_text, calls] =
      parser.parse_non_stream(R"([{"name":"SkuCommentInfo","parameters":{}}])");

  ASSERT_TRUE(normal_text.empty());
  ASSERT_EQ(calls.size(), 1);
  ASSERT_TRUE(calls[0].name.has_value());
  EXPECT_EQ(calls[0].name.value(), "SkuCommentInfo");
  EXPECT_EQ(calls[0].parameters, R"({"sku_ids":"ALL"})");
}

TEST(FunctionCallParserTest, CompleteToolCallRequiresSchemaValidArguments) {
  JsonTool tool;
  tool.type = "function";
  tool.function.name = "SkuCommentInfo";
  tool.function.parameters = nlohmann::json::parse(R"({
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
  })");

  EXPECT_FALSE(is_complete_tool_call_json(
      R"([{"name":"SkuCommentInfo","parameters":{}}])", {tool}));
  EXPECT_TRUE(is_complete_tool_call_json(
      R"([{"name":"SkuCommentInfo","parameters":{"sku_ids":"ALL"}}])", {tool}));
}

}  // namespace xllm::function_call
