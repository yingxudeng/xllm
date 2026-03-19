/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "tool_choice_utils.h"

#include <gtest/gtest.h>

namespace xllm {
namespace {

JsonTool make_tool(const std::string& name, const std::string& description) {
  JsonTool tool;
  tool.type = "function";
  tool.function.name = name;
  tool.function.description = description;
  tool.function.parameters = nlohmann::json::object();
  return tool;
}

TEST(ToolChoiceUtilsTest, InjectsInstructionForRequiredToolChoice) {
  RequestParams params;
  params.tool_choice = "required";
  params.tools = {make_tool("get_weather", "Get the current weather")};

  std::vector<Message> messages = {
      Message("system", "You are helpful."),
      Message("user", "What is the capital of France?")};

  inject_tool_choice_instruction(messages, params);

  ASSERT_EQ(messages.size(), 2);
  EXPECT_EQ(messages[0].role, "system");
  EXPECT_TRUE(std::holds_alternative<std::string>(messages[0].content));

  const auto& content = std::get<std::string>(messages[0].content);
  EXPECT_NE(content.find("You are helpful."), std::string::npos);
  EXPECT_NE(content.find("Tool choice is required."), std::string::npos);
  EXPECT_NE(content.find("<tool_call>...</tool_call>"), std::string::npos);
  EXPECT_NE(content.find("get_weather"), std::string::npos);
}

TEST(ToolChoiceUtilsTest, DoesNotInjectWithoutRequiredToolChoice) {
  RequestParams params;
  params.tool_choice = "auto";
  params.tools = {make_tool("get_weather", "Get the current weather")};

  std::vector<Message> messages = {Message("user", "Hello")};

  inject_tool_choice_instruction(messages, params);

  ASSERT_EQ(messages.size(), 1);
  EXPECT_EQ(messages[0].role, "user");
}

TEST(ToolChoiceUtilsTest, DoesNotInjectWhenToolsAreMissing) {
  RequestParams params;
  params.tool_choice = "required";

  std::vector<Message> messages = {Message("user", "Hello")};

  inject_tool_choice_instruction(messages, params);

  ASSERT_EQ(messages.size(), 1);
  EXPECT_EQ(messages[0].role, "user");
}

TEST(ToolChoiceUtilsTest, InsertsInstructionAtFrontWithoutSystemMessage) {
  RequestParams params;
  params.tool_choice = "required";
  params.tools = {make_tool("get_weather", "Get the current weather")};

  std::vector<Message> messages = {Message("user", "Hello")};

  inject_tool_choice_instruction(messages, params);

  ASSERT_EQ(messages.size(), 2);
  EXPECT_EQ(messages[0].role, "system");
  EXPECT_EQ(messages[1].role, "user");
}

}  // namespace
}  // namespace xllm
