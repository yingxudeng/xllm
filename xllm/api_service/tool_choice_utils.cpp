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

#include <sstream>
#include <string>

namespace xllm {
namespace {

std::string summarize_tools(const std::vector<JsonTool>& tools) {
  std::ostringstream oss;
  for (const auto& tool : tools) {
    oss << "  - " << tool.function.name;
    if (!tool.function.description.empty()) {
      oss << ": " << tool.function.description;
    }
    oss << "\n";
  }
  return oss.str();
}

std::string build_required_tool_choice_instruction(
    const std::vector<JsonTool>& tools) {
  std::ostringstream oss;
  oss << "Tool choice is required. You must emit at least one tool call in "
         "your next assistant response before any natural-language answer.\n"
      << "Use Qwen's function-calling format and wrap each call in "
         "<tool_call>...</tool_call>.\n"
      << "Inside each <tool_call> block, return JSON that includes the "
         "function name and arguments.\n"
      << "Available tools:\n"
      << summarize_tools(tools) << "Do not answer with plain text only.";
  return oss.str();
}

}  // namespace

bool should_inject_tool_choice_instruction(const RequestParams& params) {
  return false;
}

void inject_tool_choice_instruction(std::vector<Message>& messages,
                                    const RequestParams& params) {
  if (!should_inject_tool_choice_instruction(params)) {
    return;
  }

  const std::string instruction =
      build_required_tool_choice_instruction(params.tools);

  if (!messages.empty() && messages.front().role == "system" &&
      std::holds_alternative<std::string>(messages.front().content)) {
    auto& system_content = std::get<std::string>(messages.front().content);
    if (!system_content.empty()) {
      system_content += "\n\n";
    }
    system_content += instruction;
    return;
  }

  messages.emplace(messages.begin(), Message("system", instruction));
}

}  // namespace xllm
