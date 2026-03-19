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

#pragma once

#include <string>
#include <vector>

#include "core/framework/sampling/sampling_params.h"
#include "core_types.h"

namespace xllm {
namespace function_call {

struct ToolChoiceConstraint {
  ToolCallConstraintMode mode = ToolCallConstraintMode::NONE;
  std::vector<std::string> allowed_tool_names;
  std::vector<JsonTool> allowed_tools;

  bool enabled() const { return mode != ToolCallConstraintMode::NONE; }
};

ToolChoiceConstraint resolve_tool_choice_constraint(
    const std::vector<JsonTool>& tools,
    const std::string& tool_choice);

std::string resolve_tool_call_parser_for_choice(
    const std::vector<JsonTool>& tools,
    const std::string& tool_choice,
    const std::string& default_parser);

bool is_complete_tool_call_json(const std::string& text,
                                const std::vector<JsonTool>& allowed_tools);

}  // namespace function_call
}  // namespace xllm
