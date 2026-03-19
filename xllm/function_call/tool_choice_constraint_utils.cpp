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

#include "tool_choice_constraint_utils.h"

#include <glog/logging.h>

#include <algorithm>
#include <nlohmann/json.hpp>

#include "json_array_detector.h"

namespace xllm {
namespace function_call {
namespace {

const JsonTool* find_tool_by_name(const std::vector<JsonTool>& tools,
                                  const std::string& name) {
  for (const auto& tool : tools) {
    if (tool.function.name == name) {
      return &tool;
    }
  }
  return nullptr;
}

bool matches_schema_type(const nlohmann::json& value,
                         const std::string& type_name) {
  if (type_name == "object") {
    return value.is_object();
  }
  if (type_name == "array") {
    return value.is_array();
  }
  if (type_name == "string") {
    return value.is_string();
  }
  if (type_name == "integer") {
    return value.is_number_integer() || value.is_number_unsigned();
  }
  if (type_name == "number") {
    return value.is_number();
  }
  if (type_name == "boolean") {
    return value.is_boolean();
  }
  if (type_name == "null") {
    return value.is_null();
  }
  return true;
}

bool matches_schema(const nlohmann::json& value, const nlohmann::json& schema) {
  if (!schema.is_object()) {
    return true;
  }

  if (schema.contains("enum") && schema["enum"].is_array()) {
    return std::any_of(
        schema["enum"].begin(),
        schema["enum"].end(),
        [&](const nlohmann::json& item) { return item == value; });
  }

  for (const auto& key : {"anyOf", "oneOf"}) {
    if (schema.contains(key) && schema[key].is_array()) {
      return std::any_of(schema[key].begin(),
                         schema[key].end(),
                         [&](const nlohmann::json& branch) {
                           return matches_schema(value, branch);
                         });
    }
  }

  if (schema.contains("type") && schema["type"].is_string() &&
      !matches_schema_type(value, schema["type"].get<std::string>())) {
    return false;
  }

  if (schema.contains("properties") && schema["properties"].is_object()) {
    if (!value.is_object()) {
      return false;
    }
    if (schema.contains("required") && schema["required"].is_array()) {
      for (const auto& required_key : schema["required"]) {
        if (required_key.is_string() &&
            !value.contains(required_key.get<std::string>())) {
          return false;
        }
      }
    }
    for (auto it = schema["properties"].begin();
         it != schema["properties"].end();
         ++it) {
      if (!value.contains(it.key())) {
        continue;
      }
      if (!matches_schema(value[it.key()], it.value())) {
        return false;
      }
    }
  }

  if (schema.contains("items") && schema["items"].is_object()) {
    if (!value.is_array()) {
      return false;
    }
    for (const auto& item : value) {
      if (!matches_schema(item, schema["items"])) {
        return false;
      }
    }
  }

  return true;
}

bool validate_tool_call_item(const nlohmann::json& action,
                             const std::vector<JsonTool>& allowed_tools) {
  if (!action.is_object()) {
    return false;
  }
  if (!action.contains("name") || !action["name"].is_string()) {
    return false;
  }
  const std::string name = action["name"].get<std::string>();
  const JsonTool* tool = find_tool_by_name(allowed_tools, name);
  if (tool == nullptr) {
    return false;
  }

  nlohmann::json parameters = nlohmann::json::object();
  if (action.contains("parameters")) {
    parameters = action["parameters"];
  } else if (action.contains("arguments")) {
    parameters = action["arguments"];
  } else {
    return false;
  }
  if (!parameters.is_object()) {
    return false;
  }

  if (!tool->function.parameters.is_object() ||
      tool->function.parameters.empty()) {
    return true;
  }
  return matches_schema(parameters, tool->function.parameters);
}

}  // namespace

ToolChoiceConstraint resolve_tool_choice_constraint(
    const std::vector<JsonTool>& tools,
    const std::string& tool_choice) {
  ToolChoiceConstraint constraint;
  if (tools.empty() || tool_choice.empty() || tool_choice == "auto" ||
      tool_choice == "none") {
    return constraint;
  }

  if (tool_choice == "required") {
    constraint.mode = ToolCallConstraintMode::REQUIRED;
    constraint.allowed_tool_names.reserve(tools.size());
    constraint.allowed_tools.reserve(tools.size());
    for (const auto& tool : tools) {
      if (!tool.function.name.empty()) {
        constraint.allowed_tool_names.push_back(tool.function.name);
        constraint.allowed_tools.push_back(tool);
      }
    }
    return constraint;
  }

  try {
    auto tool_choice_json = nlohmann::json::parse(tool_choice);
    if (!tool_choice_json.is_object()) {
      return constraint;
    }
    const std::string type = tool_choice_json.value("type", "");
    if (type != "function") {
      return constraint;
    }
    if (!tool_choice_json.contains("function") ||
        !tool_choice_json["function"].is_object()) {
      return constraint;
    }
    const std::string tool_name =
        tool_choice_json["function"].value("name", "");
    const JsonTool* tool = find_tool_by_name(tools, tool_name);
    if (tool_name.empty() || tool == nullptr) {
      return constraint;
    }
    constraint.mode = ToolCallConstraintMode::NAMED;
    constraint.allowed_tool_names = {tool_name};
    constraint.allowed_tools = {*tool};
    return constraint;
  } catch (const std::exception& e) {
    LOG(WARNING) << "Failed to parse tool_choice as JSON: " << e.what();
    return constraint;
  }
}

std::string resolve_tool_call_parser_for_choice(
    const std::vector<JsonTool>& tools,
    const std::string& tool_choice,
    const std::string& default_parser) {
  auto constraint = resolve_tool_choice_constraint(tools, tool_choice);
  if (constraint.enabled()) {
    return "json_array";
  }
  return default_parser;
}

bool is_complete_tool_call_json(const std::string& text,
                                const std::vector<JsonTool>& allowed_tools) {
  if (text.empty() || allowed_tools.empty()) {
    return false;
  }

  try {
    auto json_obj = nlohmann::json::parse(text);
    if (json_obj.is_array()) {
      if (json_obj.empty()) {
        return false;
      }
      return std::all_of(
          json_obj.begin(), json_obj.end(), [&](const nlohmann::json& item) {
            return validate_tool_call_item(item, allowed_tools);
          });
    }
    return validate_tool_call_item(json_obj, allowed_tools);
  } catch (const std::exception&) {
    return false;
  }
}

}  // namespace function_call
}  // namespace xllm
