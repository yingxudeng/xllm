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

#include "base_format_detector.h"

#include <algorithm>
#include <iostream>
#include <regex>
#include <sstream>

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

int schema_branch_priority(const nlohmann::json& schema) {
  if (!schema.is_object()) {
    return 100;
  }
  if (schema.contains("default")) {
    return 0;
  }
  if (schema.contains("enum") && schema["enum"].is_array() &&
      !schema["enum"].empty()) {
    return 1;
  }
  if (schema.contains("type") && schema["type"].is_string()) {
    const auto& type = schema["type"];
    if (type == "object") {
      return 2;
    }
    if (type == "array") {
      return 3;
    }
    if (type == "string") {
      return 4;
    }
    if (type == "integer") {
      return 5;
    }
    if (type == "number") {
      return 6;
    }
    if (type == "boolean") {
      return 7;
    }
  }
  if (schema.contains("properties")) {
    return 2;
  }
  if (schema.contains("items")) {
    return 3;
  }
  return 50;
}

const nlohmann::json* choose_schema_branch(const nlohmann::json& schema) {
  const nlohmann::json* best = nullptr;
  int best_priority = 1000;
  for (const auto& key : {"anyOf", "oneOf"}) {
    if (!schema.contains(key) || !schema[key].is_array()) {
      continue;
    }
    for (const auto& branch : schema[key]) {
      int priority = schema_branch_priority(branch);
      if (priority < best_priority) {
        best_priority = priority;
        best = &branch;
      }
    }
    if (best) {
      return best;
    }
  }
  return nullptr;
}

bool matches_schema(const nlohmann::json& value, const nlohmann::json& schema);
nlohmann::json synthesize_value_from_schema(const nlohmann::json& schema);
nlohmann::json repair_value_with_schema(const nlohmann::json& value,
                                        const nlohmann::json& schema);

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
    for (const auto& enum_value : schema["enum"]) {
      if (value == enum_value) {
        return true;
      }
    }
    return false;
  }

  for (const auto& key : {"anyOf", "oneOf"}) {
    if (schema.contains(key) && schema[key].is_array()) {
      for (const auto& branch : schema[key]) {
        if (matches_schema(value, branch)) {
          return true;
        }
      }
      return false;
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

nlohmann::json synthesize_value_from_schema(const nlohmann::json& schema) {
  if (!schema.is_object()) {
    return nlohmann::json::object();
  }
  if (schema.contains("default")) {
    return schema["default"];
  }
  if (schema.contains("enum") && schema["enum"].is_array() &&
      !schema["enum"].empty()) {
    return schema["enum"].front();
  }
  if (const auto* branch = choose_schema_branch(schema); branch != nullptr) {
    return synthesize_value_from_schema(*branch);
  }
  if (schema.contains("type") && schema["type"].is_string()) {
    const auto type_name = schema["type"].get<std::string>();
    if (type_name == "object") {
      nlohmann::json result = nlohmann::json::object();
      if (schema.contains("required") && schema["required"].is_array() &&
          schema.contains("properties") && schema["properties"].is_object()) {
        for (const auto& required_key : schema["required"]) {
          if (!required_key.is_string()) {
            continue;
          }
          const auto key = required_key.get<std::string>();
          if (schema["properties"].contains(key)) {
            result[key] =
                synthesize_value_from_schema(schema["properties"][key]);
          }
        }
      }
      return result;
    }
    if (type_name == "array") {
      return nlohmann::json::array();
    }
    if (type_name == "string") {
      return "";
    }
    if (type_name == "integer") {
      return 0;
    }
    if (type_name == "number") {
      return 0.0;
    }
    if (type_name == "boolean") {
      return false;
    }
    if (type_name == "null") {
      return nullptr;
    }
  }
  if (schema.contains("properties")) {
    return synthesize_value_from_schema(nlohmann::json{
        {"type", "object"},
        {"properties", schema["properties"]},
        {"required", schema.value("required", nlohmann::json::array())}});
  }
  if (schema.contains("items")) {
    return nlohmann::json::array();
  }
  return nlohmann::json::object();
}

nlohmann::json repair_value_with_schema(const nlohmann::json& value,
                                        const nlohmann::json& schema) {
  if (!schema.is_object()) {
    return value;
  }

  for (const auto& key : {"anyOf", "oneOf"}) {
    if (schema.contains(key) && schema[key].is_array()) {
      for (const auto& branch : schema[key]) {
        if (matches_schema(value, branch)) {
          return repair_value_with_schema(value, branch);
        }
      }
      if (const auto* branch = choose_schema_branch(schema);
          branch != nullptr) {
        return synthesize_value_from_schema(*branch);
      }
      return synthesize_value_from_schema(schema);
    }
  }

  if (schema.contains("enum") && schema["enum"].is_array()) {
    return matches_schema(value, schema) ? value
                                         : synthesize_value_from_schema(schema);
  }

  if (schema.contains("type") && schema["type"].is_string()) {
    const auto type_name = schema["type"].get<std::string>();
    if (type_name == "object" || schema.contains("properties")) {
      if (!value.is_object()) {
        return synthesize_value_from_schema(schema);
      }
      nlohmann::json repaired = value;
      if (schema.contains("properties") && schema["properties"].is_object()) {
        for (auto it = schema["properties"].begin();
             it != schema["properties"].end();
             ++it) {
          if (repaired.contains(it.key())) {
            repaired[it.key()] =
                repair_value_with_schema(repaired[it.key()], it.value());
          }
        }
      }
      if (schema.contains("required") && schema["required"].is_array() &&
          schema.contains("properties") && schema["properties"].is_object()) {
        for (const auto& required_key : schema["required"]) {
          if (!required_key.is_string()) {
            continue;
          }
          const auto key = required_key.get<std::string>();
          if (!repaired.contains(key) && schema["properties"].contains(key)) {
            repaired[key] =
                synthesize_value_from_schema(schema["properties"][key]);
          }
        }
      }
      return repaired;
    }
    if (type_name == "array") {
      if (!value.is_array()) {
        return synthesize_value_from_schema(schema);
      }
      if (schema.contains("items") && schema["items"].is_object()) {
        nlohmann::json repaired = nlohmann::json::array();
        for (const auto& item : value) {
          repaired.push_back(repair_value_with_schema(item, schema["items"]));
        }
        return repaired;
      }
      return value;
    }
    return matches_schema_type(value, type_name)
               ? value
               : synthesize_value_from_schema(schema);
  }

  if (schema.contains("properties")) {
    return repair_value_with_schema(
        value,
        nlohmann::json{
            {"type", "object"},
            {"properties", schema["properties"]},
            {"required", schema.value("required", nlohmann::json::array())}});
  }

  return value;
}

std::string normalize_parameters_for_tool(const std::string& tool_name,
                                          const nlohmann::json& parameters,
                                          const std::vector<JsonTool>& tools) {
  const auto* tool = find_tool_by_name(tools, tool_name);
  nlohmann::json normalized = parameters;
  if (tool != nullptr && tool->function.parameters.is_object() &&
      !tool->function.parameters.empty()) {
    normalized =
        repair_value_with_schema(parameters, tool->function.parameters);
  }
  try {
    return normalized.dump(
        -1, ' ', false, nlohmann::json::error_handler_t::ignore);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to serialize normalized arguments for tool: "
               << tool_name << ", error: " << e.what();
    return "{}";
  }
}

}  // namespace

BaseFormatDetector::BaseFormatDetector()
    : current_tool_id_(-1),
      current_tool_name_sent_(false),
      bot_token_(""),
      eot_token_(""),
      tool_call_separator_(", ") {}

std::unordered_map<std::string, int32_t> BaseFormatDetector::get_tool_indices(
    const std::vector<JsonTool>& tools) const {
  std::unordered_map<std::string, int32_t> indices;
  for (size_t i = 0; i < tools.size(); ++i) {
    if (!tools[i].function.name.empty()) {
      indices[tools[i].function.name] = static_cast<int32_t>(i);
    } else {
      LOG(ERROR) << "Tool at index " << i
                 << " has empty function name, skipping";
    }
  }
  return indices;
}

std::vector<ToolCallItem> BaseFormatDetector::parse_base_json(
    const nlohmann::json& json_obj,
    const std::vector<JsonTool>& tools) {
  auto tool_indices = get_tool_indices(tools);
  std::vector<ToolCallItem> results;

  std::vector<nlohmann::json> actions;
  if (json_obj.is_array()) {
    for (const auto& item : json_obj) {
      actions.emplace_back(item);
    }
  } else {
    actions.emplace_back(json_obj);
  }

  for (const auto& act : actions) {
    if (!act.is_object()) {
      LOG(ERROR) << "Invalid tool call item, expected object, got: "
                 << act.type_name();
      continue;
    }

    std::string name;
    if (act.contains("name") && act["name"].is_string()) {
      name = act["name"].get<std::string>();
    } else {
      LOG(ERROR) << "Invalid tool call: missing 'name' field or invalid type";
      continue;
    }

    if (tool_indices.find(name) == tool_indices.end()) {
      LOG(ERROR) << "Model attempted to call undefined function: " << name;
      continue;
    }

    nlohmann::json parameters = nlohmann::json::object();

    if (act.contains("parameters")) {
      parameters = act["parameters"];
    } else if (act.contains("arguments")) {
      parameters = act["arguments"];
    } else {
      LOG(ERROR) << "No parameters or arguments field found for tool: " << name;
    }

    if (!parameters.is_object()) {
      LOG(ERROR) << "Invalid arguments type for tool: " << name
                 << ", expected object, got: " << parameters.type_name();
      parameters = nlohmann::json::object();
    }

    std::string parameters_str =
        normalize_parameters_for_tool(name, parameters, tools);

    results.emplace_back(-1, name, parameters_str);
  }

  return results;
}

int32_t BaseFormatDetector::ends_with_partial_token(
    const std::string& buffer,
    const std::string& bot_token) const {
  // Check if buffer ends with a partial bot_token.
  // Return the length of the partial bot_token.
  // For some format, the bot_token is not a token in model's vocabulary, such
  // as
  // `[TOOL_CALLS] [` in Mistral.
  for (int32_t i = 1; i <= std::min(static_cast<int32_t>(buffer.length()),
                                    static_cast<int32_t>(bot_token.length()));
       ++i) {
    if (bot_token.substr(0, i) == buffer.substr(buffer.length() - i)) {
      return i;
    }
  }
  return 0;
}

StreamingParseResult BaseFormatDetector::parse_streaming_increment(
    const std::string& new_text,
    const std::vector<JsonTool>& tools) {
  // Streaming incremental parsing with tool validation.
  // This base implementation works best with formats where:
  // 1. bot_token is followed immediately by JSON (e.g., bot_token + JSON_array)
  // 2. JSON can be parsed incrementally using partial_json_loads
  // 3. Multiple tool calls are separated by "; " or ", "
  //
  // Examples of incompatible formats (need custom implementation, may reuse
  // some logic from this class):
  // - Each tool call is wrapped in a separate block: See Qwen25Detector
  // - Multiple separate blocks: [TOOL_CALLS] [...] \n [TOOL_CALLS] [...]
  // - Tool call is Pythonic style
  //
  // For incompatible formats, detectors should override this method with custom
  // logic.

  // Append new text to buffer
  buffer_ += new_text;
  std::string current_text = buffer_;

  // The current_text has tool_call if it is the start of a new tool call
  // sequence or it is the start of a new tool call after a tool call separator,
  // when there is a previous tool call
  if (!(has_tool_call(current_text) ||
        (current_tool_id_ > 0 &&
         current_text.find(tool_call_separator_) == 0))) {
    if (ends_with_partial_token(buffer_, bot_token_) == 0) {
      std::string normal_text = buffer_;
      buffer_.clear();

      size_t eot_pos = normal_text.find(eot_token_);
      if (eot_pos != std::string::npos) {
        normal_text = normal_text.substr(0, eot_pos) +
                      normal_text.substr(eot_pos + eot_token_.length());
      }

      return StreamingParseResult(normal_text, {});
    } else {
      return StreamingParseResult();
    }
  }

  if (tool_indices_.empty()) {
    tool_indices_ = get_tool_indices(tools);
  }

  Allow flags =
      current_tool_name_sent_ ? Allow::ALL : (Allow::ALL & ~Allow::STR);

  try {
    int32_t start_idx = 0;

    if (current_text.find(bot_token_) == 0) {
      start_idx = bot_token_.length();
    } else if (current_tool_id_ > 0 &&
               current_text.find(tool_call_separator_ + bot_token_) == 0) {
      start_idx = tool_call_separator_.length() + bot_token_.length();
    } else if (current_tool_id_ > 0 &&
               current_text.find(tool_call_separator_) == 0) {
      start_idx = tool_call_separator_.length();
    }

    if (start_idx >= static_cast<int32_t>(current_text.length())) {
      return StreamingParseResult();
    }

    std::string json_part = current_text.substr(start_idx);
    auto [obj, end_idx] = partial_json_loads(json_part, flags);

    bool is_current_complete = is_complete_json(json_part.substr(0, end_idx));

    if (obj.contains("name") && obj["name"].is_string()) {
      std::string tool_name = obj["name"].get<std::string>();
      if (tool_indices_.find(tool_name) == tool_indices_.end()) {
        buffer_.clear();
        current_tool_id_ = -1;
        current_tool_name_sent_ = false;
        if (!streamed_args_for_tool_.empty()) {
          streamed_args_for_tool_.pop_back();
        }
        return StreamingParseResult();
      }
    }

    nlohmann::json current_tool_call = obj;
    if (current_tool_call.contains("parameters")) {
      if (current_tool_call.contains("arguments")) {
        LOG(ERROR) << "Model generated both parameters and arguments";
        return StreamingParseResult();
      }
      current_tool_call["arguments"] = current_tool_call["parameters"];
    }

    if (current_tool_call.empty()) {
      return StreamingParseResult();
    }

    StreamingParseResult res;

    // Case 1: Handle tool name streaming
    if (!current_tool_name_sent_) {
      if (current_tool_call.contains("name") &&
          current_tool_call["name"].is_string()) {
        std::string function_name =
            current_tool_call["name"].get<std::string>();

        if (tool_indices_.find(function_name) != tool_indices_.end()) {
          // If this is a new tool (current_tool_id was -1), initialize it
          if (current_tool_id_ == -1) {
            current_tool_id_ = 0;
            streamed_args_for_tool_.push_back("");
          }
          // If this is a subsequent tool, ensure streamed_args_for_tool is
          // large enough
          else if (current_tool_id_ >=
                   static_cast<int32_t>(streamed_args_for_tool_.size())) {
            while (static_cast<int32_t>(streamed_args_for_tool_.size()) <=
                   current_tool_id_) {
              streamed_args_for_tool_.push_back("");
            }
          }

          // Send the tool name with empty parameters
          res = StreamingParseResult(
              "", {ToolCallItem(current_tool_id_, function_name, "")});
          current_tool_name_sent_ = true;
        } else {
          res = StreamingParseResult();
        }
      } else {
        res = StreamingParseResult();
      }
    }
    // Case 2: Handle streaming arguments
    else {
      if (current_tool_call.contains("arguments")) {
        nlohmann::json cur_arguments = current_tool_call["arguments"];

        // Calculate how much of the arguments we've already streamed
        int sent = streamed_args_for_tool_[current_tool_id_].length();
        std::string cur_args_json = cur_arguments.dump();

        std::string argument_diff;
        int completing_tool_id = current_tool_id_;

        // If the current tool's JSON is complete, send all remaining arguments
        if (is_current_complete) {
          argument_diff = cur_args_json.substr(sent);

          // Only remove the processed portion, keep unprocessed content
          buffer_ = current_text.substr(start_idx + end_idx);

          if (current_tool_id_ < static_cast<int>(prev_tool_call_arr_.size())) {
            prev_tool_call_arr_[current_tool_id_].clear();
          }
          current_tool_name_sent_ = false;
          streamed_args_for_tool_[current_tool_id_] = "";
          current_tool_id_++;
        }
        // If the tool is still being parsed, send incremental changes
        else if (current_tool_id_ <
                 static_cast<int>(prev_tool_call_arr_.size())) {
          auto prev_args_it =
              prev_tool_call_arr_[current_tool_id_].find("arguments");
          if (prev_args_it != prev_tool_call_arr_[current_tool_id_].end()) {
            std::string prev_args_json = prev_args_it->second;
            if (cur_args_json != prev_args_json) {
              std::string prefix =
                  find_common_prefix(prev_args_json, cur_args_json);
              argument_diff = prefix.substr(sent);
            }
          }
        }

        if (!argument_diff.empty()) {
          int tool_index_to_use =
              is_current_complete ? completing_tool_id : current_tool_id_;
          res = StreamingParseResult(
              "",
              {ToolCallItem(tool_index_to_use, std::nullopt, argument_diff)});

          if (!is_current_complete) {
            streamed_args_for_tool_[current_tool_id_] += argument_diff;
          }
        } else {
          res = StreamingParseResult();
        }
      } else {
        res = StreamingParseResult();
      }
    }

    if (current_tool_id_ >= 0) {
      while (static_cast<int>(prev_tool_call_arr_.size()) <= current_tool_id_) {
        prev_tool_call_arr_.push_back({});
      }

      std::unordered_map<std::string, std::string> tool_call_map;
      if (current_tool_call.contains("name") &&
          current_tool_call["name"].is_string()) {
        tool_call_map["name"] = current_tool_call["name"].get<std::string>();
      }
      if (current_tool_call.contains("arguments")) {
        tool_call_map["arguments"] = current_tool_call["arguments"].dump();
      }

      prev_tool_call_arr_[current_tool_id_] = tool_call_map;
    }

    return res;

  } catch (const std::exception& e) {
    return StreamingParseResult();
  }
}

}  // namespace function_call
}  // namespace xllm
