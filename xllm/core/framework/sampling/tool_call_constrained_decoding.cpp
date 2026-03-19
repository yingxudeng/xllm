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

#include <glog/logging.h>

#include <algorithm>
#include <iterator>
#include <unordered_set>

#include "util/tensor_helper.h"

namespace xllm {

namespace {

std::string json_dump(const nlohmann::json& value) {
  return value.dump(-1, ' ', false, nlohmann::json::error_handler_t::ignore);
}

const nlohmann::json* get_property_schema(const nlohmann::json& schema,
                                          const std::string& key) {
  if (!schema.is_object() || !schema.contains("properties") ||
      !schema["properties"].is_object() ||
      !schema["properties"].contains(key)) {
    return nullptr;
  }
  return &schema["properties"][key];
}

std::vector<std::string> collect_value_prefixes(const nlohmann::json& schema) {
  if (!schema.is_object()) {
    return {};
  }

  if (schema.contains("default")) {
    const auto full = json_dump(schema["default"]);
    if (!full.empty()) {
      return {full};
    }
  }

  if (schema.contains("enum") && schema["enum"].is_array()) {
    std::vector<std::string> result;
    for (const auto& item : schema["enum"]) {
      const auto dumped = json_dump(item);
      if (!dumped.empty()) {
        result.push_back(dumped);
      }
    }
    if (!result.empty()) {
      return result;
    }
  }

  for (const auto& key : {"anyOf", "oneOf"}) {
    if (!schema.contains(key) || !schema[key].is_array()) {
      continue;
    }
    std::vector<std::string> result;
    for (const auto& branch : schema[key]) {
      auto branch_result = collect_value_prefixes(branch);
      result.insert(result.end(), branch_result.begin(), branch_result.end());
    }
    if (!result.empty()) {
      return result;
    }
  }

  const auto type_name = schema.value("type", "");
  if (type_name == "string") {
    return {"\""};
  }
  if (type_name == "boolean") {
    return {"true", "false"};
  }
  if (type_name == "integer" || type_name == "number") {
    return {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-"};
  }
  if (type_name == "array" || schema.contains("items")) {
    std::vector<std::string> result = {"[]", "["};
    if (schema.contains("items") && schema["items"].is_object()) {
      const auto item_type = schema["items"].value("type", "");
      if (item_type == "string") {
        result.push_back("[\"");
      } else if (item_type == "object" ||
                 schema["items"].contains("properties")) {
        result.push_back("[{");
      }
    }
    return result;
  }
  if (type_name == "object" || schema.contains("properties")) {
    return {"{}", "{"};
  }

  return {};
}

std::vector<std::string> build_required_property_prefixes(
    const function_call::JsonTool& tool) {
  std::vector<std::string> result;
  const auto& schema = tool.function.parameters;
  if (!schema.is_object() || !schema.contains("required") ||
      !schema["required"].is_array() || !schema.contains("properties") ||
      !schema["properties"].is_object()) {
    return result;
  }

  const std::string base =
      "[{\"name\":\"" + tool.function.name + "\",\"parameters\":{";
  for (const auto& required_key : schema["required"]) {
    if (!required_key.is_string()) {
      continue;
    }
    const auto key = required_key.get<std::string>();
    const auto* property_schema = get_property_schema(schema, key);
    if (property_schema == nullptr) {
      continue;
    }

    const std::string key_prefix = base + "\"" + key + "\":";
    const auto value_prefixes = collect_value_prefixes(*property_schema);
    if (value_prefixes.empty()) {
      result.push_back(key_prefix);
    }
    for (const auto& value_prefix : value_prefixes) {
      result.push_back(key_prefix + value_prefix);
      if (!value_prefix.empty() && value_prefix.back() != '"' &&
          value_prefix.back() != '[' && value_prefix.back() != '{' &&
          value_prefix.back() != '-') {
        result.push_back(key_prefix + value_prefix + "}}]");
      } else if (value_prefix == "[]") {
        result.push_back(key_prefix + value_prefix + "}}]");
      } else if (value_prefix == "{}") {
        result.push_back(key_prefix + value_prefix + "}}]");
      }
    }
    if (!result.empty()) {
      break;
    }
  }
  return result;
}

}  // namespace

ToolCallConstrainedDecoding::ToolCallConstrainedDecoding(
    const Tokenizer& tokenizer,
    int32_t vocab_size,
    torch::ScalarType dtype,
    torch::Device device,
    const std::vector<ToolCallConstraintMode>& modes,
    const std::vector<std::vector<std::string>>& allowed_tool_names_vec,
    const std::vector<std::vector<std::string>>& allowed_tool_schema_jsons_vec)
    : tokenizer_(tokenizer),
      vocab_size_(vocab_size),
      dtype_(dtype),
      device_(device),
      modes_(modes),
      allowed_tool_names_vec_(allowed_tool_names_vec),
      allowed_tool_schema_jsons_vec_(allowed_tool_schema_jsons_vec) {}

bool ToolCallConstrainedDecoding::build_mask_cache() {
  scaffold_token_ids_vec_ = build_scaffold_tokens();
  return true;
}

std::vector<function_call::JsonTool>
ToolCallConstrainedDecoding::parse_tools_for_sequence(size_t index) const {
  std::vector<function_call::JsonTool> tools;
  if (index >= allowed_tool_schema_jsons_vec_.size()) {
    return tools;
  }

  tools.reserve(allowed_tool_schema_jsons_vec_[index].size());
  for (const auto& tool_json : allowed_tool_schema_jsons_vec_[index]) {
    if (tool_json.empty()) {
      continue;
    }
    try {
      auto obj = nlohmann::json::parse(tool_json);
      function_call::JsonTool tool;
      tool.type = obj.value("type", "function");
      if (obj.contains("function") && obj["function"].is_object()) {
        const auto& function = obj["function"];
        tool.function.name = function.value("name", "");
        tool.function.description = function.value("description", "");
        if (function.contains("parameters")) {
          tool.function.parameters = function["parameters"];
        }
      }
      if (!tool.function.name.empty()) {
        tools.push_back(std::move(tool));
      }
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to parse tool schema JSON: " << e.what();
    }
  }
  return tools;
}

std::vector<int32_t> ToolCallConstrainedDecoding::encode_text(
    const std::string& text) const {
  std::vector<int32_t> token_ids;
  if (!tokenizer_.encode(text, &token_ids, /*add_special_tokens=*/false)) {
    LOG(ERROR) << "Failed to encode tool-call constraint text: " << text;
    return {};
  }
  return token_ids;
}

std::vector<std::vector<int32_t>>
ToolCallConstrainedDecoding::build_tool_prefix_paths(
    const function_call::JsonTool& tool) const {
  std::vector<std::vector<int32_t>> paths;
  const auto required_prefixes = build_required_property_prefixes(tool);
  for (const auto& prefix : required_prefixes) {
    if (auto encoded = encode_text(prefix); !encoded.empty()) {
      paths.push_back(std::move(encoded));
    }
  }

  if (!paths.empty()) {
    return paths;
  }

  const std::string base =
      "[{\"name\":\"" + tool.function.name + "\",\"parameters\":{";
  if (auto encoded = encode_text(base); !encoded.empty()) {
    paths.push_back(std::move(encoded));
  }

  return paths;
}

std::vector<std::vector<std::vector<int32_t>>>
ToolCallConstrainedDecoding::build_scaffold_tokens() const {
  std::vector<std::vector<std::vector<int32_t>>> scaffolds;
  scaffolds.reserve(allowed_tool_names_vec_.size());
  for (size_t i = 0; i < allowed_tool_names_vec_.size(); ++i) {
    std::vector<std::vector<int32_t>> seq_scaffolds;
    if (modes_[i] == ToolCallConstraintMode::NONE) {
      scaffolds.push_back(std::move(seq_scaffolds));
      continue;
    }

    auto tools = parse_tools_for_sequence(i);
    for (const auto& tool : tools) {
      auto paths = build_tool_prefix_paths(tool);
      seq_scaffolds.insert(seq_scaffolds.end(),
                           std::make_move_iterator(paths.begin()),
                           std::make_move_iterator(paths.end()));
    }

    if (seq_scaffolds.empty()) {
      for (const auto& tool_name : allowed_tool_names_vec_[i]) {
        const auto scaffold =
            encode_text("[{\"name\":\"" + tool_name + "\",\"parameters\":{");
        if (!scaffold.empty()) {
          seq_scaffolds.push_back(scaffold);
        }
      }
    }

    scaffolds.push_back(std::move(seq_scaffolds));
  }
  return scaffolds;
}

torch::Tensor ToolCallConstrainedDecoding::generate_mask(
    const std::vector<std::vector<int32_t>>& generated_token_list) {
  if (generated_token_list.empty() || scaffold_token_ids_vec_.empty()) {
    return torch::Tensor();
  }

  auto options = torch::TensorOptions().dtype(dtype_).device(device_);
  auto mask = torch::zeros(
      {static_cast<int64_t>(generated_token_list.size()), vocab_size_},
      options);

  bool any_constrained = false;
  for (size_t i = 0; i < generated_token_list.size(); ++i) {
    if (i >= scaffold_token_ids_vec_.size() ||
        scaffold_token_ids_vec_[i].empty()) {
      continue;
    }

    const auto& generated = generated_token_list[i];
    std::unordered_set<int32_t> allowed_next_tokens;
    bool exact_terminal = false;
    for (const auto& scaffold_tokens : scaffold_token_ids_vec_[i]) {
      if (generated.size() > scaffold_tokens.size()) {
        continue;
      }
      if (!std::equal(generated.begin(),
                      generated.end(),
                      scaffold_tokens.begin(),
                      scaffold_tokens.begin() + generated.size())) {
        continue;
      }
      if (generated.size() == scaffold_tokens.size()) {
        exact_terminal = true;
        break;
      }
      allowed_next_tokens.insert(scaffold_tokens[generated.size()]);
    }

    if (exact_terminal || allowed_next_tokens.empty()) {
      continue;
    }

    any_constrained = true;
    auto row =
        torch::full({vocab_size_}, PRE_MASK_FACTOR, torch::dtype(dtype_));
    for (int32_t token_id : allowed_next_tokens) {
      if (token_id >= 0 && token_id < vocab_size_) {
        row[token_id] = 0.0f;
      }
    }
    mask.index_put_({static_cast<int64_t>(i)}, safe_to(row, device_, true));
  }

  return any_constrained ? mask : torch::Tensor();
}

}  // namespace xllm
