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

#include <glog/logging.h>
#include <google/protobuf/util/json_util.h>

#include <cctype>
#include <functional>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <tuple>
#include <vector>

#include "api_service/stream_output_parser.h"
#include "chat.pb.h"
#include "core/common/types.h"
#include "function_call/function_call.h"

namespace xllm {
namespace api_service {

// Check for unstreamed tool arguments and send them using the provided sender
// This is shared between Chat API and Anthropic API implementations
using SendFunc = std::function<bool(const std::string&, int)>;
inline bool check_for_unstreamed_tool_args(
    std::shared_ptr<StreamOutputParser> stream_parser,
    size_t index,
    SendFunc send_func) {
  auto* parser = stream_parser->get_tool_call_parser(index);
  if (!parser) {
    return true;
  }

  auto* detector = parser->get_detector();
  if (!detector) {
    return true;
  }

  if (!detector->prev_tool_call_arr_.empty() &&
      !detector->streamed_args_for_tool_.empty()) {
    size_t tool_index = detector->prev_tool_call_arr_.size() - 1;
    if (tool_index < detector->streamed_args_for_tool_.size()) {
      const auto& expected_args = detector->prev_tool_call_arr_[tool_index];
      const std::string& actual_args =
          detector->streamed_args_for_tool_[tool_index];

      if (expected_args.find("arguments") != expected_args.end()) {
        const std::string& expected_call = expected_args.at("arguments");

        if (expected_call.length() > actual_args.length()) {
          std::string remaining_call =
              expected_call.substr(actual_args.length());

          if (!remaining_call.empty()) {
            return send_func(remaining_call, static_cast<int>(tool_index));
          }
        }
      }
    }
  }

  return true;
}

struct ToolCallResult {
  std::optional<google::protobuf::RepeatedPtrField<proto::ToolCall>> tool_calls;
  std::string text;
  std::string finish_reason;
};

inline std::tuple<std::string, std::vector<function_call::ToolCallItem>>
try_parse_leading_json_tool_calls(const std::string& text,
                                  const std::vector<xllm::JsonTool>& tools) {
  size_t start = 0;
  while (start < text.size() &&
         std::isspace(static_cast<unsigned char>(text[start])) != 0) {
    ++start;
  }
  if (start >= text.size() || (text[start] != '[' && text[start] != '{')) {
    return {"", {}};
  }

  bool in_string = false;
  bool escaped = false;
  int depth = 0;
  size_t end = std::string::npos;
  for (size_t i = start; i < text.size(); ++i) {
    const char c = text[i];
    if (in_string) {
      if (escaped) {
        escaped = false;
      } else if (c == '\\') {
        escaped = true;
      } else if (c == '"') {
        in_string = false;
      }
      continue;
    }

    if (c == '"') {
      in_string = true;
      continue;
    }
    if (c == '[' || c == '{') {
      ++depth;
      continue;
    }
    if (c == ']' || c == '}') {
      --depth;
      if (depth == 0) {
        end = i;
        break;
      }
    }
  }

  if (end == std::string::npos) {
    return {"", {}};
  }

  const std::string json_prefix = text.substr(start, end - start + 1);
  try {
    function_call::FunctionCallParser json_array_parser(tools, "json_array");
    auto [ignored_text, calls] =
        json_array_parser.parse_non_stream(json_prefix);
    if (calls.empty()) {
      return {"", {}};
    }
    return {ignored_text, std::move(calls)};
  } catch (const std::exception&) {
    return {"", {}};
  }
}

inline ToolCallResult process_tool_calls(
    std::string text,
    const std::vector<xllm::JsonTool>& tools,
    const std::string& parser_format,
    std::string finish_reason,
    google::protobuf::Arena* arena = nullptr) {
  ToolCallResult result;

  function_call::FunctionCallParser parser(tools, parser_format);
  auto leading_json_parse = try_parse_leading_json_tool_calls(text, tools);
  auto leading_json_calls = std::move(std::get<1>(leading_json_parse));

  if (!parser.has_tool_call(text) && leading_json_calls.empty()) {
    result.text = std::move(text);
    result.finish_reason = std::move(finish_reason);
    return result;
  }

  if (finish_reason == "stop" || finish_reason == "function_call") {
    result.finish_reason = "tool_calls";
  } else {
    result.finish_reason = std::move(finish_reason);
  }

  try {
    auto [parsed_text, call_info_list] = parser.parse_non_stream(text);
    bool used_leading_json_fallback = false;
    if (call_info_list.empty() && !leading_json_calls.empty()) {
      parsed_text.clear();
      call_info_list = std::move(leading_json_calls);
      used_leading_json_fallback = true;
    }
    result.text = used_leading_json_fallback ? "" : std::move(parsed_text);

    google::protobuf::RepeatedPtrField<proto::ToolCall> tool_calls;

    for (const auto& call_info : call_info_list) {
      proto::ToolCall* tool_call =
          arena ? google::protobuf::Arena::CreateMessage<proto::ToolCall>(arena)
                : new proto::ToolCall();

      tool_call->set_id(function_call::utils::generate_tool_call_id());
      tool_call->set_type("function");

      auto* function = tool_call->mutable_function();
      if (call_info.name) {
        function->set_name(*call_info.name);
      }
      function->set_arguments(call_info.parameters);

      tool_calls.AddAllocated(tool_call);
    }

    result.tool_calls = std::move(tool_calls);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Tool call parsing error: " << e.what();
  }

  return result;
}

// Convert google::protobuf::Struct to nlohmann::json
inline nlohmann::json struct_to_json(
    const google::protobuf::Struct& pb_struct) {
  std::string json_str;
  google::protobuf::util::JsonPrintOptions options;
  options.preserve_proto_field_names = true;
  auto status = google::protobuf::util::MessageToJsonString(
      pb_struct, &json_str, options);
  if (status.ok()) {
    try {
      return nlohmann::json::parse(json_str);
    } catch (...) {
      return nlohmann::json::object();
    }
  }
  return nlohmann::json::object();
}

}  // namespace api_service
}  // namespace xllm
