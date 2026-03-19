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

#include "json_array_detector.h"

#include <nlohmann/json.hpp>

namespace xllm {
namespace function_call {

JsonArrayDetector::JsonArrayDetector() {
  bot_token_ = "[";
  eot_token_ = "]";
  tool_call_separator_ = ",";
}

StreamingParseResult JsonArrayDetector::detect_and_parse(
    const std::string& text,
    const std::vector<JsonTool>& tools) {
  try {
    auto json_obj = nlohmann::json::parse(text);
    return StreamingParseResult("", parse_base_json(json_obj, tools));
  } catch (const std::exception&) {
    return StreamingParseResult(text, {});
  }
}

bool JsonArrayDetector::has_tool_call(const std::string& text) {
  return text.find('[') != std::string::npos ||
         text.find('{') != std::string::npos;
}

}  // namespace function_call
}  // namespace xllm
