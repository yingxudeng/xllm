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

#include "request_params.h"

#include <gtest/gtest.h>

namespace xllm {
namespace {

TEST(RequestParamsTest, RejectsRequiredToolChoiceWithoutTools) {
  RequestParams params;
  params.tool_choice = "required";

  Status captured_status;
  bool callback_called = false;
  auto callback = [&](const RequestOutput& output) -> bool {
    callback_called = true;
    captured_status = output.status.value_or(Status(StatusCode::OK));
    return true;
  };

  EXPECT_FALSE(params.verify_params(callback));
  EXPECT_TRUE(callback_called);
  EXPECT_EQ(captured_status.code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(captured_status.message(),
            "tools cannot be empty when tool_choice is required");
}

TEST(RequestParamsTest, AllowsRequiredToolChoiceWithTools) {
  RequestParams params;
  params.tool_choice = "required";

  JsonTool tool;
  tool.type = "function";
  tool.function.name = "reply";
  tool.function.parameters = nlohmann::json::object();
  params.tools = {tool};

  auto callback = [](const RequestOutput&) -> bool { return true; };
  EXPECT_TRUE(params.verify_params(callback));
}

}  // namespace
}  // namespace xllm
