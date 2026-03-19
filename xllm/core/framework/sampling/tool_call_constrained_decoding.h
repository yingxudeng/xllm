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

#include <torch/torch.h>

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "constrained_decoding.h"
#include "core/framework/tokenizer/tokenizer.h"
#include "function_call/core_types.h"
#include "sampling_params.h"

namespace xllm {

class ToolCallConstrainedDecoding final : public ConstrainedDecoding {
 public:
  ToolCallConstrainedDecoding(
      const Tokenizer& tokenizer,
      int32_t vocab_size,
      torch::ScalarType dtype,
      torch::Device device,
      const std::vector<ToolCallConstraintMode>& modes,
      const std::vector<std::vector<std::string>>& allowed_tool_names_vec,
      const std::vector<std::vector<std::string>>&
          allowed_tool_schema_jsons_vec);

  bool build_mask_cache() override;

  torch::Tensor generate_mask(
      const std::vector<std::vector<int32_t>>& generated_token_list) override;

 private:
  std::vector<std::vector<std::vector<int32_t>>> build_scaffold_tokens() const;
  std::vector<std::vector<int32_t>> build_tool_prefix_paths(
      const function_call::JsonTool& tool) const;
  std::vector<int32_t> encode_text(const std::string& text) const;
  std::vector<function_call::JsonTool> parse_tools_for_sequence(
      size_t index) const;

 private:
  constexpr static float PRE_MASK_FACTOR = -10000.0f;

  const Tokenizer& tokenizer_;
  int32_t vocab_size_;
  torch::ScalarType dtype_;
  torch::Device device_;
  std::vector<ToolCallConstraintMode> modes_;
  std::vector<std::vector<std::string>> allowed_tool_names_vec_;
  std::vector<std::vector<std::string>> allowed_tool_schema_jsons_vec_;
  std::vector<std::vector<std::vector<int32_t>>> scaffold_token_ids_vec_;
};

}  // namespace xllm
