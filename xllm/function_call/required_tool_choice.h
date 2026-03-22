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

#include <google/protobuf/repeated_field.h>
#include <xgrammar/compiler.h>
#include <xgrammar/matcher.h>

#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "chat.pb.h"
#include "core/common/types.h"
#include "core/framework/tokenizer/tokenizer.h"
#include "core/framework/tokenizer/tokenizer_args.h"

namespace xllm {
namespace function_call {

struct RequiredToolCallDelta {
  std::optional<std::string> name;
  std::string arguments;
  int32_t tool_index = 0;
};

struct RequiredToolCallStreamingResult {
  std::vector<RequiredToolCallDelta> calls;
};

nlohmann::json get_required_tool_choice_json_schema(
    const std::vector<xllm::JsonTool>& tools);

std::optional<google::protobuf::RepeatedPtrField<proto::ToolCall>>
parse_required_tool_choice_tool_calls(const std::string& text,
                                      google::protobuf::Arena* arena = nullptr);

RequiredToolCallStreamingResult extract_required_tool_choice_streaming_delta(
    const std::string& previous_text,
    const std::string& current_text,
    const std::string& delta_text,
    bool* function_name_returned);

class RequiredToolChoiceGrammar {
 public:
  RequiredToolChoiceGrammar(xgrammar::CompiledGrammar compiled_grammar,
                            int32_t vocab_size);

  xgrammar::GrammarMatcher create_matcher() const;

  int32_t vocab_size() const { return vocab_size_; }

  int32_t bitmask_size() const { return bitmask_size_; }

 private:
  xgrammar::CompiledGrammar compiled_grammar_;
  int32_t vocab_size_ = 0;
  int32_t bitmask_size_ = 0;
};

class RequiredToolChoiceGrammarFactory {
 public:
  RequiredToolChoiceGrammarFactory(const TokenizerArgs& tokenizer_args,
                                   const Tokenizer& tokenizer,
                                   int32_t model_vocab_size,
                                   int32_t eos_token_id);

  std::shared_ptr<const RequiredToolChoiceGrammar> create(
      const std::vector<xllm::JsonTool>& tools);

 private:
  xgrammar::TokenizerInfo build_tokenizer_info(
      const TokenizerArgs& tokenizer_args,
      const Tokenizer& tokenizer,
      int32_t eos_token_id) const;

  int32_t vocab_size_ = 0;
  mutable std::mutex mutex_;
  std::unique_ptr<xgrammar::GrammarCompiler> compiler_;
  std::unordered_map<std::string,
                     std::weak_ptr<const RequiredToolChoiceGrammar>>
      grammar_cache_;
};

class RequiredToolChoiceMatcher {
 public:
  explicit RequiredToolChoiceMatcher(
      std::shared_ptr<const RequiredToolChoiceGrammar> grammar);
  RequiredToolChoiceMatcher(const RequiredToolChoiceMatcher& other);

  bool accept_token(int32_t token_id);

  bool fill_next_token_bitmask(std::vector<int32_t>* bitmask);

  int32_t bitmask_size() const;

  void reset();

 private:
  std::shared_ptr<const RequiredToolChoiceGrammar> grammar_;
  xgrammar::GrammarMatcher matcher_;
};

}  // namespace function_call
}  // namespace xllm
