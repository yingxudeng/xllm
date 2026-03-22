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

#include "required_tool_choice.h"

#include <dlpack/dlpack.h>
#include <glog/logging.h>

#include <fstream>
#include <functional>
#include <limits>
#include <regex>
#include <utility>

#include "function_call/function_call.h"
#include "function_call/utils.h"

namespace xllm {
namespace function_call {
namespace {

constexpr int32_t kXgrammarMaxThreads = 8;

bool is_byte_level_tokenizer_json(const nlohmann::json& tokenizer_json,
                                  bool* add_prefix_space) {
  bool found = false;

  std::function<void(const nlohmann::json&)> visit =
      [&](const nlohmann::json& node) {
        if (!node.is_object()) {
          if (node.is_array()) {
            for (const auto& item : node) {
              visit(item);
            }
          }
          return;
        }

        auto type_it = node.find("type");
        if (type_it != node.end() && type_it->is_string() &&
            type_it->get<std::string>() == "ByteLevel") {
          found = true;
          if (node.contains("add_prefix_space") &&
              node["add_prefix_space"].is_boolean()) {
            *add_prefix_space = node["add_prefix_space"].get<bool>();
          }
        }

        for (const auto& [_, value] : node.items()) {
          visit(value);
        }
      };

  visit(tokenizer_json);
  return found;
}

xgrammar::VocabType detect_vocab_type(const TokenizerArgs& tokenizer_args,
                                      bool* add_prefix_space) {
  *add_prefix_space = false;

  if (tokenizer_args.tokenizer_type() == "sentencepiece") {
    return xgrammar::VocabType::BYTE_FALLBACK;
  }
  if (tokenizer_args.tokenizer_type() == "tiktoken" ||
      tokenizer_args.tokenizer_class() == "TikTokenTokenizer") {
    return xgrammar::VocabType::RAW;
  }
  if (tokenizer_args.tokenizer_type() != "fast") {
    return xgrammar::VocabType::RAW;
  }

  try {
    nlohmann::json tokenizer_json =
        nlohmann::json::parse(std::ifstream(tokenizer_args.vocab_file()));
    if (is_byte_level_tokenizer_json(tokenizer_json, add_prefix_space)) {
      return xgrammar::VocabType::BYTE_LEVEL;
    }
  } catch (const std::exception& e) {
    LOG(WARNING) << "Failed to inspect tokenizer json for xgrammar metadata: "
                 << e.what();
  }

  return xgrammar::VocabType::RAW;
}

nlohmann::json get_tool_schema_from_tool(const xllm::JsonTool& tool) {
  nlohmann::json params = tool.function.parameters;
  if (params.is_null() || params.empty()) {
    params = nlohmann::json{{"type", "object"}, {"properties", {}}};
  } else if (params.is_object()) {
    params.erase("$defs");
  }
  return {
      {"properties",
       {{"name", {{"type", "string"}, {"enum", {tool.function.name}}}},
        {"parameters", std::move(params)}}},
      {"required", {"name", "parameters"}},
  };
}

nlohmann::json get_tool_schema_defs(const std::vector<xllm::JsonTool>& tools) {
  nlohmann::json all_defs = nlohmann::json::object();
  for (const auto& tool : tools) {
    if (!tool.function.parameters.is_object()) {
      continue;
    }
    auto defs_it = tool.function.parameters.find("$defs");
    if (defs_it == tool.function.parameters.end() || !defs_it->is_object()) {
      continue;
    }
    for (const auto& [name, schema] : defs_it->items()) {
      auto existing = all_defs.find(name);
      if (existing != all_defs.end() && *existing != schema) {
        throw std::invalid_argument(
            "Tool definition '" + name +
            "' has multiple schemas, which is not supported.");
      }
      all_defs[name] = schema;
    }
  }
  return all_defs;
}

int bracket_level(const std::string& text) {
  int level = 0;
  for (char ch : text) {
    if (ch == '{') {
      ++level;
    } else if (ch == '}') {
      --level;
    }
  }
  return level;
}

std::pair<std::string, bool> filter_delta_text(
    const std::string& delta_text,
    const std::string& previous_text) {
  int level = bracket_level(previous_text);
  bool passed_zero = false;
  std::string updated_delta;
  updated_delta.reserve(delta_text.size());
  for (char ch : delta_text) {
    if (ch == '{') {
      ++level;
      passed_zero = level == 0;
    } else if (ch == '}') {
      --level;
      passed_zero = level == 0;
    }

    if (level != 0) {
      updated_delta.push_back(ch);
    } else if (ch == ',') {
      break;
    }
  }
  return {updated_delta, passed_zero};
}

std::string extract_parameters_suffix(const std::string& current_text) {
  static const std::regex kParametersRegex(
      R"regex(.*"parameters"\s*:\s*(.*))regex", std::regex::ECMAScript);
  std::smatch match;
  if (std::regex_match(current_text, match, kParametersRegex) &&
      match.size() >= 2) {
    return match[1].str();
  }
  return "";
}

std::vector<std::string> build_encoded_vocab(const Tokenizer& tokenizer,
                                             int32_t tokenizer_vocab_size,
                                             int32_t model_vocab_size) {
  CHECK_GE(model_vocab_size, tokenizer_vocab_size)
      << "model vocab size must be >= tokenizer vocab size";
  std::vector<std::string> encoded_vocab(model_vocab_size);
  for (int32_t token_id = 0; token_id < tokenizer_vocab_size; ++token_id) {
    encoded_vocab[token_id] = tokenizer.id_to_token(token_id);
  }
  return encoded_vocab;
}

DLTensor build_cpu_bitmask_tensor(std::vector<int32_t>* bitmask) {
  static_assert(sizeof(int32_t) == sizeof(uint32_t));
  auto* shape = new int64_t[1]{static_cast<int64_t>(bitmask->size())};
  auto* strides = new int64_t[1]{1};
  DLTensor tensor;
  tensor.data = bitmask->data();
  tensor.device = DLDevice{kDLCPU, 0};
  tensor.ndim = 1;
  tensor.dtype = xgrammar::GetBitmaskDLType();
  tensor.shape = shape;
  tensor.strides = strides;
  tensor.byte_offset = 0;
  return tensor;
}

void release_cpu_bitmask_tensor(DLTensor* tensor) {
  delete[] tensor->shape;
  delete[] tensor->strides;
}

}  // namespace

nlohmann::json get_required_tool_choice_json_schema(
    const std::vector<xllm::JsonTool>& tools) {
  nlohmann::json schema = {
      {"type", "array"},
      {"minItems", 1},
      {"items", {{"type", "object"}, {"anyOf", nlohmann::json::array()}}},
  };

  for (const auto& tool : tools) {
    schema["items"]["anyOf"].push_back(get_tool_schema_from_tool(tool));
  }

  nlohmann::json defs = get_tool_schema_defs(tools);
  if (!defs.empty()) {
    schema["$defs"] = std::move(defs);
  }
  return schema;
}

std::optional<google::protobuf::RepeatedPtrField<proto::ToolCall>>
parse_required_tool_choice_tool_calls(const std::string& text,
                                      google::protobuf::Arena* arena) {
  try {
    nlohmann::json parsed = nlohmann::json::parse(text);
    if (!parsed.is_array() || parsed.empty()) {
      return std::nullopt;
    }

    google::protobuf::RepeatedPtrField<proto::ToolCall> tool_calls;
    for (const auto& item : parsed) {
      if (!item.is_object() || !item.contains("name") ||
          !item.contains("parameters") || !item["name"].is_string()) {
        return std::nullopt;
      }

      proto::ToolCall* tool_call =
          arena ? google::protobuf::Arena::CreateMessage<proto::ToolCall>(arena)
                : new proto::ToolCall();
      tool_call->set_id(function_call::utils::generate_tool_call_id());
      tool_call->set_type("function");
      tool_call->mutable_function()->set_name(item["name"].get<std::string>());
      tool_call->mutable_function()->set_arguments(item["parameters"].dump());
      tool_calls.AddAllocated(tool_call);
    }

    return tool_calls;
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to parse required tool choice output: " << e.what();
    return std::nullopt;
  }
}

RequiredToolCallStreamingResult extract_required_tool_choice_streaming_delta(
    const std::string& previous_text,
    const std::string& current_text,
    const std::string& delta_text,
    bool* function_name_returned) {
  RequiredToolCallStreamingResult result;
  if (function_name_returned == nullptr || current_text.empty()) {
    return result;
  }

  nlohmann::json obj;
  try {
    std::tie(obj, std::ignore) = partial_json_loads(current_text, Allow::ALL);
  } catch (const std::exception&) {
    return result;
  }

  if (!obj.is_array() || obj.empty()) {
    *function_name_returned = false;
    return result;
  }

  auto [_, finishes_previous_tool] =
      filter_delta_text(delta_text, previous_text);
  nlohmann::json current_tool_call = obj.back();
  const bool has_name = current_tool_call.contains("name") &&
                        current_tool_call["name"].is_string();
  const bool has_parameters = current_tool_call.contains("parameters");

  if (!finishes_previous_tool && (!has_name || !has_parameters)) {
    *function_name_returned = false;
    return result;
  }

  RequiredToolCallDelta delta;
  delta.tool_index = static_cast<int32_t>(obj.size() - 1);

  if (!*function_name_returned) {
    std::string arguments = extract_parameters_suffix(current_text);
    arguments = filter_delta_text(arguments, previous_text).first;

    if (finishes_previous_tool && !has_parameters && obj.size() >= 2) {
      current_tool_call = obj[obj.size() - 2];
    }

    if (!current_tool_call.contains("name") ||
        !current_tool_call["name"].is_string()) {
      return result;
    }

    delta.name = current_tool_call["name"].get<std::string>();
    delta.arguments = std::move(arguments);
    result.calls.push_back(std::move(delta));
    *function_name_returned = true;
    return result;
  }

  delta.arguments = filter_delta_text(delta_text, previous_text).first;
  if (!delta.arguments.empty()) {
    result.calls.push_back(std::move(delta));
  }
  return result;
}

RequiredToolChoiceGrammar::RequiredToolChoiceGrammar(
    xgrammar::CompiledGrammar compiled_grammar,
    int32_t vocab_size)
    : compiled_grammar_(std::move(compiled_grammar)),
      vocab_size_(vocab_size),
      bitmask_size_(xgrammar::GetBitmaskSize(vocab_size)) {}

xgrammar::GrammarMatcher RequiredToolChoiceGrammar::create_matcher() const {
  return xgrammar::GrammarMatcher(compiled_grammar_);
}

RequiredToolChoiceGrammarFactory::RequiredToolChoiceGrammarFactory(
    const TokenizerArgs& tokenizer_args,
    const Tokenizer& tokenizer,
    int32_t model_vocab_size,
    int32_t eos_token_id)
    : vocab_size_(model_vocab_size) {
  auto tokenizer_info =
      build_tokenizer_info(tokenizer_args, tokenizer, eos_token_id);
  compiler_ = std::make_unique<xgrammar::GrammarCompiler>(
      tokenizer_info, kXgrammarMaxThreads, true, -1);
}

std::shared_ptr<const RequiredToolChoiceGrammar>
RequiredToolChoiceGrammarFactory::create(
    const std::vector<xllm::JsonTool>& tools) {
  const std::string schema = get_required_tool_choice_json_schema(tools).dump();

  std::lock_guard<std::mutex> lock(mutex_);
  auto cache_it = grammar_cache_.find(schema);
  if (cache_it != grammar_cache_.end()) {
    auto cached = cache_it->second.lock();
    if (cached != nullptr) {
      return cached;
    }
  }

  xgrammar::CompiledGrammar compiled = compiler_->CompileJSONSchema(schema);
  auto grammar = std::make_shared<RequiredToolChoiceGrammar>(
      std::move(compiled), vocab_size_);
  grammar_cache_[schema] = grammar;
  return grammar;
}

xgrammar::TokenizerInfo RequiredToolChoiceGrammarFactory::build_tokenizer_info(
    const TokenizerArgs& tokenizer_args,
    const Tokenizer& tokenizer,
    int32_t eos_token_id) const {
  bool add_prefix_space = false;
  xgrammar::VocabType vocab_type =
      detect_vocab_type(tokenizer_args, &add_prefix_space);
  std::optional<std::vector<int32_t>> stop_token_ids = std::nullopt;
  if (eos_token_id >= 0) {
    stop_token_ids = std::vector<int32_t>{eos_token_id};
  }
  const int32_t tokenizer_vocab_size =
      static_cast<int32_t>(tokenizer.vocab_size());
  return xgrammar::TokenizerInfo(
      build_encoded_vocab(tokenizer, tokenizer_vocab_size, vocab_size_),
      vocab_type,
      vocab_size_,
      std::move(stop_token_ids),
      add_prefix_space);
}

RequiredToolChoiceMatcher::RequiredToolChoiceMatcher(
    std::shared_ptr<const RequiredToolChoiceGrammar> grammar)
    : grammar_(std::move(grammar)), matcher_(grammar_->create_matcher()) {}

RequiredToolChoiceMatcher::RequiredToolChoiceMatcher(
    const RequiredToolChoiceMatcher& other)
    : grammar_(other.grammar_), matcher_(other.matcher_.Fork()) {}

bool RequiredToolChoiceMatcher::accept_token(int32_t token_id) {
  if (token_id < 0) {
    return true;
  }
  return matcher_.AcceptToken(token_id);
}

bool RequiredToolChoiceMatcher::fill_next_token_bitmask(
    std::vector<int32_t>* bitmask) {
  if (bitmask == nullptr) {
    return false;
  }
  bitmask->assign(grammar_->bitmask_size(), 0);
  DLTensor tensor = build_cpu_bitmask_tensor(bitmask);
  const bool need_apply = matcher_.FillNextTokenBitmask(&tensor);
  release_cpu_bitmask_tensor(&tensor);
  return need_apply;
}

int32_t RequiredToolChoiceMatcher::bitmask_size() const {
  return grammar_->bitmask_size();
}

void RequiredToolChoiceMatcher::reset() { matcher_.Reset(); }

}  // namespace function_call
}  // namespace xllm
