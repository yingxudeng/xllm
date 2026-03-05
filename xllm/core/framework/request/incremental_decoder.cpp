/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "incremental_decoder.h"

#include <absl/strings/match.h>

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace xllm {

namespace {

constexpr size_t kDecodeLookbackTokens = 16;

std::string decode_with_vocab_guard(const Slice<int32_t>& token_ids,
                                    const Tokenizer& tokenizer,
                                    bool skip_special_tokens) {
  if (token_ids.empty()) {
    return "";
  }

  const size_t tokenizer_vocab_size = tokenizer.vocab_size();
  if (tokenizer_vocab_size == 0) {
    return tokenizer.decode(token_ids, skip_special_tokens);
  }

  std::vector<int32_t> valid_token_ids;
  valid_token_ids.reserve(token_ids.size());
  for (size_t i = 0; i < token_ids.size(); ++i) {
    const int32_t token_id = token_ids[i];
    if (token_id >= 0 && static_cast<size_t>(token_id) < tokenizer_vocab_size) {
      valid_token_ids.push_back(token_id);
      continue;
    }
  }

  if (valid_token_ids.empty()) {
    return "";
  }
  return tokenizer.decode(
      Slice<int32_t>(valid_token_ids.data(), valid_token_ids.size()),
      skip_special_tokens);
}

}  // namespace

IncrementalDecoder::IncrementalDecoder(const std::string_view& prompt,
                                       size_t num_prompt_tokens,
                                       bool echo,
                                       bool skip_special_tokens)
    : prompt_(prompt),
      num_prompt_tokens_(num_prompt_tokens),
      skip_special_tokens_(skip_special_tokens) {
  // if echo is true, set prefix_offset_ and output_offset_ to 0 to print the
  // whole sequence, otherwise set them to the length of the prompt to skip the
  // prompt.
  prefix_offset_ = echo ? 0 : num_prompt_tokens_;
  output_offset_ = echo ? 0 : num_prompt_tokens_;
}

std::string IncrementalDecoder::decode(const Slice<int32_t>& token_ids,
                                       const Tokenizer& tokenizer) {
  std::stringstream ss;
  // return prompt directly if prompt string is not empty
  if (output_offset_ < num_prompt_tokens_ && !prompt_.empty()) {
    // leave 6 tokens for the prefix to defeat cleanup algorithms in decode
    // which decide to add a space or not depending on the surrouding ids.
    prefix_offset_ = num_prompt_tokens_ <= 6 ? 0 : num_prompt_tokens_ - 6;
    output_offset_ = num_prompt_tokens_;
    ss << prompt_;
  }

  // In PD mode, if a prefill token can directly generate characters, the decode
  // phase needs to skip that token. If it cannot, the decode token and that
  // token need to generate characters together.
  if (checking_prefill_token_) {
    const auto prefill_token_text = decode_with_vocab_guard(
        token_ids.slice(output_offset_, output_offset_ + 1),
        tokenizer,
        skip_special_tokens_);
    if (!absl::EndsWith(prefill_token_text, "�")) {
      output_offset_ += 1;
    }
    checking_prefill_token_ = false;
  }

  auto prefix_text =
      decode_with_vocab_guard(token_ids.slice(prefix_offset_, output_offset_),
                              tokenizer,
                              skip_special_tokens_);
  auto new_text = decode_with_vocab_guard(
      token_ids.slice(prefix_offset_), tokenizer, skip_special_tokens_);
  bool has_monotonic_prefix = absl::StartsWith(new_text, prefix_text);

  // Fallback to a small lookback window only when needed. This preserves
  // stable incremental boundaries for normal text while still handling
  // byte-fallback fragments that end with replacement char.
  if (absl::EndsWith(new_text, "�") || new_text.size() <= prefix_text.size() ||
      !has_monotonic_prefix) {
    const size_t decode_start = prefix_offset_ <= kDecodeLookbackTokens
                                    ? 0
                                    : prefix_offset_ - kDecodeLookbackTokens;
    const auto prefix_text_with_lookback =
        decode_with_vocab_guard(token_ids.slice(decode_start, output_offset_),
                                tokenizer,
                                skip_special_tokens_);
    const auto new_text_with_lookback = decode_with_vocab_guard(
        token_ids.slice(decode_start), tokenizer, skip_special_tokens_);

    prefix_text = prefix_text_with_lookback;
    new_text = new_text_with_lookback;
    has_monotonic_prefix = absl::StartsWith(new_text, prefix_text);
  }
  // utf-8 char � at the end means it is a potential unfinished byte sequence
  // from byte fallback tokenization.
  if (has_monotonic_prefix && new_text.size() > prefix_text.size() &&
      !absl::EndsWith(new_text, "�")) {
    prefix_offset_ = output_offset_;
    output_offset_ = token_ids.size();
    // only print the delta text
    ss << new_text.substr(prefix_text.size());
  }
  return ss.str();
}

}  // namespace xllm
