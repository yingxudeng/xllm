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

#include "framework/request/incremental_decoder.h"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "framework/tokenizer/tokenizer.h"

namespace xllm {
namespace {

class FakeContextTokenizer final : public Tokenizer {
 public:
  std::string decode(const Slice<int32_t>& ids,
                     bool /*skip_special_tokens*/) const override {
    const std::vector<int32_t> v(ids.begin(), ids.end());
    if (v == std::vector<int32_t>{99}) {
      return "P";
    }
    if (v == std::vector<int32_t>{10}) {
      return "A";
    }
    if (v == std::vector<int32_t>{11}) {
      return "B";
    }
    if (v == std::vector<int32_t>{20}) {
      return "�";
    }
    if (v == std::vector<int32_t>{99, 10}) {
      return "PA";
    }
    if (v == std::vector<int32_t>{10, 11}) {
      return "AB";
    }
    if (v == std::vector<int32_t>{11, 20}) {
      return "B�";
    }
    if (v == std::vector<int32_t>{99, 10, 11}) {
      return "PAB";
    }
    if (v == std::vector<int32_t>{10, 11, 20}) {
      return "AB中";
    }
    if (v == std::vector<int32_t>{99, 10, 11, 20}) {
      return "PAB中";
    }
    if (v == std::vector<int32_t>{12}) {
      return "Z";
    }
    if (v == std::vector<int32_t>{11, 12}) {
      return "XY";
    }
    if (v == std::vector<int32_t>{11, 12, 13}) {
      return "BCD";
    }
    if (v == std::vector<int32_t>{99, 10, 11, 12}) {
      return "PXY";
    }
    if (v == std::vector<int32_t>{99, 10, 11, 12, 13}) {
      return "PABCD";
    }
    return "";
  }

  size_t vocab_size() const override { return 128; }

  std::unique_ptr<Tokenizer> clone() const override {
    return std::make_unique<FakeContextTokenizer>(*this);
  }
};

TEST(IncrementalDecoderTest, KeepLookbackContextToAvoidReplacementChar) {
  FakeContextTokenizer tokenizer;
  IncrementalDecoder decoder("",
                             /*num_prompt_tokens=*/1,
                             /*echo=*/false,
                             /*skip_special_tokens=*/true);

  std::vector<int32_t> ids1 = {99, 10};
  EXPECT_EQ(decoder.decode(ids1, tokenizer), "A");

  std::vector<int32_t> ids2 = {99, 10, 11};
  EXPECT_EQ(decoder.decode(ids2, tokenizer), "B");

  // Without lookback context, detokenizing [11, 20] yields "B�".
  // With lookback context, detokenizing [10, 11, 20] yields "AB中".
  std::vector<int32_t> ids3 = {99, 10, 11, 20};
  EXPECT_EQ(decoder.decode(ids3, tokenizer), "中");
}

TEST(IncrementalDecoderTest, SkipTransientNonMonotonicDecodeText) {
  FakeContextTokenizer tokenizer;
  IncrementalDecoder decoder("",
                             /*num_prompt_tokens=*/1,
                             /*echo=*/false,
                             /*skip_special_tokens=*/true);

  std::vector<int32_t> ids1 = {99, 10};
  EXPECT_EQ(decoder.decode(ids1, tokenizer), "A");

  std::vector<int32_t> ids2 = {99, 10, 11};
  EXPECT_EQ(decoder.decode(ids2, tokenizer), "B");

  // Decoding [11, 12] returns "XY", which is not a monotonic extension of
  // decoding [11] -> "B". Decoder should delay emitting corrupted delta.
  std::vector<int32_t> ids3 = {99, 10, 11, 12};
  EXPECT_EQ(decoder.decode(ids3, tokenizer), "");

  // Once context recovers, decoder resumes monotonic output.
  std::vector<int32_t> ids4 = {99, 10, 11, 12, 13};
  EXPECT_EQ(decoder.decode(ids4, tokenizer), "CD");
}

}  // namespace
}  // namespace xllm
