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

#include "stopping_checker.h"

#include <gtest/gtest.h>

#include <vector>

namespace xllm {
namespace {

TEST(StoppingCheckerTest, StopsOnEosToken) {
  StoppingChecker checker(/*max_generated_tokens=*/128,
                          /*max_context_len=*/0,
                          /*eos_token=*/42,
                          /*ignore_eos=*/false,
                          /*stop_tokens=*/{},
                          /*stop_sequences=*/{});

  std::vector<int32_t> token_ids = {1, 2, 42};
  EXPECT_EQ(checker.check(token_ids, /*num_prompt_tokens=*/1),
            FinishReason::STOP);
}

TEST(StoppingCheckerTest, StopsOnStopSequence) {
  StoppingChecker checker(/*max_generated_tokens=*/128,
                          /*max_context_len=*/0,
                          /*eos_token=*/-1,
                          /*ignore_eos=*/false,
                          /*stop_tokens=*/{},
                          /*stop_sequences=*/{{11, 12, 13}});

  std::vector<int32_t> token_ids = {1, 10, 11, 12, 13};
  EXPECT_EQ(checker.check(token_ids, /*num_prompt_tokens=*/1),
            FinishReason::STOP);
}

TEST(StoppingCheckerTest, StopsOnMaxGeneratedTokens) {
  StoppingChecker checker(/*max_generated_tokens=*/2,
                          /*max_context_len=*/0,
                          /*eos_token=*/-1,
                          /*ignore_eos=*/false,
                          /*stop_tokens=*/{},
                          /*stop_sequences=*/{});

  std::vector<int32_t> token_ids = {1, 2, 3};
  EXPECT_EQ(checker.check(token_ids, /*num_prompt_tokens=*/1),
            FinishReason::LENGTH);
}

}  // namespace
}  // namespace xllm
