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

#include "sampler.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <limits>

#include "common/global_flags.h"
#include "logits_utils.h"
#include "sampling_params.h"

namespace xllm {
namespace {

torch::Tensor select_sample_rows(const torch::Tensor& tensor,
                                 const SamplingParameters& params) {
  if (!tensor.defined()) {
    return {};
  }
  if (params.selected_token_idxes.numel() == params.sample_idxes.numel()) {
    return tensor;
  }
  return tensor.index_select(/*dim=*/0, params.sample_idxes);
}

void apply_required_tool_choice_bitmask(torch::Tensor& sample_logits,
                                        const SamplingParameters& params) {
  if (!params.required_tool_choice_bitmasks.defined()) {
    return;
  }

  CHECK_EQ(sample_logits.size(0), params.required_tool_choice_bitmasks.size(0));
  CHECK_GT(params.required_tool_choice_bitmask_size, 0);

  const int64_t vocab_size = sample_logits.size(-1);
  auto index_options =
      torch::TensorOptions().device(sample_logits.device()).dtype(torch::kLong);
  auto bitmask_options = torch::TensorOptions()
                             .device(sample_logits.device())
                             .dtype(torch::kInt32);

  auto vocab_indices = torch::arange(vocab_size, index_options);
  auto word_indices = torch::bitwise_right_shift(vocab_indices, 5);
  auto bit_offsets = torch::bitwise_and(vocab_indices, 31);
  auto gathered_words = params.required_tool_choice_bitmasks.index_select(
      /*dim=*/1, word_indices);
  auto bit_values =
      torch::bitwise_left_shift(torch::ones({vocab_size}, bitmask_options),
                                bit_offsets.to(torch::kInt32));
  auto accepted =
      torch::ne(torch::bitwise_and(gathered_words, bit_values.unsqueeze(0)), 0);

  sample_logits.masked_fill_(accepted.logical_not(),
                             -std::numeric_limits<float>::infinity());
}

}  // namespace

SampleOutput Sampler::forward(torch::Tensor& logits,
                              const SamplingParameters& params) const {
  SampleOutput output;
  // apply frequency and presence penalties
  if (params.frequency_penalties.defined()) {
    apply_frequency_presence_penalties(logits,
                                       params.unique_token_ids,
                                       params.unique_token_counts,
                                       params.frequency_penalties,
                                       params.presence_penalties);
  }

  // apply repetition penalties
  if (params.repetition_penalties.defined()) {
    apply_repetition_penalties(
        logits, params.unique_token_ids, params.repetition_penalties);
  }

  torch::Tensor sample_logits = logits;
  if (params.selected_token_idxes.numel() != params.sample_idxes.numel()) {
    sample_logits = logits.index_select(/*dim=*/0, params.sample_idxes);
  }

  // same batch size
  CHECK_EQ(sample_logits.size(0), params.do_sample.size(0));

  auto temperatures = select_sample_rows(params.temperatures, params);
  auto top_k = select_sample_rows(params.top_k, params);
  auto top_p = select_sample_rows(params.top_p, params);

  apply_required_tool_choice_bitmask(sample_logits, params);
  apply_top_k_top_p(sample_logits, temperatures, top_k, top_p);

  auto probs = sample_logits;
  torch::Tensor samples;
  if (params.all_random_sample) {
    // use float32 for probabilities and log probabilities
    probs =
        torch::softmax(sample_logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);
    samples = random_sample(probs);
  } else if (params.all_greedy_sample) {
    samples = greedy_sample(probs);
  } else {
    // use float32 for probabilities and log probabilities
    probs =
        torch::softmax(sample_logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);
    // mixed sample, sample both then choose based on do_sample
    auto random = random_sample(probs);
    auto greedy = greedy_sample(probs);
    samples = torch::where(params.do_sample, random, greedy);
  }
  output.probs = probs.to(logits.dtype());
  output.next_tokens = samples;

  if (params.logprobs) {
    if (FLAGS_enable_qwen3_reranker) {
      int32_t false_id = 2152;  // "no"
      int32_t true_id = 9693;   // "yes"
      auto indices =
          torch::tensor({false_id, true_id}, torch::kLong).to(samples.device());
      sample_logits = sample_logits.index_select(/*dim=*/1, indices);
      auto logprobs = torch::log_softmax(
          sample_logits, /*dim=*/1, /*dtype=*/torch::kFloat32);
      logprobs = logprobs.index({torch::indexing::Slice(), 1});
      output.logprobs = logprobs.view({-1}).exp();
      return output;
    }
    // log_softmax is equivalent to log(softmax) but more numerically stable
    const auto logprobs = torch::log_softmax(
        sample_logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);
    // select the logprobs for each sequence
    auto selected_logprobs = logprobs.gather(/*dim=*/-1, samples.view({-1, 1}));
    output.logprobs = selected_logprobs.view({-1});

    if (params.max_top_logprobs > 0) {
      auto [values, indices] =
          logprobs.topk(params.max_top_logprobs, /*dim=*/-1);
      output.top_logprobs = values;
      output.top_tokens = indices;
    }
  }

  return output;
}

torch::Tensor Sampler::greedy_sample(const torch::Tensor& probs) {
  return probs.argmax(/*dim=*/-1);
}

torch::Tensor Sampler::random_sample(const torch::Tensor& probs) {
#if defined(USE_MLU)
  xllm::kernel::RandomSampleParams params;
  params.logits = probs;
  return xllm::kernel::random_sample(params);
#endif
  if (probs.dim() == 3) {
    auto batch_size = probs.size(0);
    auto seq_len = probs.size(1);
    auto vocab_size = probs.size(2);
    auto flat_probs = probs.reshape({-1, vocab_size});
    auto sampled =
        flat_probs.multinomial(/*num_samples=*/1, /*replacement=*/false);
    return sampled.reshape({batch_size, seq_len});
  } else {
    return probs.multinomial(/*num_samples=*/1, /*replacement=*/false)
        .flatten();
  }
}

}  // namespace xllm
