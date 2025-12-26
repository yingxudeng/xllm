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

#pragma once
#include <torch/torch.h>

#include <cstdint>
#include <vector>

#include "core/util/tensor_helper.h"

namespace xllm {

struct RequestSamplingParam {
  float frequency_penalty = 0.0;
  float presence_penalty = 0.0;
  float repetition_penalty = 1.0;
  float temperature = 0.0;
  float top_p = 1.0;
  int64_t top_k = -1;
  bool logprobs = false;
  int64_t top_logprobs = 0;
  bool do_sample = false;
  bool is_embeddings = false;
  int32_t beam_width = 0;
};

struct SamplingParameters {
 public:
  void init(const std::vector<const RequestSamplingParam*>& req_sampling_params,
            const std::vector<int32_t>& selected_token_idxes,
            const std::vector<int32_t>& sample_idxes,
            const std::vector<std::vector<int64_t>>& unique_token_ids_vec,
            const std::vector<std::vector<int32_t>>& unique_token_counts_vec,
            const std::vector<int32_t>& unique_token_lens_vec);

  SamplingParameters to(const torch::Device& device,
                        torch::ScalarType dtype) const {
    SamplingParameters params;

    // all tensors should be on the same device
    params.selected_token_idxes = safe_to(selected_token_idxes, device, true);

    auto options = torch::device(device).dtype(dtype);
    params.frequency_penalties = safe_to(frequency_penalties, options, true);
    params.presence_penalties = safe_to(presence_penalties, options, true);
    params.repetition_penalties = safe_to(repetition_penalties, options, true);
    params.temperatures = safe_to(temperatures, options, true);
    params.top_p = safe_to(top_p, options, true);
    params.top_k = safe_to(top_k, device, true);

    params.unique_token_ids = safe_to(unique_token_ids, device, true);
    params.unique_token_counts = safe_to(unique_token_counts, device, true);
    params.unique_token_ids_lens = safe_to(unique_token_ids_lens, device, true);

    params.sample_idxes = safe_to(sample_idxes, device, true);
    params.do_sample = safe_to(do_sample, device, true);
    if (params.do_sample.defined()) {
      params.all_random_sample = params.do_sample.all().item<bool>();
      params.all_greedy_sample = !params.do_sample.any().item<bool>();
    }
    params.logprobs = logprobs;
    params.max_top_logprobs = max_top_logprobs;
    params.is_embeddings = is_embeddings;

    // for beam search
    params.use_beam_search = use_beam_search;
    return params;
  }

  // concat two SamplingParameters into one
  void concat(const SamplingParameters& param);

  // selected tokens are tokens for sampling the next token,
  // including the generated tokens and the last prompt token
  // IntTensor
  torch::Tensor selected_token_idxes;

  // [num_tokens] FloatTensor
  torch::Tensor frequency_penalties;

  // [num_tokens] FloatTensor
  torch::Tensor presence_penalties;

  // [num_tokens] FloatTensor
  torch::Tensor repetition_penalties;

  // [num_tokens] FloatTensor
  torch::Tensor temperatures;

  // [num_tokens] FloatTensor
  torch::Tensor top_p;

  // [num_tokens] LongTensor
  torch::Tensor top_k;

  // the unique token id and count of each sequence in the batch.
  // [num_tokens, max_unique_tokens] LongTensor
  torch::Tensor unique_token_ids;

  // [num_tokens, max_unique_tokens] IntTensor
  torch::Tensor unique_token_counts;

  // the number of unique tokens in each sequence.
  // [num_tokens] IntTensor
  torch::Tensor unique_token_ids_lens;

  // the last index of the selected tokens for sampling.
  // [num_seqs] IntTensor
  torch::Tensor sample_idxes;

  // whether to sample for each sequence.
  // [num_seqs] BoolTensor
  torch::Tensor do_sample;
  bool all_random_sample = false;
  bool all_greedy_sample = true;

  // whether to output logprobs for each generated token.
  bool logprobs = false;

  // wheteher to get the embeddings of the tokens. used by embeddings model.
  bool is_embeddings = false;

  // max number of top logprobs in the batch.
  // only used when logprobs is true.
  int64_t max_top_logprobs = 0;

  // for beam search
  bool use_beam_search = false;

  void print() const {
    LOG(INFO) << "SamplingParameters {";
    
    // Tensor Helper Logic (inline expansion for safety)
    if (selected_token_idxes.defined()) {
      LOG(INFO) << "  selected_token_idxes: " << selected_token_idxes
                << " " << selected_token_idxes.scalar_type();
    } else {
      LOG(INFO) << "  selected_token_idxes: (undefined)";
    }

    if (frequency_penalties.defined()) {
      LOG(INFO) << "  frequency_penalties: " << frequency_penalties
                << " " << frequency_penalties.scalar_type();
    } else {
      LOG(INFO) << "  frequency_penalties: (undefined)";
    }

    if (presence_penalties.defined()) {
      LOG(INFO) << "  presence_penalties: " << presence_penalties
                << " " << presence_penalties.scalar_type();
    } else {
      LOG(INFO) << "  presence_penalties: (undefined)";
    }

    if (repetition_penalties.defined()) {
      LOG(INFO) << "  repetition_penalties: " << repetition_penalties
                << " " << repetition_penalties.scalar_type();
    } else {
      LOG(INFO) << "  repetition_penalties: (undefined)";
    }

    if (temperatures.defined()) {
      LOG(INFO) << "  temperatures: " << temperatures
                << " " << temperatures.scalar_type();
    } else {
      LOG(INFO) << "  temperatures: (undefined)";
    }

    if (top_p.defined()) {
      LOG(INFO) << "  top_p: " << top_p
                << " " << top_p.scalar_type();
    } else {
      LOG(INFO) << "  top_p: (undefined)";
    }

    if (top_k.defined()) {
      LOG(INFO) << "  top_k: " << top_k
                << " " << top_k.scalar_type();
    } else {
      LOG(INFO) << "  top_k: (undefined)";
    }

    if (unique_token_ids.defined()) {
      LOG(INFO) << "  unique_token_ids: " << unique_token_ids
                << " " << unique_token_ids.scalar_type();
    } else {
      LOG(INFO) << "  unique_token_ids: (undefined)";
    }

    if (unique_token_counts.defined()) {
      LOG(INFO) << "  unique_token_counts: " << unique_token_counts
                << " " << unique_token_counts.scalar_type();
    } else {
      LOG(INFO) << "  unique_token_counts: (undefined)";
    }

    if (unique_token_ids_lens.defined()) {
      LOG(INFO) << "  unique_token_ids_lens: " << unique_token_ids_lens
                << " " << unique_token_ids_lens.scalar_type();
    } else {
      LOG(INFO) << "  unique_token_ids_lens: (undefined)";
    }

    if (sample_idxes.defined()) {
      LOG(INFO) << "  sample_idxes: " << sample_idxes
                << " " << sample_idxes.scalar_type();
    } else {
      LOG(INFO) << "  sample_idxes: (undefined)";
    }

    if (do_sample.defined()) {
      LOG(INFO) << "  do_sample: " << do_sample
                << " " << do_sample.scalar_type();
    } else {
      LOG(INFO) << "  do_sample: (undefined)";
    }

    // Scalar values
    LOG(INFO) << "  all_random_sample: " << (all_random_sample ? "true" : "false");
    LOG(INFO) << "  all_greedy_sample: " << (all_greedy_sample ? "true" : "false");
    LOG(INFO) << "  logprobs: " << (logprobs ? "true" : "false");
    LOG(INFO) << "  is_embeddings: " << (is_embeddings ? "true" : "false");
    LOG(INFO) << "  max_top_logprobs: " << max_top_logprobs;
    LOG(INFO) << "  use_beam_search: " << (use_beam_search ? "true" : "false");

    LOG(INFO) << "}";
  }
};

struct SampleOutput {
  // [num_seq, ...] LongTensor
  torch::Tensor next_tokens;

  // [num_seq, ...] FloatTensor
  torch::Tensor probs;

  // [num_seq, ...] FloatTensor
  torch::Tensor logprobs;

  // [num_seq, ..., top_k] FloatTensor
  torch::Tensor top_logprobs;
  // [num_seq, ..., top_k] LongTensor
  torch::Tensor top_tokens;

  // [num_seq, ..., embed_dim] FloatTensor
  torch::Tensor embeddings;

  void print() const {
    LOG(INFO) << "SampleOutput {";
    if (next_tokens.defined()) {
      LOG(INFO) << "  next_tokens: " << next_tokens
                << " " << next_tokens.scalar_type();
    } else {
      LOG(INFO) << "  next_tokens: (undefined)";
    }

    if (probs.defined()) {
      LOG(INFO) << "  probs: " << probs.sizes()
                << " " << probs.scalar_type();
    } else {
      LOG(INFO) << "  probs: (undefined)";
    }

    if (logprobs.defined()) {
      LOG(INFO) << "  logprobs: " << logprobs
                << " " << logprobs.scalar_type();
    } else {
      LOG(INFO) << "  logprobs: (undefined)";
    }

    if (top_logprobs.defined()) {
      LOG(INFO) << "  top_logprobs: " << top_logprobs
                << " " << top_logprobs.scalar_type();
    } else {
      LOG(INFO) << "  top_logprobs: (undefined)";
    }

    if (top_tokens.defined()) {
      LOG(INFO) << "  top_tokens: " << top_tokens
                << " " << top_tokens.scalar_type();
    } else {
      LOG(INFO) << "  top_tokens: (undefined)";
    }

    if (embeddings.defined()) {
      LOG(INFO) << "  embeddings: " << embeddings
                << " " << embeddings.scalar_type();
    } else {
      LOG(INFO) << "  embeddings: (undefined)";
    }
    LOG(INFO) << "}";
  }
};

}  // namespace xllm
