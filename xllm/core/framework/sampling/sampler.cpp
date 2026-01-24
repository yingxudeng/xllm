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

#include <cstdlib>
#include <memory>

#include "common/global_flags.h"
#include "logits_utils.h"
#include "sampling_params.h"
#if defined(USE_CUDA)
#include "kernels/cuda/air_log_softmax_last_dim.h"
#include "kernels/cuda/air_topk_last_dim.h"
#include "kernels/cuda/cuda_ops_api.h"
#include "kernels/cuda/cuda_utils.h"
#endif

namespace xllm {
namespace {
static inline bool use_air_log_softmax_env() {
  const char* v = std::getenv("XLLM_USE_AIR_LOG_SOFTMAX");
  if (!v) {
    return false;
  }
  if (v[0] == '0' || v[0] == 'f' || v[0] == 'F' || v[0] == 'n' || v[0] == 'N') {
    return false;
  }
  return true;
}

static inline bool use_air_topk_env() {
  const char* v = std::getenv("XLLM_USE_AIR_TOPK");
  if (!v) {
    return false;
  }
  if (v[0] == '0' || v[0] == 'f' || v[0] == 'F' || v[0] == 'n' || v[0] == 'N') {
    return false;
  }
  return true;
}

static inline torch::Tensor log_softmax_last_dim(
    const torch::Tensor& input,
    const torch::Tensor& temperatures) {
  const bool has_temps = temperatures.defined();
#if defined(USE_CUDA)
  if (input.is_cuda()) {
    if (use_air_log_softmax_env()) {
      kernel::cuda::NvtxRange range("softmax.air");
      return kernel::cuda::air_log_softmax_last_dim(input, temperatures);
    }
    kernel::cuda::NvtxRange range("softmax.torch");
    if (!has_temps) {
      return torch::log_softmax(input, /*dim=*/-1, /*dtype=*/torch::kFloat32);
    }
    auto logits = input.to(torch::kFloat32);
    auto temps =
        temperatures.to(torch::kFloat32).to(input.device()).unsqueeze(1);
    temps = torch::where(temps == 0, torch::ones_like(temps), temps);
    logits.div_(temps);
    return torch::log_softmax(logits, /*dim=*/-1);
  }
#endif
  if (!has_temps) {
    return torch::log_softmax(input, /*dim=*/-1, /*dtype=*/torch::kFloat32);
  }
  auto logits = input.to(torch::kFloat32);
  auto temps = temperatures.to(torch::kFloat32).to(input.device()).unsqueeze(1);
  temps = torch::where(temps == 0, torch::ones_like(temps), temps);
  logits.div_(temps);
  return torch::log_softmax(logits, /*dim=*/-1);
}
}  // namespace

SampleOutput Sampler::forward(torch::Tensor& logits,
                              const SamplingParameters& params) const {
  SampleOutput output;
#if defined(USE_CUDA)
  std::unique_ptr<kernel::cuda::NvtxRange> sampler_range;
  if (logits.is_cuda()) {
    sampler_range = std::make_unique<kernel::cuda::NvtxRange>("sampler");
  }
#endif
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

  // Fast path for pure-device multi-round REC beam search.
  if (params.use_beam_search && params.logprobs && FLAGS_enable_fast_sampler &&
      params.max_top_logprobs > 0 && !params.top_p.defined() &&
      !FLAGS_enable_qwen3_reranker && FLAGS_max_decode_rounds > 0) {
    torch::Tensor sample_logits = logits;
    if (params.selected_token_idxes.numel() != params.sample_idxes.numel()) {
      sample_logits = logits.index_select(/*dim=*/0, params.sample_idxes);
    }

    CHECK_EQ(sample_logits.size(0), params.do_sample.size(0));

    torch::Tensor topk_values;
    torch::Tensor topk_indices;
#if defined(USE_CUDA)
    if (use_air_topk_env() && sample_logits.is_cuda()) {
      std::tie(topk_values, topk_indices) =
          xllm::kernel::cuda::air_topk_last_dim(
              sample_logits,
              static_cast<int32_t>(params.max_top_logprobs),
              /*largest=*/true,
              /*sorted_by_value=*/FLAGS_enable_topk_sorted);
    } else {
#endif
      std::tie(topk_values, topk_indices) =
          sample_logits.topk(params.max_top_logprobs,
                             /*dim=*/-1,
                             /*largest=*/true,
                             /*sorted=*/FLAGS_enable_topk_sorted);
#if defined(USE_CUDA)
    }
#endif

    output.top_tokens = (topk_indices.scalar_type() == torch::kLong)
                            ? topk_indices
                            : topk_indices.to(torch::kLong);

    torch::Tensor temperatures;
    if (params.temperatures.defined()) {
      temperatures = params.temperatures;
      if (params.selected_token_idxes.numel() != params.sample_idxes.numel()) {
        temperatures =
            temperatures.index_select(/*dim=*/0, params.sample_idxes);
      }
      temperatures = temperatures.to(torch::kFloat32);
    }

    output.top_logprobs = log_softmax_last_dim(topk_values, temperatures);
    return output;
  }

  // apply temperatures, top-k and top-p
  apply_top_k_top_p(logits, params.temperatures, params.top_k, params.top_p);

  torch::Tensor sample_logits = logits;
  if (params.selected_token_idxes.numel() != params.sample_idxes.numel()) {
    sample_logits = logits.index_select(/*dim=*/0, params.sample_idxes);
  }

  // same batch size
  CHECK_EQ(sample_logits.size(0), params.do_sample.size(0));

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
  output.probs = probs;
  output.next_tokens = samples;

  if (params.logprobs) {
    if (FLAGS_enable_qwen3_reranker) {
      int32_t false_id = 2152;  // "no"
      int32_t true_id = 9693;   // "yes"
      auto indices =
          torch::tensor({false_id, true_id}, torch::kLong).to(samples.device());
      sample_logits = sample_logits.index_select(/*dim=*/1, indices);
      auto logprobs = log_softmax_last_dim(sample_logits, torch::Tensor());
      logprobs = logprobs.index({torch::indexing::Slice(), 1});
      output.logprobs = logprobs.view({-1}).exp();
      return output;
    }
    // log_softmax is equivalent to log(softmax) but more numerically stable
    auto logprobs = log_softmax_last_dim(sample_logits, torch::Tensor());
    // select the logprobs for each sequence
    auto selected_logprobs = logprobs.gather(/*dim=*/-1, samples.view({-1, 1}));
    output.logprobs = selected_logprobs.view({-1});

    if (params.max_top_logprobs > 0) {
#if defined(USE_CUDA)
      auto batch_size = static_cast<uint32_t>(logprobs.size(0));
      auto vocab_size = static_cast<uint32_t>(logprobs.size(1));
      uint32_t k = static_cast<uint32_t>(params.max_top_logprobs);

      torch::Tensor values;
      torch::Tensor indices;
      if (use_air_topk_env() && logprobs.is_cuda()) {
        std::tie(values, indices) =
            kernel::cuda::compute_topk_general(logprobs,
                                               batch_size,
                                               vocab_size,
                                               k,
                                               logprobs.device(),
                                               FLAGS_enable_topk_sorted);
      } else {
        std::tie(values, indices) =
            logprobs.topk(params.max_top_logprobs,
                          /*dim=*/-1,
                          /*largest=*/true,
                          /*sorted=*/FLAGS_enable_topk_sorted);
      }
#else
      auto [values, indices] =
          logprobs.topk(params.max_top_logprobs,
                        /*dim=*/-1,
                        /*largest=*/true,
                        /*sorted=*/FLAGS_enable_topk_sorted);
#endif
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
