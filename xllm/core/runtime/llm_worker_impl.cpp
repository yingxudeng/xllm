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

#include "llm_worker_impl.h"

#include <c10/core/DeviceGuard.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>
#include <optional>
#include <sstream>
#include <utility>

#include "common/device_monitor.h"
#include "common/metrics.h"
#include "common/types.h"
#include "core/common/global_flags.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/state_dict/state_dict.h"
#if defined(USE_CUDA) || defined(USE_ILU)
#include "layers/cuda/flashinfer_workspace.h"
#endif
#include "models/model_registry.h"
#include "util/threadpool.h"
#include "util/timer.h"
#include "util/utils.h"
#if defined(USE_NPU)
#include <tuple>

#include "kernels/npu/xllm_ops/beam_search_group.h"
#include "kernels/npu/xllm_ops/cache_select.h"
#endif

namespace xllm {

LLMWorkerImpl::LLMWorkerImpl(const ParallelArgs& parallel_args,
                             const torch::Device& device,
                             const runtime::Options& options)
    : WorkerImpl(parallel_args, device, options) {
  device_.set_device();
#if defined(USE_CUDA)
  // initialize flashinfer workspace
  layer::FlashinferWorkspace::get_instance().initialize(device_);
#endif
}

bool LLMWorkerImpl::init_model(ModelContext& context) {
  CHECK(model_ == nullptr) << "Model is already initialized.";

  // Try to create a causal LM model
  model_ = create_llm_model(context);

  // Dont find model in causal models
  CHECK(model_ != nullptr) << "Failed to create model.";
  model_executor_ = std::make_unique<Executor>(
      model_.get(), context.get_model_args(), device_, options_);

  if (FLAGS_enable_eplb) {
    eplb_executor_ = std::make_unique<EplbExecutor>(model_.get(), device_);
  }

  if (FLAGS_enable_beam_search_kernel) {
    beam_searcher_ = std::make_unique<BeamSearcher>();
  }
  return true;
}

std::optional<ForwardOutput> LLMWorkerImpl::step(const ForwardInput& input) {
  LOG(INFO) << "input.input_params.batch_forward_type: " << input.input_params.batch_forward_type.to_string();
  Timer timer;
  // Only enter multi-round decode when explicitly enabled via global flag.
  if (FLAGS_max_decode_rounds > 0 && input.total_round > 0) {
    return step_multi_round(input);
  }
  auto& sampling_params = input.sampling_params;

  std::vector<folly::SemiFuture<bool>> futures;

  if (options_.kv_cache_transfer_mode() == "PUSH" &&
      !input.transfer_kv_infos.empty()) {
#if defined(USE_NPU)
    std::shared_ptr<NPULayerSynchronizerImpl> layer_synchronizer =
        std::make_shared<NPULayerSynchronizerImpl>(
            context_.get_model_args().n_layers());
    const_cast<ModelInputParams*>(&(input.input_params))->layer_synchronizer =
        layer_synchronizer;

    futures.emplace_back(
        kv_cache_transfer_->push_kv_blocks_async(input.transfer_kv_infos,
                                                 context_.get_parallel_args(),
                                                 layer_synchronizer,
                                                 is_spec_draft_));
#endif
  }

  if (FLAGS_enable_eplb) {
    eplb_executor_->eplb_execute(input.eplb_info);
  }

  // temporarily use [0], will be adapted in next pr
  // call model executor forward to get hidden states
  auto hidden_states = model_executor_->forward(
      input.token_ids, input.positions, kv_caches_, input.input_params);
  if (!hidden_states.defined()) {
    return std::nullopt;
  }

  torch::Tensor logits;
  if (sampling_params.selected_token_idxes.defined()) {
    logits =
        model_->logits(hidden_states, sampling_params.selected_token_idxes);
  }

  ForwardOutput output;
  if (FLAGS_enable_eplb) {
    output.expert_load_data = expert_load_data_;
    output.prepared_layer_id = eplb_executor_->get_ready_layer_id();
    if (output.prepared_layer_id != -1) {
      eplb_executor_->reset_ready_layer_id();
    }
  }

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_ &&
      !options_.enable_speculative_decode()) {
    auto ret = device_.synchronize_default_stream();
    // in p-d disaggregation scene, all micro batches should be in same
    // prefill/decode stage, so, to judge transfer_kv_infos.empty,
    if (options_.kv_cache_transfer_mode() == "PUSH" &&
        !input.transfer_kv_infos.empty()) {
      auto results =
          folly::collectAll(futures).within(std::chrono::seconds(60)).get();
      for (const auto& result : results) {
        if (!result.value()) {
          LOG(ERROR) << "kv_cache_transfer_ failed";
          return std::nullopt;
        }
      }
    }
    if (FLAGS_enable_eplb) {
      return output;
    }
    return std::nullopt;
  }

  // driver prepare model output
  SampleOutput sample_output;
  if (sampling_params.selected_token_idxes.defined()) {
    sample_output = sampler_->forward(logits, sampling_params);
    output.logits = logits;

    // beam search kernel
    BeamSearchOutput beam_search_output;
    if (sampling_params.use_beam_search && input.acc_logprob.defined() &&
        input.acc_logprob.numel() > 0) {
      beam_search_output = beam_searcher_->forward(input.acc_logprob,
                                                   sample_output.top_tokens,
                                                   sample_output.top_logprobs);
    }

    // set sample output to output
    output.sample_output = sample_output;
    // carry over the sampling params
    output.do_sample = sampling_params.do_sample;
    output.logprobs = sampling_params.logprobs;
    output.max_top_logprobs = sampling_params.max_top_logprobs;
    // set beam search output to output
    output.beam_search_output = beam_search_output;
  }

  if (options_.enable_speculative_decode()) {
    if (!input.input_params.batch_forward_type.is_decode() && !is_spec_draft_) {
      output.sample_output.embeddings = hidden_states;
    } else if (sampling_params.selected_token_idxes.defined()) {
      auto embeddings = hidden_states.index_select(
          /*dim=*/0, sampling_params.selected_token_idxes);
      output.sample_output.embeddings = embeddings;
    }
  }

  auto ret = device_.synchronize_default_stream();

  if (options_.kv_cache_transfer_mode() == "PUSH" &&
      !input.transfer_kv_infos.empty()) {
    auto results =
        folly::collectAll(futures).within(std::chrono::seconds(60)).get();
    for (const auto& result : results) {
      if (!result.value()) {
        LOG(ERROR) << "kv_cache_transfer_ failed";
        return std::nullopt;
      }
    }
  }

  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
  DeviceMonitor::get_instance().update_active_activation_memory(
      device_.index());

  return output;
}

std::optional<ForwardOutput> LLMWorkerImpl::step_multi_round(
    ForwardInput input) {
  device_.set_device();
  Timer timer;

  int32_t total_rounds = input.total_round;
  
  std::vector<torch::Tensor> unshared_k_cache;
  std::vector<torch::Tensor> unshared_v_cache;
  auto args = context_.get_model_args();
  int32_t layer_num = static_cast<int32_t>(args.n_layers());
  for (auto i = 0; i < layer_num; ++i) {
    unshared_k_cache.push_back(kv_caches_[i].get_k_cache());
    unshared_v_cache.push_back(kv_caches_[i].get_v_cache());
  }
  int32_t batch = input.input_params.num_sequences;

  int32_t beam_width_init = input.beam_width;
  input.input_params.beam_width = beam_width_init;
  auto int_options =
      torch::TensorOptions().dtype(torch::kInt32).device(device_);
  auto fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device_);
  torch::Tensor sequence_group =
      torch::zeros({batch, beam_width_init, total_rounds}, int_options);

  // preallocate outputs and cached inputs
  int64_t num_seq = batch * beam_width_init;
  torch::Tensor acc_logprob = torch::empty({num_seq, 1}, fp32_options);
  torch::Tensor out_log_probs = torch::empty({num_seq, 1}, fp32_options);
  torch::Tensor out_token_ids = torch::empty({num_seq, 1}, int_options);
  torch::Tensor out_token_index = torch::empty({num_seq, 1}, int_options);
  torch::Tensor out_beam_count_prefix_sums =
      torch::empty({num_seq, 1}, int_options);
  auto out_seqgroup = sequence_group.clone();

  ForwardOutput output;

  for (int32_t round = 0; round < total_rounds; ++round) {

    const auto& sampling_params =
        round > 0 ? input.decoder_sampling_params : input.sampling_params;
    input.input_params.is_prefill = round == 0;

    if (!input.input_params.current_round_tensor_list.empty() && round >= 0 &&
        round < static_cast<int32_t>(
                    input.input_params.current_round_tensor_list.size())) {
      input.input_params.current_round_tensor =
          input.input_params.current_round_tensor_list[round];
      input.input_params.current_round = round - 1;
    }

    auto hidden_states = model_executor_->forward(
        input.token_ids, input.positions, kv_caches_, input.input_params);
    if (!hidden_states.defined()) {
      return std::nullopt;
    }

    torch::Tensor logits;
    if (sampling_params.selected_token_idxes.defined()) {
      logits =
          model_->logits(hidden_states, sampling_params.selected_token_idxes);
    }

    if (sampling_params.selected_token_idxes.defined()) {
      auto sample_output = sampler_->forward(logits, sampling_params);
      torch::Tensor top_tokens;
      torch::Tensor top_logprobs;
      int32_t beam_width = beam_width_init;

      if (round == 0) {
        top_tokens =
            sample_output.top_tokens.to(torch::kInt32).reshape({-1, 1});
        LOG(INFO) << "top_tokens.shape: " << top_tokens.sizes();
        top_logprobs = sample_output.top_logprobs.reshape({-1, 1});
        
        // 强制改为decode模式
        input.input_params.batch_forward_type = BatchForwardType(2);

        input.token_ids =
            sample_output.top_tokens.to(torch::kInt32).reshape({-1});
        LOG(INFO) << "input.input_params.decode_positions_tensor_list.empty(): " 
                  << input.input_params.decode_positions_tensor_list.empty();
        if (!input.input_params.decode_positions_tensor_list.empty() &&
            round >= 0 &&
            round < static_cast<int32_t>(
                        input.input_params.decode_positions_tensor_list.size())) {
          input.positions =
              input.input_params.decode_positions_tensor_list[round];
        }
        LOG(INFO) << "input.positions.shape: " << input.positions.sizes();

        // 更新q_seq_len，原来保留的是prefill的seq_len，需要改成decode的
        LOG(INFO) << "num_seq: " << num_seq;
        input.input_params.q_seq_lens = torch::arange(batch + 1, int_options) * beam_width;  
        
      } else {
        top_tokens = sample_output.top_tokens.to(torch::kInt32)
                         .reshape({-1, beam_width});
        top_logprobs = sample_output.top_logprobs.reshape({-1, beam_width});
      }
      #if defined(USE_NPU)
      xllm_ops::beam_search(acc_logprob,
                            top_tokens,
                            top_logprobs,
                            sequence_group,
                            round,
                            out_token_ids,
                            out_token_index,
                            out_log_probs,
                            out_beam_count_prefix_sums,
                            out_seqgroup);
      sequence_group.copy_(out_seqgroup);
      acc_logprob.copy_(out_log_probs);
      // keep group offset contiguous across rounds (already in out_* tensors)
      // update next round tokens.
      if (round == 0) {
        input.token_ids =
            sample_output.top_tokens.to(torch::kInt32).reshape({-1});
      } else {
        input.token_ids = out_token_ids.clone().reshape({-1});
      }

      // update next round positions.
      if (!input.input_params.decode_positions_tensor_list.empty() &&
          round >= 0 &&
          round < static_cast<int32_t>(
                      input.input_params.decode_positions_tensor_list.size())) {
        input.positions =
            input.input_params.decode_positions_tensor_list[round];
      }

      // update output at the last round.
      if (round == total_rounds - 1) {
        output.logits = logits;
        output.sample_output = sample_output;
        output.do_sample = sampling_params.do_sample;
        output.logprobs = sampling_params.logprobs;
        output.max_top_logprobs = sampling_params.max_top_logprobs;
        output.beam_search_output.src_seq_idxes = out_token_index.reshape({-1});
        output.beam_search_output.out_tokens = out_token_ids.reshape({-1});
        output.beam_search_output.out_logprobs = out_log_probs.reshape({-1});
        output.beam_search_output.group_offset =
            out_beam_count_prefix_sums.reshape({-1});
        output.beam_sequence_group = sequence_group;
      }
      #endif

#if defined(USE_NPU)
      if (beam_width > 1 && round > 0) {
        xllm_ops::cache_select(out_token_index,
                               unshared_k_cache,
                               unshared_v_cache,
                               input.input_params.block_tables,
                               out_beam_count_prefix_sums,
                               round,
                               beam_width,
                               layer_num);
      }
#endif
    }
  }

  auto ret = device_.synchronize_default_stream();
  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
  DeviceMonitor::get_instance().update_active_activation_memory(
      device_.index());
  return output;
}

}  // namespace xllm
