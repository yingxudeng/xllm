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
#include <utility>

#include "common/device_monitor.h"
#include "common/metrics.h"
#include "common/types.h"
#include "core/common/global_flags.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/state_dict/state_dict.h"
#if defined(USE_CUDA) || defined(USE_ILU) || defined(USE_MUSA)
#include "layers/cuda/flashinfer_workspace.h"
#endif
#include "framework/model/model_args.h"
#include "models/model_registry.h"
#include "util/threadpool.h"
#include "util/timer.h"

namespace xllm {

namespace {

torch::Device get_linear_state_device(std::vector<KVCache>& kv_caches,
                                      const ModelArgs& args) {
  for (int64_t i = 0; i < static_cast<int64_t>(kv_caches.size()); ++i) {
    if (!is_full_attention_layer(args, i)) {
      auto conv = kv_caches[i].get_conv_cache();
      if (conv.defined()) return conv.device();
    }
  }
  return torch::kCPU;
}

void copy_linear_state_slots(std::vector<KVCache>& kv_caches,
                             const ModelArgs& args,
                             const torch::Tensor& src_indices,
                             const torch::Tensor& dst_indices) {
  const int64_t num_layers = static_cast<int64_t>(kv_caches.size());
  for (int64_t layer_id = 0; layer_id < num_layers; ++layer_id) {
    if (is_full_attention_layer(args, layer_id)) {
      continue;
    }
    auto conv_cache = kv_caches[layer_id].get_conv_cache();
    auto ssm_cache = kv_caches[layer_id].get_ssm_cache();
    if (!conv_cache.defined() || !ssm_cache.defined()) {
      continue;
    }
    auto src_conv = conv_cache.index_select(0, src_indices);
    conv_cache.index_copy_(0, dst_indices, src_conv);
    auto src_ssm = ssm_cache.index_select(0, src_indices);
    ssm_cache.index_copy_(0, dst_indices, src_ssm);
  }
}

}  // namespace

LLMWorkerImpl::LLMWorkerImpl(const ParallelArgs& parallel_args,
                             const torch::Device& device,
                             const runtime::Options& options)
    : WorkerImpl(parallel_args, device, options) {
  device_.set_device();
#if defined(USE_CUDA) || defined(USE_MUSA)
  threadpool_.schedule([this]() mutable {
    // initialize flashinfer workspace
    ::xllm::layer::flashinfer::FlashinferWorkspace::get_instance().initialize(
        device_);
  });
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
  if (FLAGS_enable_manual_loader) {
#if defined(USE_NPU)
    if (!enable_schedule_overlap() && options_.backend() == "llm") {
      aclrtStream current_stream =
          c10_npu::getCurrentNPUStream(device_.index()).stream();
      atb::Context* atb_context =
          const_cast<atb::Context*>(context_.get_atb_context());
      atb_context->SetExecuteStream(current_stream);
    } else {
      SET_ATB_EXECUTE_STREAM(compute_stream_, device_, context_);
    }
#endif
    return step_internal(input);
  }
  return step_internal(input);
}

std::optional<ForwardOutput> LLMWorkerImpl::step_internal(
    const ForwardInput& input) {
  MULTI_MODEL_STEP_LOCK(FLAGS_enable_xtensor);

  Timer timer;
  auto& sampling_params = input.sampling_params;

  std::vector<folly::SemiFuture<bool>> futures;

  if (options_.kv_cache_transfer_mode() == "PUSH" &&
      !input.transfer_kv_infos.empty()) {
#if defined(USE_NPU)
    std::shared_ptr<NPULayerSynchronizerImpl> layer_synchronizer =
        std::make_shared<NPULayerSynchronizerImpl>(
            context_.get_model_args().n_layers());
#elif defined(USE_MLU)
    std::shared_ptr<MLULayerSynchronizerImpl> layer_synchronizer =
        std::make_shared<MLULayerSynchronizerImpl>(
            context_.get_model_args().n_layers());
#endif
#if defined(USE_NPU) || defined(USE_MLU)
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

  // Restore linear state from checkpoint slots before forward.
  const auto& restore_ids = input.input_params.restore_checkpoint_slot_ids;
  const auto& save_ids = input.input_params.save_checkpoint_slot_ids;
  const auto& live_ids = input.input_params.linear_state_ids;
  const auto& args = context_.get_model_args();
  if (!restore_ids.empty()) {
    std::vector<int64_t> src_vec, dst_vec;
    for (size_t i = 0; i < restore_ids.size(); ++i) {
      if (restore_ids[i] >= 0 && i < live_ids.size() && live_ids[i] >= 0) {
        src_vec.push_back(restore_ids[i]);
        dst_vec.push_back(live_ids[i]);
      }
    }
    if (!src_vec.empty()) {
      auto device = get_linear_state_device(kv_caches_, args);
      copy_linear_state_slots(
          kv_caches_, args,
          torch::tensor(src_vec, torch::kLong).to(device),
          torch::tensor(dst_vec, torch::kLong).to(device));
    }
  }

  // call model executor forward to get hidden states
  auto model_output = model_executor_->forward(
      input.token_ids, input.positions, kv_caches_, input.input_params);
  if (!model_output.hidden_states.defined()) {
    return std::nullopt;
  }

  // Save linear state to checkpoint slots after forward.
  if (!save_ids.empty()) {
    std::vector<int64_t> src_vec, dst_vec;
    for (size_t i = 0; i < save_ids.size(); ++i) {
      if (save_ids[i] >= 0 && i < live_ids.size() && live_ids[i] >= 0) {
        src_vec.push_back(live_ids[i]);
        dst_vec.push_back(save_ids[i]);
      }
    }
    if (!src_vec.empty()) {
      auto device = get_linear_state_device(kv_caches_, args);
      copy_linear_state_slots(
          kv_caches_, args,
          torch::tensor(src_vec, torch::kLong).to(device),
          torch::tensor(dst_vec, torch::kLong).to(device));
    }
  }

  torch::Tensor logits;
  if (sampling_params.selected_token_idxes.defined()) {
    logits = model_->logits(model_output.hidden_states,
                            sampling_params.selected_token_idxes);
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
    MULTI_MODEL_STEP_UNLOCK();
    auto ret = device_.synchronize_default_stream();
    // in p-d disaggregation scene, all micro batches should be in same
    // prefill/decode stage, so, to judge transfer_kv_infos.empty,
    if (options_.kv_cache_transfer_mode() == "PUSH" &&
        !input.transfer_kv_infos.empty()) {
      auto results =
          folly::collectAll(futures).within(std::chrono::seconds(60)).get();
      for (const auto& result : results) {
        // TODO: Add error handling
        if (!result.value()) {
          LOG(ERROR) << "kv_cache_transfer_ failed";
          break;
        }
      }
    }
    if (FLAGS_enable_eplb) {
      return output;
    }
    return std::nullopt;
  }

  // driver prepare model output
  if (sampling_params.selected_token_idxes.defined()) {
    output.logits = logits;
    output.do_sample = sampling_params.do_sample;
    output.logprobs = sampling_params.logprobs;
    output.max_top_logprobs = sampling_params.max_top_logprobs;
    if (!input.skip_sampling_for_logits_only) {
      auto sample_output = sampler_->forward(logits, sampling_params);

      // beam search kernel
      BeamSearchOutput beam_search_output;
      if (sampling_params.use_beam_search && input.acc_logprob.defined() &&
          input.acc_logprob.numel() > 0) {
        beam_search_output =
            beam_searcher_->forward(input.acc_logprob,
                                    sample_output.top_tokens,
                                    sample_output.top_logprobs);
      }

      // set sample output to output
      output.sample_output = sample_output;
      // set beam search output to output
      output.beam_search_output = beam_search_output;
    }
  }

  if (options_.enable_speculative_decode()) {
    torch::Tensor embeddings;
    if (model_output.aux_hidden_states.defined()) {
      embeddings = model_output.aux_hidden_states;
    } else {
      embeddings = model_output.hidden_states;
    }
    if (!input.input_params.batch_forward_type.is_decode() && !is_spec_draft_) {
      output.sample_output.embeddings = embeddings;
    } else if (sampling_params.selected_token_idxes.defined()) {
      output.sample_output.embeddings = embeddings.index_select(
          /*dim=*/0, sampling_params.selected_token_idxes);
    }
  }

  MULTI_MODEL_STEP_UNLOCK();
  auto ret = device_.synchronize_default_stream();

  if (options_.kv_cache_transfer_mode() == "PUSH" &&
      !input.transfer_kv_infos.empty()) {
    auto results =
        folly::collectAll(futures).within(std::chrono::seconds(60)).get();
    for (const auto& result : results) {
      // TODO: Add error handling
      if (!result.value()) {
        LOG(ERROR) << "kv_cache_transfer_ failed";
        break;
      }
    }
  }

  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
  DeviceMonitor::get_instance().update_active_activation_memory(
      device_.index());

  return output;
}

}  // namespace xllm
