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

#include "concurrent_rec_worker_impl.h"

#include <c10/core/DeviceGuard.h>
#include <c10/core/StreamGuard.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <thread>
#include <unordered_map>
#include <utility>

#include "common/device_monitor.h"
#include "common/metrics.h"
#include "common/rec_model_utils.h"
#include "common/types.h"
#include "core/common/global_flags.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/model_loader.h"
#include "framework/state_dict/state_dict.h"
#if defined(USE_CUDA) || defined(USE_ILU)
#include "kernels/cuda/cuda_ops_api.h"
#include "layers/cuda/flashinfer_workspace.h"
#endif
#if defined(USE_NPU)
#include "kernels/npu/npu_ops_api.h"
#endif
#include "models/model_registry.h"
#include "util/threadpool.h"
#include "util/timer.h"

DECLARE_int32(max_batch_size);

namespace xllm {

ConcurrentRecWorkerImpl::ConcurrentRecWorkerImpl(
    const ParallelArgs& parallel_args,
    const torch::Device& device,
    const runtime::Options& options)
    : RecWorkerImpl(parallel_args, device, options),
      max_concurrency_(FLAGS_rec_worker_max_concurrency) {
  CHECK_GT(max_concurrency_, 0)
      << "rec_worker_max_concurrency must be greater than 0";
  // Create independent step_threadpool_ dedicated to parallel execution of
  // step() Use schedule() to assign tasks, letting ThreadPool automatically
  // select idle threads
  step_threadpool_ = std::make_unique<ThreadPool>(
      max_concurrency_, [this]() mutable { device_.set_device(); });

  LOG(INFO) << "ConcurrentRecWorkerImpl: Created step_threadpool_ with "
            << max_concurrency_ << " threads for parallel step execution";
}

bool ConcurrentRecWorkerImpl::init_model(ModelContext& context) {
  CHECK(model_ == nullptr) << "Model is already initialized.";

  // Determine rec model kind and pipeline type
  const auto& model_type = context.get_model_args().model_type();
  rec_model_kind_ = get_rec_model_kind(model_type);
  CHECK(rec_model_kind_ != RecModelKind::kNone)
      << "Unsupported rec model_type: " << model_type;

  // Create concurrent pipeline (not base class pipeline)
  auto pipeline_type = get_rec_pipeline_type(rec_model_kind_);

  // Reserve space for model instances
  multi_stream_pipelines_.reserve(max_concurrency_);
  for (size_t i = 0; i < max_concurrency_; ++i) {
    auto worker_pipeline = create_concurrent_pipeline(pipeline_type, *this);

    auto stream = device_.get_stream_from_pool();
    worker_pipeline->stream_ = std::move(stream);
    auto stream_guard = worker_pipeline->stream_->set_stream_guard();

    worker_pipeline->context_ =
        std::make_unique<ModelContext>(context.get_parallel_args(),
                                       context.get_model_args(),
                                       context.get_quant_args(),
                                       context.get_tensor_options());

    worker_pipeline->model_ =
        create_llm_model(*worker_pipeline->context_.get());
    CHECK(worker_pipeline->model_ != nullptr)
        << "Failed to create model instance " << i;

    worker_pipeline->executor_ =
        std::make_unique<Executor>(worker_pipeline->model_.get(),
                                   worker_pipeline->context_->get_model_args(),
                                   device_,
                                   options_);

#if defined(USE_CUDA)
    worker_pipeline->flashinfer_workspace_.initialize(device_);
#endif

    multi_stream_pipelines_.emplace_back(std::move(worker_pipeline));
    index_queue_.enqueue(i);
  }

  model_.reset(multi_stream_pipelines_[0]->model_.get());
  model_executor_.reset(multi_stream_pipelines_[0]->executor_.get());
  work_pipeline_.reset(
      dynamic_cast<RecWorkPipeline*>(multi_stream_pipelines_[0].get()));

  // Complete other initialization (EPLB, BeamSearcher, etc.)
  if (FLAGS_enable_eplb) {
    eplb_executor_ = std::make_unique<EplbExecutor>(model_.get(), device_);
  }

  if (FLAGS_enable_beam_search_kernel) {
    beam_searcher_ = std::make_unique<BeamSearcher>();
  }

  LOG(INFO) << "Created " << multi_stream_pipelines_.size()
            << " pipelines for concurrent execution";
  return true;
}

void ConcurrentRecWorkerImpl::load_model(std::unique_ptr<ModelLoader> loader) {
  CHECK(!multi_stream_pipelines_.empty())
      << "Model instances are not initialized. Call init_model() first.";

  // Save model weights path to create new loaders for other instances
  std::string model_weights_path = loader->model_weights_path();

  // Load weights for the first model instance (using the original loader)
  multi_stream_pipelines_[0]->model_->load_model(std::move(loader));
  LOG(INFO) << "Loaded weights for model instance 0";

  // Create new loaders and load weights for other model instances
  for (size_t i = 1; i < multi_stream_pipelines_.size(); ++i) {
    auto model_loader = ModelLoader::create(model_weights_path);
    CHECK(model_loader != nullptr)
        << "Failed to create ModelLoader for model instance " << i;
    multi_stream_pipelines_[i]->model_->load_model(std::move(model_loader));
    LOG(INFO) << "Loaded weights for model instance " << i;
  }

  LOG(INFO) << "Loaded weights for all " << multi_stream_pipelines_.size()
            << " models";
}

folly::SemiFuture<std::optional<ForwardOutput>>
ConcurrentRecWorkerImpl::step_async(const ForwardInput& input) {
  folly::Promise<bool> pre_promise;
  auto pre_future = pre_promise.getSemiFuture();

  folly::Promise<std::optional<ForwardOutput>> promise;
  auto future = promise.getSemiFuture();

  if (FLAGS_enable_graph) {
    warmup(input);
  }
  // Use schedule() to assign tasks, letting ThreadPool automatically select
  // idle threads The logic for allocating instance_id happens when the task
  // executes (see lambda below)
  step_threadpool_->schedule([this,
                              &input,
                              pre_promise = std::move(pre_promise),
                              promise = std::move(promise)]() mutable {
    size_t index;
    index_queue_.wait_dequeue(index);

    auto stream_guard =
        multi_stream_pipelines_[index]->stream_->set_stream_guard();

    ForwardInput input_on_device;
    multi_stream_pipelines_[index]->prepare_work_before_execute(
        input, input_on_device);

    pre_promise.setValue(true);

#if defined(USE_CUDA)
    input_on_device.input_params.set_flashinfer_workspace_buffer(
        multi_stream_pipelines_[index]->flashinfer_workspace_);
#endif

    if (hierarchy_kv_cache_transfer_ != nullptr) {
      hierarchy_kv_cache_transfer_->set_layer_synchronizer(
          input_on_device.input_params);
    }
    // Handle enable_schedule_overlap logic (if needed)
    if (!enable_schedule_overlap()) {
      const auto output = multi_stream_pipelines_[index]->step(input_on_device);
      promise.setValue(output);
    } else {
      if (last_step_output_valid_ &&
          !input_on_device.input_params.empty_kv_cache) {
        // replace step i model input with true output of step i-1
        input_on_device = update_input_by_last_step_output(input_on_device);
      }

      const auto output_overlap =
          multi_stream_pipelines_[index]->step(input_on_device);
      if (output_overlap.has_value()) {
        if (is_driver() || FLAGS_enable_eplb) {
          std::unique_lock<std::mutex> lock(mtx_);
          cv_.wait(lock, [this] { return !is_recorded_; });
          update_last_step_output(output_overlap);
          is_recorded_ = true;
          cv_.notify_one();
        } else {
          update_last_step_output(output_overlap);
        }
      } else {
        if (is_driver() || FLAGS_enable_eplb) {
          std::unique_lock<std::mutex> lock(mtx_);
          cv_.wait(lock, [this] { return !is_recorded_; });
          last_step_output_valid_ = false;
          is_recorded_ = true;
          cv_.notify_one();
        } else {
          last_step_output_valid_ = false;
        }
      }
      promise.setValue(output_overlap);
    }

    index_queue_.enqueue(index);
  });

  pre_future.wait();
  return future;
}

std::optional<ForwardOutput> ConcurrentRecWorkerImpl::step(
    const ForwardInput& input) {
  LOG(ERROR) << "ConcurrentRecWorkerImpl::step should not be called.";
  return std::nullopt;
}

void ConcurrentRecWorkerImpl::prepare_work_before_execute(
    const ForwardInput& inputs,
    ForwardInput& processed_inputs) {
  LOG(ERROR) << "ConcurrentRecWorkerImpl::step should not be called.";
}

void ConcurrentRecWorkerImpl::warmup(const ForwardInput& inputs) {
  if (warmup_set_.find(inputs.input_params.num_sequences) !=
      warmup_set_.end()) {
    return;
  }
  std::unique_lock<std::mutex> lock(warmup_mutex_);
  if (warmup_set_.find(inputs.input_params.num_sequences) !=
      warmup_set_.end()) {
    return;
  }
  std::vector<size_t> warmup_indices;

  while (warmup_indices.size() < max_concurrency_) {
    size_t index;
    index_queue_.wait_dequeue(index);
    warmup_indices.push_back(index);
  }
  for (auto index : warmup_indices) {
    multi_stream_pipelines_[index]->warmup(inputs);
    LOG(INFO) << "Warmup finished for index: " << index;
  }

  warmup_set_.insert(inputs.input_params.num_sequences);

  for (auto index : warmup_indices) {
    index_queue_.enqueue(index);
  }
}

// ============================================================
// Concurrent Pipeline Implementation
// ============================================================

void ConcurrentRecWorkerImpl::ConcurrentLlmRecPureDevicePipeline::
    prepare_work_before_execute(const ForwardInput& inputs,
                                ForwardInput& processed_inputs) {
  auto dtype = concurrent_worker_.dtype();
  auto device = concurrent_worker_.device_;
  processed_inputs = inputs.to(device, dtype);

#if defined(USE_NPU) || defined(USE_CUDA)
  prepare_kv_caches_related_for_input(inputs, processed_inputs);
#endif
}

std::optional<ForwardOutput>
ConcurrentRecWorkerImpl::ConcurrentLlmRecPureDevicePipeline::step(
    const ForwardInput& input) {
  Timer timer;
  auto device = concurrent_worker_.device_;

  ForwardInput& mutable_input = const_cast<ForwardInput&>(input);

  int32_t total_rounds = mutable_input.total_round;
  int32_t max_decode_step = total_rounds - 1;
  int32_t batch_size =
      mutable_input.input_params.paged_kv_last_page_len.numel();
  int32_t beam_width = mutable_input.beam_width;
  int32_t layer_num =
      static_cast<int32_t>(context_->get_model_args().n_layers());

  auto int_options = torch::TensorOptions().dtype(torch::kInt32).device(device);
  auto fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto paged_options =
      torch::TensorOptions().dtype(torch::kInt32).device(device);
  CHECK_GT(concurrent_worker_.kv_caches_.size(), 0)
      << "KV caches are not initialized.";
  auto kv_cache_options =
      concurrent_worker_.kv_caches_[0].get_k_cache().options();

  BeamSearchTensors beam_tensors =
      prepare_beam_search_tensors(batch_size, beam_width, total_rounds, device);

  mutable_input.input_params.num_heads = context_->get_model_args().n_heads();
  mutable_input.input_params.head_dim = context_->get_model_args().head_dim();
  mutable_input.input_params.beam_width = beam_width;
  mutable_input.input_params.current_round = current_round_;

  ForwardOutput output;
  torch::Tensor logits;
  SampleOutput sample_output;
  torch::Tensor top_tokens;
  std::optional<folly::SemiFuture<NextRoundInputResults>>
      next_round_async_result;

  for (int32_t round = 0; round < total_rounds; ++round) {
    const auto& sampling_params = round > 0
                                      ? mutable_input.decoder_sampling_params
                                      : mutable_input.sampling_params;
    mutable_input.input_params.is_prefill = round == 0;
    mutable_input.input_params.attn_metadata = nullptr;
    current_round_.fill_(round - 1);

    // Start async computation for next round input (overlap with GPU
    // logits/sampling)
    // TODO: support async computation for next round input
    if (round < total_rounds - 1 && !FLAGS_enable_graph) {
      next_round_async_result =
          compute_next_round_input_async(mutable_input.input_params.kv_seq_lens,
                                         round,
                                         batch_size,
                                         beam_width,
                                         max_decode_step,
                                         paged_options);
    }

    torch::Tensor hidden_states;

    hidden_states = executor_->forward(mutable_input.token_ids,
                                       mutable_input.positions,
                                       concurrent_worker_.kv_caches_,
                                       mutable_input.input_params);

    if (!hidden_states.defined()) {
      return std::nullopt;
    }

    if (sampling_params.selected_token_idxes.defined()) {
      logits =
          model_->logits(hidden_states, sampling_params.selected_token_idxes);
      sample_output =
          concurrent_worker_.sampler_->forward(logits, sampling_params);
      top_tokens = sample_output.top_tokens.to(torch::kInt32)
                       .reshape({-1, mutable_input.beam_width});
    }

    if (sample_output.top_tokens.defined()) {
      torch::Tensor top_logprobs =
          sample_output.top_logprobs.reshape({-1, beam_width});
      execute_beam_search(
          top_tokens, top_logprobs, beam_tensors, round, batch_size);

      beam_tensors.sequence_group.copy_(beam_tensors.out_seqgroup,
                                        /*non_blocking=*/true);
      beam_tensors.acc_logprob.copy_(beam_tensors.out_log_probs,
                                     /*non_blocking=*/true);

      if (round < total_rounds - 1) {
        // Use async results if available, otherwise fallback to sync
        // computation
        if (next_round_async_result.has_value()) {
          update_input_for_next_round(mutable_input,
                                      round,
                                      sample_output,
                                      top_tokens,
                                      beam_tensors,
                                      next_round_async_result.value());
        } else {
          // Fallback to synchronous computation
          update_input_for_next_round(mutable_input,
                                      round,
                                      sample_output,
                                      top_tokens,
                                      beam_tensors,
                                      batch_size,
                                      beam_width,
                                      max_decode_step,
                                      paged_options);
        }

        if (round > 0) {
          execute_cache_select(
              beam_tensors, mutable_input, round, beam_width, layer_num);
        }
      }

      if (round == total_rounds - 1) {
        build_final_output(
            logits, sample_output, sampling_params, beam_tensors, output);
      }
    }
  }

  if (stream_ != nullptr) {
    stream_->synchronize();
  } else {
    device.synchronize_default_stream();
  }
  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
  DeviceMonitor::get_instance().update_active_activation_memory(device.index());
  return output;
}

// Factory method to create concurrent pipeline
std::unique_ptr<ConcurrentRecWorkerImpl::ConcurrentLlmRecPureDevicePipeline>
ConcurrentRecWorkerImpl::create_concurrent_pipeline(
    RecPipelineType type,
    ConcurrentRecWorkerImpl& worker) {
  // Only kLlmRecPureDevicePipeline uses Concurrent pipeline
  // Other pipeline types use base class pipelines
  if (type == RecPipelineType::kLlmRecPureDevicePipeline) {
    return std::make_unique<ConcurrentLlmRecPureDevicePipeline>(worker);
  }
  return nullptr;
}

void ConcurrentRecWorkerImpl::update_last_step_output(
    const std::optional<ForwardOutput>& output) {
  // Implement the same logic as the base class because the base class's
  // method is private
  if (output.has_value()) {
    if (output.value().sample_output.next_tokens.defined()) {
      last_step_output_ = std::move(output.value());
      last_step_output_valid_ = true;
    } else {
      if (FLAGS_enable_eplb) {
        last_step_output_ = std::move(output.value());
      }
      last_step_output_valid_ = false;
    }
  } else {
    last_step_output_valid_ = false;
  }
}

void ConcurrentRecWorkerImpl::ConcurrentLlmRecPureDevicePipeline::warmup(
    const ForwardInput& input) {
  auto stream_guard = stream_->set_stream_guard();

  ForwardInput input_on_device;
  prepare_work_before_execute(input, input_on_device);

#if defined(USE_CUDA)
  input_on_device.input_params.set_flashinfer_workspace_buffer(
      flashinfer_workspace_);
#endif

  auto output = step(input_on_device);
  CHECK(output.has_value()) << "Warmup failed.";

  LOG(INFO) << "Warmup finished for batch size: "
            << input.input_params.num_sequences;
}

}  // namespace xllm
