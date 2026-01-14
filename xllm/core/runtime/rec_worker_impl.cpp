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

#include "rec_worker_impl.h"

#include <glog/logging.h>

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <vector>

#include "common/device_monitor.h"
#include "common/metrics.h"
#include "common/types.h"
#include "framework/model/model_input_params.h"
#if defined(USE_CUDA)
#include "kernels/cuda/cuda_ops_api.h"
#endif
#include "framework/model_loader.h"
#include "models/model_registry.h"
#include "util/env_var.h"
#include "util/timer.h"

namespace xllm {

RecWorkerImpl::LlmRecWorkPipeline::LlmRecWorkPipeline(RecWorkerImpl& worker)
    : worker_(worker) {}

bool RecWorkerImpl::LlmRecWorkPipeline::create_model(RecWorkerImpl& worker,
                                                     ModelContext& context) {
  return worker.LLMWorkerImpl::init_model(context);
}

ForwardInput RecWorkerImpl::LlmRecWorkPipeline::prepare_inputs(Batch& batch) {
  return worker_.WorkerImpl::prepare_inputs(batch);
}

void RecWorkerImpl::LlmRecWorkPipeline::prepare_work_before_execute(
    const ForwardInput& inputs,
    ForwardInput& processed_inputs) {
  worker_.WorkerImpl::prepare_work_before_execute(inputs, processed_inputs);
  // LlmRecDefault (pure qwen3) does not process mm_data.
  // For mm_data processing, use LlmRecWithMmDataWorkPipeline.
}

std::optional<ForwardOutput> RecWorkerImpl::LlmRecWorkPipeline::step(
    const ForwardInput& input) {
  return worker_.LLMWorkerImpl::step(input);
}

RecWorkerImpl::OneRecWorkPipeline::OneRecWorkPipeline(RecWorkerImpl& worker)
    : worker_(worker) {}

bool RecWorkerImpl::OneRecWorkPipeline::create_model(RecWorkerImpl& worker,
                                                     ModelContext& context) {
  // OneRec also uses LLM model for now, can be extended to create_rec_model
  // later
  return worker.LLMWorkerImpl::init_model(context);
}

ForwardInput RecWorkerImpl::OneRecWorkPipeline::prepare_inputs(Batch& batch) {
  ThreadPool* thread_pool = worker_.input_builder_thread_pool_
                                ? worker_.input_builder_thread_pool_.get()
                                : nullptr;

  return batch.prepare_rec_forward_input(worker_.options_.num_decoding_tokens(),
                                         /*min_decoding_batch_size=*/0,
                                         worker_.context_.get_model_args(),
                                         thread_pool);
}

void RecWorkerImpl::OneRecWorkPipeline::prepare_work_before_execute(
    const ForwardInput& inputs,
    ForwardInput& processed_inputs) {
  worker_.WorkerImpl::prepare_work_before_execute(inputs, processed_inputs);
}

std::optional<ForwardOutput> RecWorkerImpl::OneRecWorkPipeline::step(
    const ForwardInput& input) {
  Timer timer;
  worker_.device_.set_device();

  const auto& sampling_params = input.sampling_params;
  const auto& input_params = input.input_params;

  const auto* onerec_params = input_params.onerec_params();
  CHECK(onerec_params != nullptr) << "OneRec requires rec_params.";

  const OneRecModelInputParams& rec_params = *onerec_params;

  torch::Tensor hidden_states;
  if (rec_params.rec_stage == OneRecModelInputParams::RecStage::PREFILL) {
    if (!rec_params.is_first_prefill) {
      ModelInputParams decoder_params = input_params;
      decoder_params.mutable_onerec_params().is_encoder_forward = false;
      hidden_states = worker_.model_executor_->forward(
          input.token_ids, input.positions, worker_.kv_caches_, decoder_params);
    } else {
      const bool has_sparse_embedding =
          rec_params.encoder_sparse_embedding.defined();
      const bool has_encoder_tokens = rec_params.encoder_token_ids.defined() &&
                                      rec_params.encoder_positions.defined();

      if (!has_sparse_embedding && !has_encoder_tokens) {
        LOG(ERROR) << "OneRec first prefill requires encoder inputs.";
        return std::nullopt;
      }

      ModelInputParams encoder_params = input_params;
      auto& mutable_onerec_params = encoder_params.mutable_onerec_params();
      mutable_onerec_params.is_encoder_forward = true;

      torch::Tensor encoder_tokens;
      if (has_sparse_embedding) {
        mutable_onerec_params.is_hybrid_mode = true;
        encoder_tokens = rec_params.encoder_sparse_embedding;
      } else {
        encoder_tokens = rec_params.encoder_token_ids;
      }

      worker_.model_executor_->forward(encoder_tokens,
                                       rec_params.encoder_positions,
                                       worker_.kv_caches_,
                                       encoder_params);

      ModelInputParams decoder_params = input_params;
      decoder_params.mutable_onerec_params().is_encoder_forward = false;
      hidden_states = worker_.model_executor_->forward(
          input.token_ids, input.positions, worker_.kv_caches_, decoder_params);
    }
  } else {
    ModelInputParams decoder_params = input_params;
    decoder_params.mutable_onerec_params().is_encoder_forward = false;
    hidden_states = worker_.model_executor_->forward(
        input.token_ids, input.positions, worker_.kv_caches_, decoder_params);
  }

  if (!hidden_states.defined()) {
    return std::nullopt;
  }

  if (!worker_.enable_schedule_overlap() && !worker_.driver_ &&
      !worker_.dp_driver_ && !worker_.options_.enable_speculative_decode()) {
    worker_.device_.synchronize_default_stream();
    COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
    DeviceMonitor::get_instance().update_active_activation_memory(
        worker_.device_.index());
    return std::nullopt;
  }

  torch::Tensor logits;
  if (sampling_params.selected_token_idxes.defined()) {
    logits = worker_.model_->logits(hidden_states,
                                    sampling_params.selected_token_idxes);
  }

  ForwardOutput output;

  if (sampling_params.selected_token_idxes.defined()) {
    auto sample_output = worker_.sampler_->forward(logits, sampling_params);
    output.logits = logits;
    output.sample_output = sample_output;
    output.do_sample = sampling_params.do_sample;
    output.logprobs = sampling_params.logprobs;
    output.max_top_logprobs = sampling_params.max_top_logprobs;
  }

  worker_.device_.synchronize_default_stream();
  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
  DeviceMonitor::get_instance().update_active_activation_memory(
      worker_.device_.index());

  return output;
}

// ============================================================
// LlmRecWithMmDataWorkPipeline Implementation (qwen3 with embedding)
// ============================================================

RecWorkerImpl::LlmRecWithMmDataWorkPipeline::LlmRecWithMmDataWorkPipeline(
    RecWorkerImpl& worker)
    : worker_(worker) {}

bool RecWorkerImpl::LlmRecWithMmDataWorkPipeline::create_model(
    RecWorkerImpl& worker,
    ModelContext& context) {
  return worker.LLMWorkerImpl::init_model(context);
}

ForwardInput RecWorkerImpl::LlmRecWithMmDataWorkPipeline::prepare_inputs(
    Batch& batch) {
  return worker_.WorkerImpl::prepare_inputs(batch);
}

void RecWorkerImpl::LlmRecWithMmDataWorkPipeline::prepare_work_before_execute(
    const ForwardInput& inputs,
    ForwardInput& processed_inputs) {
  worker_.WorkerImpl::prepare_work_before_execute(inputs, processed_inputs);

  if (!inputs.input_params.mm_data.valid()) {
    return;
  }

  torch::Tensor input_embedding;
  torch::Tensor input_tokens_tensor;
  torch::Tensor input_indices_tensor;

  const auto& mm_data = inputs.input_params.mm_data;
  const auto& processed_mm_data = processed_inputs.input_params.mm_data;

  if (auto res = processed_mm_data.get<torch::Tensor>(LLM_REC_INPUT_TOKENS)) {
    input_tokens_tensor = res.value();
  }

  // Input indices are generated on host side.
  if (auto res = mm_data.get<torch::Tensor>(LLM_REC_INPUT_INDICES)) {
    input_indices_tensor = res.value();
  }

  if (auto res =
          processed_mm_data.get<torch::Tensor>(LLM_REC_INPUT_EMBEDDING)) {
    input_embedding = res.value();
  }

  if (input_embedding.defined()) {
    input_embedding = input_embedding.to(worker_.dtype());
  }

  if (input_indices_tensor.defined()) {
    CHECK(input_tokens_tensor.defined())
        << "LLM_REC_INPUT_TOKENS is required when LLM_REC_INPUT_INDICES is "
           "set.";

#if defined(USE_NPU)
    layer::NpuWordEmbedding npu_word_embedding =
        worker_.get_npu_word_embedding();
    torch::Tensor input_tokens_embedding =
        npu_word_embedding(input_tokens_tensor, 0);
#else
    layer::WordEmbedding word_embedding = worker_.get_word_embedding();
    torch::Tensor input_tokens_embedding =
        word_embedding->forward(input_tokens_tensor);
#endif

    if (input_embedding.defined()) {
      torch::Tensor input_indices_cpu =
          input_indices_tensor.to(torch::kCPU).to(torch::kInt64).contiguous();
      const auto* input_indices_ptr = input_indices_cpu.data_ptr<int64_t>();
      std::vector<int64_t> input_indices(
          input_indices_ptr, input_indices_ptr + input_indices_cpu.numel());

      processed_inputs.input_params.input_embedding =
          worker_.merge_embeddings_by_indices(
              input_tokens_embedding, input_embedding, input_indices);
    } else {
      processed_inputs.input_params.input_embedding = input_tokens_embedding;
    }
  } else if (input_embedding.defined()) {
    processed_inputs.input_params.input_embedding = input_embedding;
  }
}

std::optional<ForwardOutput> RecWorkerImpl::LlmRecWithMmDataWorkPipeline::step(
    const ForwardInput& input) {
  return worker_.LLMWorkerImpl::step(input);
}

RecWorkerImpl::LlmRecPureDevicePipeline::LlmRecPureDevicePipeline(
    RecWorkerImpl& worker)
    : worker_(worker) {}

bool RecWorkerImpl::LlmRecPureDevicePipeline::create_model(
    RecWorkerImpl& worker,
    ModelContext& context) {
  return worker.LLMWorkerImpl::init_model(context);
}

ForwardInput RecWorkerImpl::LlmRecPureDevicePipeline::prepare_inputs(
    Batch& batch) {
  ThreadPool* thread_pool = worker_.input_builder_thread_pool_
                                ? worker_.input_builder_thread_pool_.get()
                                : nullptr;

  return batch.prepare_rec_forward_input(worker_.options_.num_decoding_tokens(),
                                         /*min_decoding_batch_size=*/0,
                                         worker_.context_.get_model_args(),
                                         thread_pool);
}

void RecWorkerImpl::LlmRecPureDevicePipeline::prepare_work_before_execute(
    const ForwardInput& inputs,
    ForwardInput& processed_inputs) {
  auto dtype = worker_.dtype();
  auto device = worker_.device();
  worker_.WorkerImpl::prepare_work_before_execute(inputs, processed_inputs);

#if defined(USE_NPU) || defined(USE_CUDA)
  // step-level decode full cache: allocate/attach by step_uid metadata
  if (is_pure_device_mode()) {
    auto& mip = processed_inputs.input_params;
    int32_t batch_size =
        processed_inputs.input_params.paged_kv_last_page_len.numel();
    int32_t beam_width = processed_inputs.beam_width;
    int32_t current_round = processed_inputs.current_round;
    int32_t total_round = processed_inputs.total_round;
    const auto& shape = processed_inputs.full_kv_shape;
    auto int_options =
        torch::TensorOptions().dtype(torch::kInt32).device(device);
    CHECK(shape.size() == 3) << "the dims offull_kv_shape should be three.";
    // prepare full kv caches and unshared kv caches
    int32_t full_kv_len = shape[0];
    int64_t num_kv_heads = shape[1];
    int64_t head_dim = shape[2];
    auto kv_cache_options = torch::TensorOptions().dtype(dtype).device(device);
    int32_t num_layers = worker_.context_.get_model_args().n_layers();
    // shared_kv_len is the length of shared KV cache (prefill part)
    int32_t shared_kv_len = FLAGS_max_token_per_req;
    // unshared_offset is where unshared KV cache starts (after shared part)
    int32_t unshared_offset = shared_kv_len * batch_size;
    // full_kv_len should include both shared and unshared parts
    // unshared part: batch_size * beam_width * max_decode_step
    int32_t max_decode_step = total_round - 1;

    if (!cached_full_k_caches_.empty() && cached_full_k_caches_[0].defined()) {
      mip.full_k_caches = cached_full_k_caches_;
      mip.full_v_caches = cached_full_v_caches_;
      mip.unshared_k_caches = cached_unshared_k_caches_;
      mip.unshared_v_caches = cached_unshared_v_caches_;
      mip.naive_block_table = cached_naive_block_table_;
    } else {
      mip.full_k_caches.clear();
      mip.full_v_caches.clear();
      mip.full_k_caches.reserve(num_layers);
      mip.full_v_caches.reserve(num_layers);
      mip.unshared_k_caches.clear();
      mip.unshared_v_caches.clear();
      mip.unshared_k_caches.reserve(num_layers);
      mip.unshared_v_caches.reserve(num_layers);
      for (int32_t layer_id = 0; layer_id < num_layers; ++layer_id) {
        auto target_layer_full_k_cache = torch::zeros(
            {full_kv_len, num_kv_heads, head_dim}, kv_cache_options);
        auto target_layer_full_v_cache = torch::zeros(
            {full_kv_len, num_kv_heads, head_dim}, kv_cache_options);

        auto target_layer_unshared_k_cache =
            target_layer_full_k_cache.slice(0, unshared_offset, full_kv_len);
        auto target_layer_unshared_v_cache =
            target_layer_full_v_cache.slice(0, unshared_offset, full_kv_len);

        int64_t expected_view_size =
            batch_size * beam_width * max_decode_step * num_kv_heads * head_dim;

        target_layer_unshared_k_cache = target_layer_unshared_k_cache.view(
            {static_cast<int64_t>(batch_size),
             static_cast<int64_t>(beam_width),
             static_cast<int64_t>(max_decode_step),
             num_kv_heads,
             head_dim});
        target_layer_unshared_v_cache = target_layer_unshared_v_cache.view(
            {static_cast<int64_t>(batch_size),
             static_cast<int64_t>(beam_width),
             static_cast<int64_t>(max_decode_step),
             num_kv_heads,
             head_dim});
        mip.full_k_caches.emplace_back(target_layer_full_k_cache);
        mip.full_v_caches.emplace_back(target_layer_full_v_cache);
        mip.unshared_k_caches.emplace_back(target_layer_unshared_k_cache);
        mip.unshared_v_caches.emplace_back(target_layer_unshared_v_cache);
      }
      cached_full_k_caches_ = mip.full_k_caches;
      cached_full_v_caches_ = mip.full_v_caches;
      cached_unshared_k_caches_ = mip.unshared_k_caches;
      cached_unshared_v_caches_ = mip.unshared_v_caches;
    }
    {
      const auto& dec_pos = processed_inputs.decode_positions_vec;
      mip.decode_positions_tensor_list.clear();
      if (!dec_pos.empty() && beam_width > 0 && total_round > 1) {
        const int32_t n = static_cast<int32_t>(dec_pos.size());
        for (int j = 0; j < total_round - 1; ++j) {
          std::vector<int32_t> buf;
          buf.reserve(static_cast<size_t>(n * beam_width));
          for (int i = 0; i < n; ++i) {
            const int32_t base = dec_pos[i] + j;
            for (int b = 0; b < beam_width; ++b) {
              buf.push_back(base);
            }
          }
          mip.decode_positions_tensor_list.push_back(
              torch::tensor(buf, int_options));
        }
      }
    }
    // init naive block table
    if (!cached_naive_block_table_.defined()) {
      mip.naive_block_table =
          torch::arange(batch_size, int_options).unsqueeze(1);
      cached_naive_block_table_ = mip.naive_block_table;
    } else {
      mip.naive_block_table = cached_naive_block_table_;
    }
  }
#endif
}

std::optional<ForwardOutput> RecWorkerImpl::LlmRecPureDevicePipeline::step(
    const ForwardInput& input) {
  Timer timer;
  auto device = worker_.device_;
  device.set_device();

  ForwardInput& mutable_input = const_cast<ForwardInput&>(input);

  int32_t total_rounds = mutable_input.total_round;
  int32_t max_decode_step = total_rounds - 1;
  int32_t batch_size =
      mutable_input.input_params.paged_kv_last_page_len.numel();
  int32_t beam_width = mutable_input.beam_width;
  int32_t layer_num =
      static_cast<int32_t>(worker_.context_.get_model_args().n_layers());

  auto int_options = torch::TensorOptions().dtype(torch::kInt32).device(device);
  auto fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto paged_options =
      torch::TensorOptions().dtype(torch::kInt32).device(device);
  CHECK_GT(worker_.kv_caches_.size(), 0) << "KV caches are not initialized.";
  auto kv_cache_options = worker_.kv_caches_[0].get_k_cache().options();

  BeamSearchTensors beam_tensors =
      prepare_beam_search_tensors(batch_size, beam_width, total_rounds, device);

  FixedTensors fixed_tensors = prepare_fixed_tensors(
      batch_size, beam_width, max_decode_step, paged_options);

  mutable_input.input_params.num_heads =
      worker_.context_.get_model_args().n_heads();
  mutable_input.input_params.head_dim =
      worker_.context_.get_model_args().head_dim();
  mutable_input.input_params.beam_width = beam_width;

  ForwardOutput output;
  torch::Tensor logits;
  SampleOutput sample_output;
  torch::Tensor top_tokens;

  for (int32_t round = 0; round < total_rounds; ++round) {
    const auto& sampling_params = round > 0
                                      ? mutable_input.decoder_sampling_params
                                      : mutable_input.sampling_params;
    mutable_input.input_params.is_prefill = round == 0;
    mutable_input.input_params.current_round = round - 1;

    auto hidden_states =
        worker_.model_executor_->forward(mutable_input.token_ids,
                                         mutable_input.positions,
                                         worker_.kv_caches_,
                                         mutable_input.input_params);
    if (!hidden_states.defined()) {
      return std::nullopt;
    }

    if (sampling_params.selected_token_idxes.defined()) {
      logits = worker_.model_->logits(hidden_states,
                                      sampling_params.selected_token_idxes);
      sample_output = worker_.sampler_->forward(logits, sampling_params);
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
        update_input_for_next_round(mutable_input,
                                    round,
                                    sample_output,
                                    top_tokens,
                                    beam_tensors,
                                    batch_size,
                                    beam_width,
                                    max_decode_step,
                                    paged_options,
                                    fixed_tensors);
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

  device.synchronize_default_stream();

  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
  DeviceMonitor::get_instance().update_active_activation_memory(device.index());
  return output;
}

RecWorkerImpl::LlmRecPureDevicePipeline::BeamSearchTensors
RecWorkerImpl::LlmRecPureDevicePipeline::prepare_beam_search_tensors(
    int32_t batch_size,
    int32_t beam_width,
    int32_t total_rounds,
    const torch::Device& device) {
  auto int_options = torch::TensorOptions().dtype(torch::kInt32).device(device);
  auto fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);

  BeamSearchTensors tensors;
  tensors.sequence_group =
      torch::zeros({batch_size, beam_width, total_rounds}, int_options);
  int64_t num_seq = batch_size * beam_width;
  tensors.acc_logprob = torch::zeros({num_seq, 1}, fp32_options);
  tensors.out_log_probs = torch::zeros({num_seq, 1}, fp32_options);
  tensors.out_token_ids = torch::zeros({num_seq, 1}, int_options);
  tensors.out_token_index = torch::zeros({num_seq, 1}, int_options);
  tensors.out_beam_count_prefix_sums = torch::zeros({num_seq, 1}, int_options);
  tensors.out_seqgroup = torch::zeros_like(tensors.sequence_group);
  return tensors;
}

RecWorkerImpl::LlmRecPureDevicePipeline::FixedTensors
RecWorkerImpl::LlmRecPureDevicePipeline::prepare_fixed_tensors(
    int32_t batch_size,
    int32_t beam_width,
    int32_t max_decode_step,
    const torch::TensorOptions& paged_options) {
  FixedTensors tensors;
  tensors.batch_ids = torch::arange(0, batch_size, paged_options)
                          .unsqueeze(1)
                          .unsqueeze(2)
                          .expand({-1, beam_width, max_decode_step}) *
                      (beam_width * max_decode_step);

  tensors.beams_ids = torch::arange(0, beam_width, paged_options)
                          .unsqueeze(0)
                          .unsqueeze(2)
                          .expand({batch_size, -1, max_decode_step}) *
                      max_decode_step;

  tensors.max_decode_step_ids = torch::arange(0, max_decode_step, paged_options)
                                    .unsqueeze(0)
                                    .unsqueeze(1)
                                    .expand({batch_size, beam_width, -1});
  return tensors;
}

void RecWorkerImpl::LlmRecPureDevicePipeline::execute_beam_search(
    const torch::Tensor& top_tokens,
    const torch::Tensor& top_logprobs,
    BeamSearchTensors& beam_tensors,
    int32_t round,
    int32_t batch_size) {
#if defined(USE_NPU)
// TODO: implement beam search for NPU
#elif defined(USE_CUDA)
  xllm::kernel::cuda::beam_search(beam_tensors.acc_logprob,
                                  beam_tensors.sequence_group,
                                  top_tokens,
                                  top_logprobs,
                                  beam_tensors.out_log_probs,
                                  beam_tensors.out_token_ids,
                                  beam_tensors.out_token_index,
                                  beam_tensors.out_beam_count_prefix_sums,
                                  beam_tensors.out_seqgroup,
                                  batch_size,
                                  round);
#endif
}

void RecWorkerImpl::LlmRecPureDevicePipeline::execute_cache_select(
    const BeamSearchTensors& beam_tensors,
    ForwardInput& input,
    int32_t round,
    int32_t beam_width,
    int32_t layer_num) {
#if defined(USE_NPU)
// TODO: implement cache select for NPU
#elif defined(USE_CUDA)
  xllm::kernel::cuda::cache_select(beam_tensors.out_token_index,
                                   input.input_params.unshared_k_caches,
                                   input.input_params.unshared_v_caches,
                                   input.input_params.naive_block_table,
                                   beam_tensors.out_beam_count_prefix_sums,
                                   round - 1,
                                   beam_width,
                                   layer_num);
#endif
}

void RecWorkerImpl::LlmRecPureDevicePipeline::build_final_output(
    const torch::Tensor& logits,
    const SampleOutput& sample_output,
    const SamplingParameters& sampling_params,
    const BeamSearchTensors& beam_tensors,
    ForwardOutput& output) {
  output.logits = logits;
  output.sample_output = sample_output;
  output.do_sample = sampling_params.do_sample;
  output.logprobs = sampling_params.logprobs;
  output.max_top_logprobs = sampling_params.max_top_logprobs;
  output.beam_search_output.src_seq_idxes =
      beam_tensors.out_token_index.reshape({-1});
  output.beam_search_output.out_tokens =
      beam_tensors.out_token_ids.reshape({-1});
  output.beam_search_output.out_logprobs =
      beam_tensors.out_log_probs.reshape({-1});
  output.beam_sequence_group = beam_tensors.sequence_group;
}

void RecWorkerImpl::LlmRecPureDevicePipeline::compute_shared_kv_tensors(
    const ModelInputParams& input_params,
    int32_t batch_size,
    int32_t beam_size,
    const torch::TensorOptions& paged_options,
    torch::Tensor& shared_kv_len_offsets,
    torch::Tensor& shared_mask,
    torch::Tensor& shared_kv_indices,
    int32_t& shared_kv_len) {
  auto kv_cu_seq_lens = input_params.kv_seq_lens;
  shared_kv_len = FLAGS_max_token_per_req;
  auto batch_shared_kv_lens = torch::diff(kv_cu_seq_lens);

  shared_kv_len_offsets = torch::arange(0, shared_kv_len, paged_options)
                              .unsqueeze(0)
                              .expand({batch_size, shared_kv_len});

  auto beam_shared_kv_expanded =
      batch_shared_kv_lens.unsqueeze(1).expand({-1, shared_kv_len});

  shared_mask = (shared_kv_len_offsets < beam_shared_kv_expanded)
                    .unsqueeze(1)
                    .expand({-1, beam_size, -1});

  auto kv_cu_seq_lens_prefix = kv_cu_seq_lens.slice(0, 0, -1);
  auto shared_batch_offsets =
      kv_cu_seq_lens_prefix.unsqueeze(1).expand({-1, shared_kv_len});

  shared_kv_indices = (shared_batch_offsets + shared_kv_len_offsets)
                          .unsqueeze(1)
                          .expand({-1, beam_size, -1});
}

void RecWorkerImpl::LlmRecPureDevicePipeline::compute_unshared_kv_tensors(
    int32_t current_step,
    int32_t batch_size,
    int32_t shared_kv_len,
    const FixedTensors& fixed_tensors,
    torch::Tensor& unshared_kv_indices,
    torch::Tensor& unshared_mask) {
  uint32_t unshared_begin_index = shared_kv_len * batch_size;
  auto unshared_kv_offsets = fixed_tensors.batch_ids + fixed_tensors.beams_ids +
                             fixed_tensors.max_decode_step_ids;
  unshared_kv_indices = unshared_kv_offsets + unshared_begin_index;
  unshared_mask = fixed_tensors.max_decode_step_ids <= current_step;
}

void RecWorkerImpl::LlmRecPureDevicePipeline::build_paged_kv_indices(
    const torch::Tensor& shared_kv_indices,
    const torch::Tensor& unshared_kv_indices,
    const torch::Tensor& shared_mask,
    const torch::Tensor& unshared_mask,
    int32_t batch_size,
    int32_t beam_size,
    int32_t current_step,
    int32_t shared_kv_len,
    const torch::TensorOptions& paged_options,
    const ModelInputParams& input_params,
    torch::Tensor& paged_kv_indices,
    torch::Tensor& paged_kv_indptr,
    torch::Tensor& paged_kv_last_page_len) {
  auto full_mask = torch::cat({shared_mask, unshared_mask}, 2);
  auto full_kv_indices =
      torch::cat({shared_kv_indices, unshared_kv_indices}, 2);

  auto kv_cu_seq_lens = input_params.kv_seq_lens;
  auto batch_shared_kv_lens = torch::diff(kv_cu_seq_lens);
  uint32_t unshared_kv_len = current_step + 1;
  auto batch_beam_shared_kv_lens =
      (batch_shared_kv_lens.unsqueeze(1).expand({-1, beam_size}) +
       unshared_kv_len)
          .flatten();

  auto cumsum_result = torch::cumsum(batch_beam_shared_kv_lens, 0);
  paged_kv_indptr = torch::cat(
      {torch::zeros({1}, paged_options), cumsum_result.to(paged_options)}, 0);

  paged_kv_indices = full_kv_indices.masked_select(full_mask);
  paged_kv_last_page_len = torch::ones({batch_size * beam_size}, paged_options);
}

void RecWorkerImpl::LlmRecPureDevicePipeline::update_input_for_next_round(
    ForwardInput& input,
    int32_t current_step,
    const SampleOutput& sample_output,
    const torch::Tensor& top_tokens,
    const BeamSearchTensors& beam_tensors,
    int32_t batch_size,
    int32_t beam_size,
    int32_t max_decode_step,
    const torch::TensorOptions& paged_options,
    const FixedTensors& fixed_tensors) {
  if (current_step == 0) {
    input.token_ids = top_tokens.reshape({-1});
  } else {
    input.token_ids = beam_tensors.out_token_ids.reshape({-1});
  }

  if (!input.input_params.decode_positions_tensor_list.empty() &&
      current_step >= 0 &&
      current_step <
          static_cast<int32_t>(
              input.input_params.decode_positions_tensor_list.size())) {
    input.positions =
        input.input_params.decode_positions_tensor_list[current_step];
  }

  input.input_params.batch_forward_type = BatchForwardType(2);

  torch::Tensor shared_kv_len_offsets, shared_mask, shared_kv_indices;
  int32_t shared_kv_len;
  compute_shared_kv_tensors(input.input_params,
                            batch_size,
                            beam_size,
                            paged_options,
                            shared_kv_len_offsets,
                            shared_mask,
                            shared_kv_indices,
                            shared_kv_len);

  torch::Tensor unshared_kv_indices, unshared_mask;
  compute_unshared_kv_tensors(current_step,
                              batch_size,
                              shared_kv_len,
                              fixed_tensors,
                              unshared_kv_indices,
                              unshared_mask);

  torch::Tensor paged_kv_indices, paged_kv_indptr, paged_kv_last_page_len;
  build_paged_kv_indices(shared_kv_indices,
                         unshared_kv_indices,
                         shared_mask,
                         unshared_mask,
                         batch_size,
                         beam_size,
                         current_step,
                         shared_kv_len,
                         paged_options,
                         input.input_params,
                         paged_kv_indices,
                         paged_kv_indptr,
                         paged_kv_last_page_len);

  input.input_params.paged_kv_indices = paged_kv_indices;
  input.input_params.paged_kv_indptr = paged_kv_indptr;
  input.input_params.paged_kv_last_page_len = paged_kv_last_page_len;
}

RecWorkerImpl::RecWorkerImpl(const ParallelArgs& parallel_args,
                             const torch::Device& device,
                             const runtime::Options& options)
    : LLMWorkerImpl(parallel_args, device, options) {
  if (!is_driver()) {
    return;
  }

  const int64_t num_threads = std::max<int64_t>(
      1, util::get_int_env("XLLM_REC_INPUT_BUILDER_THREADS", 16));
  input_builder_thread_pool_ =
      std::make_shared<ThreadPool>(static_cast<size_t>(num_threads));
}

bool RecWorkerImpl::init_model(ModelContext& context) {
  const auto& model_type = context.get_model_args().model_type();
  rec_model_kind_ = get_rec_model_kind(model_type);
  CHECK(rec_model_kind_ != RecModelKind::kNone)
      << "Unsupported rec model_type: " << model_type;

  // Create work pipeline first
  auto pipeline_type = get_rec_pipeline_type(rec_model_kind_);
  work_pipeline_ = create_pipeline(pipeline_type, *this);

  // Let pipeline create model
  return work_pipeline_->create_model(*this, context);
}

ForwardInput RecWorkerImpl::prepare_inputs(Batch& batch) {
  CHECK(work_pipeline_ != nullptr) << "RecWorkerImpl is not initialized.";
  return work_pipeline_->prepare_inputs(batch);
}

void RecWorkerImpl::prepare_work_before_execute(
    const ForwardInput& inputs,
    ForwardInput& processed_inputs) {
  CHECK(work_pipeline_ != nullptr) << "RecWorkerImpl is not initialized.";
  work_pipeline_->prepare_work_before_execute(inputs, processed_inputs);
}

torch::Tensor RecWorkerImpl::merge_embeddings_by_indices(
    const torch::Tensor& input_tokens_embedding,
    const torch::Tensor& input_embedding,
    const std::vector<int64_t>& input_indices) {
  CHECK_EQ(input_embedding.dim(), 2);
  CHECK_EQ(input_tokens_embedding.dim(), 2);
  CHECK_EQ(input_tokens_embedding.size(1), input_embedding.size(1));
  CHECK_EQ(input_tokens_embedding.dtype(), input_embedding.dtype());
  CHECK_EQ(input_tokens_embedding.device(), input_embedding.device());

  const int64_t total_rows =
      input_tokens_embedding.size(0) + input_embedding.size(0);
  const int64_t cols = input_embedding.size(1);

  torch::Device device = input_embedding.device();
  torch::Tensor merged = torch::empty(
      {total_rows, cols}, torch::dtype(input_embedding.dtype()).device(device));

  std::vector<int64_t> input_embedding_indices;
  for (int64_t i = 0; i < total_rows; ++i) {
    if (std::find(input_indices.begin(), input_indices.end(), i) ==
        input_indices.end()) {
      input_embedding_indices.push_back(i);
    }
  }

  CHECK_EQ(input_embedding_indices.size(), input_embedding.size(0));

  torch::Tensor input_embedding_indices_tensor =
      torch::tensor(input_embedding_indices, torch::kInt64).to(device);
  merged.index_put_({input_embedding_indices_tensor, torch::indexing::Ellipsis},
                    input_embedding);

  torch::Tensor input_indices_tensor =
      torch::tensor(input_indices, torch::kInt64).to(device);
  merged.index_put_({input_indices_tensor, torch::indexing::Ellipsis},
                    input_tokens_embedding);

  return merged;
}

std::optional<ForwardOutput> RecWorkerImpl::step(const ForwardInput& input) {
  CHECK(work_pipeline_ != nullptr) << "RecWorkerImpl is not initialized.";
  return work_pipeline_->step(input);
}

// ============================================================
// RecWorkerImpl pipeline factory (static method)
// ============================================================
std::unique_ptr<RecWorkerImpl::RecWorkPipeline> RecWorkerImpl::create_pipeline(
    RecPipelineType type,
    RecWorkerImpl& worker) {
  switch (type) {
    case RecPipelineType::kLlmRecDefault:
      return std::make_unique<LlmRecWorkPipeline>(worker);
    case RecPipelineType::kLlmRecWithMmData:
      return std::make_unique<LlmRecWithMmDataWorkPipeline>(worker);
    case RecPipelineType::kOneRecDefault:
      return std::make_unique<OneRecWorkPipeline>(worker);
    case RecPipelineType::kLlmRecPureDevicePipeline:
      return std::make_unique<LlmRecPureDevicePipeline>(worker);
    default:
      LOG(FATAL) << "Unknown RecWorkerImpl pipeline type: "
                 << static_cast<int>(type);
      return nullptr;
  }
}

}  // namespace xllm
