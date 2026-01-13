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

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "common/rec_model_utils.h"
#include "runtime/llm_worker_impl.h"
#include "util/threadpool.h"

namespace xllm {

class RecWorkerImpl : public LLMWorkerImpl {
 public:
  RecWorkerImpl(const ParallelArgs& parallel_args,
                const torch::Device& device,
                const runtime::Options& options);

  bool init_model(ModelContext& context) override;

  ForwardInput prepare_inputs(Batch& batch) override;

  void prepare_work_before_execute(const ForwardInput& inputs,
                                   ForwardInput& processed_inputs) override;

  std::optional<ForwardOutput> step(const ForwardInput& input) override;

 protected:
  std::shared_ptr<ThreadPool> input_builder_thread_pool_;

 private:
  class RecWorkPipeline {
   public:
    virtual ~RecWorkPipeline() = default;

    virtual bool create_model(RecWorkerImpl& worker, ModelContext& context) = 0;

    virtual ForwardInput prepare_inputs(Batch& batch) = 0;

    virtual void prepare_work_before_execute(
        const ForwardInput& inputs,
        ForwardInput& processed_inputs) = 0;

    virtual std::optional<ForwardOutput> step(const ForwardInput& input) = 0;
  };

  class LlmRecWorkPipeline final : public RecWorkPipeline {
   public:
    explicit LlmRecWorkPipeline(RecWorkerImpl& worker);

    bool create_model(RecWorkerImpl& worker, ModelContext& context) override;

    ForwardInput prepare_inputs(Batch& batch) override;

    void prepare_work_before_execute(const ForwardInput& inputs,
                                     ForwardInput& processed_inputs) override;

    std::optional<ForwardOutput> step(const ForwardInput& input) override;

   private:
    RecWorkerImpl& worker_;
  };

  class OneRecWorkPipeline final : public RecWorkPipeline {
   public:
    explicit OneRecWorkPipeline(RecWorkerImpl& worker);

    bool create_model(RecWorkerImpl& worker, ModelContext& context) override;

    ForwardInput prepare_inputs(Batch& batch) override;

    void prepare_work_before_execute(const ForwardInput& inputs,
                                     ForwardInput& processed_inputs) override;

    std::optional<ForwardOutput> step(const ForwardInput& input) override;

   private:
    RecWorkerImpl& worker_;
  };

  class LlmRecWithMmDataWorkPipeline final : public RecWorkPipeline {
   public:
    explicit LlmRecWithMmDataWorkPipeline(RecWorkerImpl& worker);

    bool create_model(RecWorkerImpl& worker, ModelContext& context) override;

    ForwardInput prepare_inputs(Batch& batch) override;

    void prepare_work_before_execute(const ForwardInput& inputs,
                                     ForwardInput& processed_inputs) override;

    std::optional<ForwardOutput> step(const ForwardInput& input) override;

   private:
    RecWorkerImpl& worker_;
  };

  class LlmRecPureDevicePipeline final : public RecWorkPipeline {
   public:
    explicit LlmRecPureDevicePipeline(RecWorkerImpl& worker);

    bool create_model(RecWorkerImpl& worker, ModelContext& context) override;

    ForwardInput prepare_inputs(Batch& batch) override;

    void prepare_work_before_execute(const ForwardInput& inputs,
                                     ForwardInput& processed_inputs) override;

    std::optional<ForwardOutput> step(const ForwardInput& input) override;

   private:
    // Beam search related tensors
    struct BeamSearchTensors {
      torch::Tensor sequence_group;   // [batch_size, beam_width, total_rounds]
      torch::Tensor acc_logprob;      // [num_seq, 1]
      torch::Tensor out_log_probs;    // [num_seq, 1]
      torch::Tensor out_token_ids;    // [num_seq, 1]
      torch::Tensor out_token_index;  // [num_seq, 1]
      torch::Tensor out_beam_count_prefix_sums;  // [num_seq, 1]
      torch::Tensor out_seqgroup;  // [batch_size, beam_width, total_rounds]
    };

    // Fixed tensors for multi-round decoding
    struct FixedTensors {
      torch::Tensor batch_ids;  // [batch_size, beam_width, max_decode_step]
      torch::Tensor beams_ids;  // [batch_size, beam_width, max_decode_step]
      torch::Tensor
          max_decode_step_ids;  // [batch_size, beam_width, max_decode_step]
    };
    // Prepare beam search tensors
    BeamSearchTensors prepare_beam_search_tensors(int32_t batch_size,
                                                  int32_t beam_width,
                                                  int32_t total_rounds,
                                                  const torch::Device& device);

    // Prepare fixed tensors for multi-round decoding
    FixedTensors prepare_fixed_tensors(
        int32_t batch_size,
        int32_t beam_width,
        int32_t max_decode_step,
        const torch::TensorOptions& paged_options);

    // Execute beam search kernel
    void execute_beam_search(const torch::Tensor& top_tokens,
                             const torch::Tensor& top_logprobs,
                             BeamSearchTensors& beam_tensors,
                             int32_t round,
                             int32_t batch_size);

    // Execute cache select kernel
    void execute_cache_select(const BeamSearchTensors& beam_tensors,
                              ForwardInput& input,
                              int32_t round,
                              int32_t beam_width,
                              int32_t layer_num);

    // Build final output from beam search results
    void build_final_output(const torch::Tensor& logits,
                            const SampleOutput& sample_output,
                            const SamplingParameters& sampling_params,
                            const BeamSearchTensors& beam_tensors,
                            ForwardOutput& output);

    // Compute shared KV cache related tensors
    void compute_shared_kv_tensors(const ModelInputParams& input_params,
                                   int32_t batch_size,
                                   int32_t beam_size,
                                   const torch::TensorOptions& paged_options,
                                   torch::Tensor& shared_kv_len_offsets,
                                   torch::Tensor& shared_mask,
                                   torch::Tensor& shared_kv_indices,
                                   int32_t& shared_kv_len);

    // Compute unshared KV cache related tensors
    void compute_unshared_kv_tensors(int32_t current_step,
                                     int32_t batch_size,
                                     int32_t shared_kv_len,
                                     const FixedTensors& fixed_tensors,
                                     torch::Tensor& unshared_kv_indices,
                                     torch::Tensor& unshared_mask);

    // Build paged KV indices and indptr
    void build_paged_kv_indices(const torch::Tensor& shared_kv_indices,
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
                                torch::Tensor& paged_kv_last_page_len);

    // Update input for next round in multi-round decoding
    void update_input_for_next_round(ForwardInput& input,
                                     int32_t current_step,
                                     const SampleOutput& sample_output,
                                     const torch::Tensor& top_tokens,
                                     const BeamSearchTensors& beam_tensors,
                                     int32_t batch_size,
                                     int32_t beam_size,
                                     int32_t max_decode_step,
                                     const torch::TensorOptions& paged_options,
                                     const FixedTensors& fixed_tensors);

    // Cached tensors for reuse across requests
    std::vector<torch::Tensor> cached_full_k_caches_;
    std::vector<torch::Tensor> cached_full_v_caches_;
    std::vector<torch::Tensor> cached_unshared_k_caches_;
    std::vector<torch::Tensor> cached_unshared_v_caches_;
    torch::Tensor cached_beam_width_tensor_;
    std::vector<torch::Tensor> cached_current_round_tensor_list_;
    torch::Tensor cached_naive_block_table_;
    torch::Tensor cached_prefill_preallocated_output_;
    torch::Tensor cached_decode_preallocated_output_;

    RecWorkerImpl& worker_;
  };

  // Factory method to create pipeline (can access private classes)
  static std::unique_ptr<RecWorkPipeline> create_pipeline(
      RecPipelineType type,
      RecWorkerImpl& worker);

  torch::Tensor merge_embeddings_by_indices(
      const torch::Tensor& input_tokens_embedding,
      const torch::Tensor& input_embedding,
      const std::vector<int64_t>& input_indices);

  std::unique_ptr<RecWorkPipeline> work_pipeline_;

  RecModelKind rec_model_kind_ = RecModelKind::kNone;
};

}  // namespace xllm
