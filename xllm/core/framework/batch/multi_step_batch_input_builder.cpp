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

#include "multi_step_batch_input_builder.h"

#include <c10/core/DeviceType.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstring>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/global_flags.h"
#include "common/metrics.h"
#include "framework/batch/mposition.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/request/sequence.h"
#include "framework/sampling/sampling_params.h"
#include "runtime/params_utils.h"
#include "util/blocking_counter.h"
#include "util/slice.h"
#include "util/tensor_helper.h"
#include "util/threadpool.h"
#include "util/utils.h"

namespace xllm {

MultiStepBatchInputBuilder::MultiStepBatchInputBuilder(
    const std::vector<Sequence*>& sequences,
    const std::vector<uint32_t>& allowed_max_tokens,
    const std::vector<torch::Tensor>& input_embeddings_vec,
    const std::vector<MMData>& mm_data_vec,
    std::vector<BlockTransferInfo>* swap_block_transfer_infos,
    const uint64_t batch_id,
    const ModelArgs* args,
    BatchForwardType batch_forward_type,
    ThreadPool* thread_pool)
    : sequences_(sequences),
      allowed_max_tokens_(allowed_max_tokens),
      input_embeddings_vec_(input_embeddings_vec),
      mm_data_vec_(mm_data_vec),
      args_(args),
      num_sequences_(static_cast<int32_t>(sequences.size())),
      swap_block_transfer_infos_(swap_block_transfer_infos),
      thread_pool_(thread_pool),
      batch_id_(batch_id) {
  // LOG(INFO) << "sequences_.size(): " << sequences_.size();
  // LOG(INFO) << "batch_forward_type: " << batch_forward_type.to_string();
  if (args_ != nullptr) {
    use_mrope_ = (args_->rope_scaling_rope_type() == "mrope");
  }
  // Initialize MultiStep specific state
  multi_step_state_.total_steps = FLAGS_max_decode_rounds;
  // multi_step_state_.step_tokens_vec.reserve(1000);
  // multi_step_state_.step_positions_vec.reserve(1000);
  // TODO: Add multi-step specific initialization
  multi_step_state_.base_state.batch_forward_type = batch_forward_type;
  // LOG(INFO) << "multi_step_state_.base_state.batch_forward_type: " << multi_step_state_.base_state.batch_forward_type.to_string();
}

void MultiStepBatchInputBuilder::process_single_sequence(
    int32_t seq_index,
    BuilderState* state_ptr,
    std::unordered_set<int32_t>* write_block_ids_ptr) {
  MultiStepBuilderState& state = multi_step_state_;
  BuilderState& base_state = state.base_state;

  auto* sequence = sequences_[seq_index];
  const auto token_ids = sequence->tokens();
  const uint32_t n_tokens = token_ids.size();
  const uint32_t n_kv_cache_tokens = sequence->kv_state().kv_cache_tokens_num();

  // Validate and calculate sequence lengths
  CHECK(allowed_max_tokens_[seq_index] > 0);
  uint32_t q_seq_len =
      std::min(n_tokens - n_kv_cache_tokens, allowed_max_tokens_[seq_index]);
  uint32_t seq_len = q_seq_len + n_kv_cache_tokens;

  // add decode data;
  uint32_t decode_q_seq_len = 1;
  uint32_t decode_seq_len = n_kv_cache_tokens + 1;

  // Validation
  // CHECK_GE(sequence->kv_state().current_max_tokens_capacity(), seq_len);
  CHECK_GT(q_seq_len, 0) << "at least one token should be processed. "
                         << "n_tokens: " << n_tokens
                         << ", n_kv_cache_tokens: " << n_kv_cache_tokens
                         << ", current_max_tokens_capacity: "
                         << sequence->kv_state().current_max_tokens_capacity()
                         << ", allowed_max_tokens: "
                         << allowed_max_tokens_[seq_index];

  // Update state
  int32_t offset = is_mtp_decode_ ? -1 : 0;
  base_state.empty_kv_cache = true;
  base_state.max_seq_len = std::max(base_state.max_seq_len, seq_len);
  base_state.q_max_seq_len = std::max(base_state.q_max_seq_len, q_seq_len);
#if defined(USE_NPU)
  base_state.seq_lens.push_back(seq_len);
  base_state.q_seq_lens.push_back(q_seq_len);
  if (FLAGS_max_decode_rounds > 0) {
    state.decode_seq_lens.push_back(decode_seq_len);
    state.decode_q_seq_lens.push_back(decode_q_seq_len);
  }
#elif defined(USE_MLU) || defined(USE_CUDA) || defined(USE_ILU)
  base_state.seq_lens.push_back(base_state.seq_lens.back() + seq_len);
  base_state.q_seq_lens.push_back(base_state.q_seq_lens.back() + q_seq_len);
  if (FLAGS_max_decode_rounds > 0) {
    state.decode_seq_lens.push_back(decode_seq_len);
    state.decode_q_seq_lens.push_back(decode_q_seq_len);
  }
#endif

  // Call our enhanced method to process tokens and positions
  // This handles both regular decode and step-level decode cases
  extract_tokens_and_positions(sequence, n_kv_cache_tokens, seq_len, &state);

  // not Setup KV cache for prefill

  if (!FLAGS_enable_continuous_kvcache) {
    setup_kv_cache_info(sequence,
                        n_kv_cache_tokens,
                        decode_seq_len,
                        decode_q_seq_len,
                        &base_state,
                        write_block_ids_ptr);
  } else {
    setup_continuous_kv_cache_info(sequence,
                                   n_kv_cache_tokens,
                                   decode_seq_len,
                                   decode_q_seq_len,
                                   &base_state);
  }

  // Track prefill sequences
  if (sequence->stage() == SequenceStage::PREFILL) {
    base_state.prefill_seq_len++;
  }

  // Input for beam search kernel
  // if (FLAGS_enable_beam_search_kernel && sequence->check_beam_search()) {
  //   int32_t bw = std::max(1, FLAGS_beam_width);
  //   base_state.acc_logprob_vec.insert(base_state.acc_logprob_vec.end(), bw,
  //   0.0f);
  // }

  // Multi-step specific processing
}

RawForwardInput MultiStepBatchInputBuilder::build_raw_forward_input() {
  // Reset multi-step state for this build
  // multi_step_state_ = MultiStepBuilderState{};
  multi_step_state_.total_steps = FLAGS_max_decode_rounds;

  is_mtp_decode_ = false;
  // LOG(INFO) << "sequences_.size(): " << sequences_.size();
  // Single-threaded processing for now; can be extended to use thread_pool_
  for (int32_t i = 0; i < static_cast<int32_t>(sequences_.size()); ++i) {
    process_single_sequence(i, &multi_step_state_.base_state, nullptr);
  }
  // LOG(INFO) << "multi_step_state_.base_state.batch_forward_type: " << multi_step_state_.base_state.batch_forward_type.to_string();
  return state_to_raw_forward_input(&multi_step_state_.base_state);
}

void MultiStepBatchInputBuilder::extract_tokens_and_positions(
    Sequence* sequence,
    uint32_t n_kv_cache_tokens,
    uint32_t seq_len,
    MultiStepBuilderState* state_ptr) {
  // First build the "base" view that matches the single-round builder
  BuilderState& base_state = state_ptr->base_state;

  const auto& token_ids = sequence->tokens();
  const uint32_t n_tokens = token_ids.size();

  // Prepare adjusted token counts for sampling
  std::unordered_map<int32_t, int32_t> adjusted_token_to_count_map;
  for (uint32_t j = n_kv_cache_tokens; j < seq_len; ++j) {
    // skip prompt tokens except the last one
    if (j + 1 < n_tokens) continue;
    ++adjusted_token_to_count_map[token_ids[j]];
  }

  // Handle MRope positions
  if (use_mrope_) {
    const auto& args = *args_;
    MPositionHelper helper(*sequence, args);
    base_state.mrope_positions_vec.push_back(helper.get_positions());
  }

  // Process each token
  for (uint32_t j = n_kv_cache_tokens; j < seq_len; ++j) {
    base_state.flatten_tokens_vec.push_back(token_ids[j]);

    if (!use_mrope_) {
      base_state.flatten_positions_vec.push_back(static_cast<int32_t>(j));
    }

    // Handle sampling for last tokens
    if (j + 1 < n_tokens) continue;

    // Inlined sampling/unique-token logic on base_state.
    BuilderState& state = base_state;

    const auto token_id = sequence->tokens()[j];
    // Adjust token count
    --adjusted_token_to_count_map[token_id];

    // Select token for sampling
    state.selected_token_idxes.push_back(
        static_cast<int32_t>(state.flatten_tokens_vec.size() - 1));
    state.sampling_params.push_back(sequence->sampling_param());

    // Process unique tokens
    const auto& seq_token_counts = sequence->token_to_count_map();
    auto& ids = state.unique_token_ids_vec.emplace_back();
    auto& counts = state.unique_token_counts_vec.emplace_back();

    ids.reserve(seq_token_counts.size());
    counts.reserve(seq_token_counts.size());

    for (const auto& [tok_id, count] : seq_token_counts) {
      const auto it = adjusted_token_to_count_map.find(tok_id);
      const auto adjust_count =
          (it != adjusted_token_to_count_map.end()) ? it->second : 0;

      if (count > adjust_count) {
        ids.push_back(tok_id);
        counts.push_back(count - adjust_count);
      }
    }

    state.unique_token_lens_vec.push_back(static_cast<int32_t>(ids.size()));

    // Mark sample token if it's the last token
    if (j == seq_len - 1) {
      state.sample_idxes.push_back(
          static_cast<int32_t>(state.selected_token_idxes.size() - 1));
    }
  }

  // Add extra token id
  if (n_tokens == seq_len) {
    // last chunk of prefill and decode
    // add -1 as extra token id
    base_state.extra_token_ids.push_back(-1);
    base_state.embedding_ids.push_back(sequence->get_embedding_id());
  } else {
    base_state.extra_token_ids.push_back(token_ids[seq_len]);
  }

  // begin process decode data
  seq_len = n_kv_cache_tokens + 1;
  // std::unordered_map<int32_t, int32_t> adjusted_token_to_count_map;
  // 从第一个 decode 到 max_decode round 都生成采样参数，
  // 都采用prompt最后一个token做占位，实际上这些需要执行 beam_width 次
  // 应当在 worker 上再broadcast
  // uint32_t pos_idx = n_kv_cache_tokens;
  // adjusted_token_to_count_map[token_ids[pos_idx - 1]] = 1;
  // TODO_1111 不考虑 use_mrope_ 为true 的情况
  // 是对 sequence 循环调的这个，所以每个sequence 内容理论上不一样，但是prefill
  // 阶段只有一个 sequnece 所以在worker 侧需要替换或broadcast.
  // state_ptr->flatten_tokens_vec.push_back(token_ids[pos_idx - 1]);
  // state_ptr->flatten_positions_vec.push_back(static_cast<int32_t>(pos_idx));
  // 不需要执行 handle_sampling_parameters 直接填充
  // handle_sampling_parameters(
  //     sequence, pos_idx, seq_len, adjusted_token_to_count_map, state_ptr);
  // state_ptr->extra_token_ids.push_back(-1);
  // state_ptr->embedding_ids.push_back(sequence->get_embedding_id());
  // 用于sample，每次都一样
  // 以下需要配套构造 sample_params ，在param_utils.cpp 中
  // proto_to_forward_input 中使用 因为每个 request 的 每个 sequence
  // 都一样，所以decode 可以只传一份，在 worker 端进行broadcast.
  uint32_t prompt_len = sequence->num_prompt_tokens();
  state_ptr->decode_positions_vec.push_back(static_cast<int32_t>(prompt_len));

  int32_t bw = std::max(1, FLAGS_beam_width);
  const int32_t sel_start =
      static_cast<int32_t>(state_ptr->decode_selected_token_idxes.size());
  state_ptr->decode_selected_token_idxes.reserve(sel_start + bw);
  state_ptr->decode_sample_idxes.reserve(state_ptr->decode_sample_idxes.size() +
                                         bw);
  state_ptr->decode_unique_token_ids_vec.resize(
      state_ptr->decode_unique_token_ids_vec.size() + bw);
  state_ptr->decode_unique_token_counts_vec.resize(
      state_ptr->decode_unique_token_counts_vec.size() + bw);
  state_ptr->decode_unique_token_lens_vec.insert(
      state_ptr->decode_unique_token_lens_vec.end(), bw, 0);
  state_ptr->decode_sampling_params.reserve(
      state_ptr->decode_sampling_params.size() + bw);
  for (int32_t i = 0; i < bw; ++i) {
    const int32_t idx = sel_start + i;
    state_ptr->decode_selected_token_idxes.push_back(idx);
    state_ptr->decode_sample_idxes.push_back(idx);
    state_ptr->decode_sampling_params.push_back(sequence->sampling_param());
  }
}

void MultiStepBatchInputBuilder::setup_kv_cache_info(
    Sequence* sequence,
    uint32_t n_kv_cache_tokens,
    uint32_t seq_len,
    uint32_t q_seq_len,
    BuilderState* state_ptr,
    std::unordered_set<int32_t>* write_block_ids_ptr) {
#if defined(USE_NPU)
  (void)write_block_ids_ptr;
  BuilderState& state = *state_ptr;
  const auto blocks = sequence->kv_state().kv_blocks();
  std::vector<int32_t> block_ids;
  block_ids.reserve(blocks.size());
  for (const auto& block : blocks) {
    block_ids.push_back(block.id());
  }
  state.block_tables_vec.emplace_back(std::move(block_ids));
#elif defined(USE_CUDA)
  MultiStepBuilderState& state = multi_step_state_;
  BuilderState& base_state = state.base_state;

  std::unordered_set<int32_t>& write_block_ids =
      write_block_ids_ptr ? *write_block_ids_ptr : write_block_ids_;

  // update kv cache tokens num
  sequence->kv_state().incr_kv_cache_tokens_num(/*size=*/q_seq_len);

  int32_t offset = is_mtp_decode_ ? -1 : 0;
  seq_len += offset;
  n_kv_cache_tokens += offset;
  const auto blocks = sequence->kv_state().kv_blocks();
  const auto slot_ids =
      sequence->kv_state().kv_cache_slots(n_kv_cache_tokens, seq_len);
  base_state.new_token_slot_ids.insert(
      base_state.new_token_slot_ids.end(), slot_ids.begin(), slot_ids.end());

  std::vector<int32_t> block_ids;
  std::vector<uint64_t> u_block_ids;
  block_ids.reserve(blocks.size());
  int32_t block_size = 0;
  for (const auto& block : blocks) {
    block_size = block.size();
    block_ids.push_back(block.id());
    u_block_ids.emplace_back(block.id());
    base_state.paged_kv_indices.push_back(block.id());
    // LOG(INFO) << "block.id(): " << block.id();
  }
  base_state.paged_kv_indptr.push_back(base_state.paged_kv_indptr.back() + blocks.size());
  int32_t last_page_len =
      (seq_len % block_size == 0) ? block_size : seq_len % block_size;
  base_state.paged_kv_last_page_len.push_back(last_page_len);

  // calculate the block ids that need to be written
  int32_t kv_cache_block_idx = n_kv_cache_tokens / block_size;
  for (auto iter = block_ids.begin() + kv_cache_block_idx;
       iter != block_ids.end();
       ++iter) {
    write_block_ids.insert(*iter);
  }

  auto& transfer_kv_info = sequence->kv_state().transfer_kv_info();
  if (transfer_kv_info.has_value()) {
    base_state.transfer_kv_infos.emplace_back(transfer_kv_info.value());
    base_state.transfer_kv_infos.back().local_blocks_ids = std::move(u_block_ids);
  }
  // LOG(INFO) << "before block_tables_vec.emplace_back";
  base_state.block_tables_vec.emplace_back(std::move(block_ids));
#endif

}

void MultiStepBatchInputBuilder::setup_continuous_kv_cache_info(
    Sequence* sequence,
    uint32_t n_kv_cache_tokens,
    uint32_t seq_len,
    uint32_t q_seq_len,
    BuilderState* state_ptr) {
  (void)sequence;
  (void)n_kv_cache_tokens;
  (void)seq_len;
  (void)q_seq_len;
  (void)state_ptr;
}

ForwardInput MultiStepBatchInputBuilder::state_to_forward_input() {
  // No current caller relies on the ForwardInput path for multi-step;
  // keep a minimal implementation that mirrors the RawForwardInput metadata.
  ForwardInput forward_input;

  // Add multi-step specific data using existing ForwardInput fields
  auto& multi_step_state = multi_step_state_;
  multi_step_state.total_steps = FLAGS_max_decode_rounds;

  // Set step-level decode metadata for multi-step processing
  // These fields are already present in ForwardInput for step-level decode
  forward_input.beam_width = FLAGS_beam_width;
  forward_input.total_round = multi_step_state.total_steps;

  // Set full_kv_shape if we have multi-step decode data
  if (!multi_step_state.decode_seq_lens.empty() && !sequences_.empty()) {
    // Set decode kv cache shape for step-level decode
    // Format: [batch_size * beam_width, n_kv_heads, step_rounds, head_dim]
    int64_t batch_size = static_cast<int64_t>(sequences_.size());
    int64_t step_rounds = static_cast<int64_t>(multi_step_state.total_steps);

    int64_t n_kv_heads =
        args_ ? args_->n_kv_heads().value_or(args_->n_heads()) : 0;
    int64_t head_dim = args_ ? args_->head_dim() : 0;

    forward_input.full_kv_shape = {
      batch_size * FLAGS_max_token_per_req +
          batch_size * FLAGS_beam_width * (FLAGS_max_decode_rounds - 1),
      n_kv_heads,
      head_dim};
  }

  if (!multi_step_state.decode_seq_lens.empty()) {
    auto tensor_options = torch::TensorOptions()
                              .dtype(torch::kInt)
                              .device(torch::kCPU)
                              .pinned_memory(true);
    forward_input.input_params.decode_kv_seq_lens =
        torch::tensor(multi_step_state.decode_seq_lens, tensor_options);
    forward_input.input_params.decode_q_seq_lens =
        torch::tensor(multi_step_state.decode_q_seq_lens, tensor_options);
    forward_input.input_params.decode_kv_seq_lens_vec =
        multi_step_state.decode_seq_lens;
    forward_input.input_params.decode_q_seq_lens_vec =
        multi_step_state.decode_q_seq_lens;
  }

  if (!multi_step_state.decode_positions_vec.empty()) {
    forward_input.decode_positions_vec = multi_step_state.decode_positions_vec;
  }

  return forward_input;
}

RawForwardInput MultiStepBatchInputBuilder::state_to_raw_forward_input(
    BuilderState* state_ptr) {
  BuilderState& src = state_ptr ? *state_ptr : multi_step_state_.base_state;
  if (src.flatten_tokens_vec.empty()) {
    return {};
  }
  // LOG(INFO) << "multi_step_state_.base_state.batch_forward_type: " << multi_step_state_.base_state.batch_forward_type.to_string();
  RawForwardInput raw_forward_input;
  VLOG(1) << "[SEL/RAW] selected_token_idxes.size(before move)="
          << src.selected_token_idxes.size();
  raw_forward_input.flatten_tokens_vec = std::move(src.flatten_tokens_vec);
  raw_forward_input.flatten_positions_vec =
      std::move(src.flatten_positions_vec);
  raw_forward_input.sampling_params = std::move(src.sampling_params);
  raw_forward_input.selected_token_idxes = std::move(src.selected_token_idxes);
  raw_forward_input.sample_idxes = std::move(src.sample_idxes);
  raw_forward_input.unique_token_ids_vec = std::move(src.unique_token_ids_vec);
  raw_forward_input.unique_token_counts_vec =
      std::move(src.unique_token_counts_vec);
  raw_forward_input.unique_token_lens_vec =
      std::move(src.unique_token_lens_vec);
  raw_forward_input.empty_kv_cache = src.empty_kv_cache;
  // LOG(INFO) << "src.batch_forward_type: " << src.batch_forward_type.to_string();
  raw_forward_input.batch_forward_type = src.batch_forward_type;
  raw_forward_input.max_seq_len = src.max_seq_len;
  raw_forward_input.q_max_seq_len = src.q_max_seq_len;
  raw_forward_input.seq_lens = std::move(src.seq_lens);
  raw_forward_input.q_seq_lens = std::move(src.q_seq_lens);
  raw_forward_input.new_token_slot_ids = std::move(src.new_token_slot_ids);
  // LOG(INFO) << "src.block_tables_vec.size(): " << src.block_tables_vec.size();
  raw_forward_input.block_tables_vec = std::move(src.block_tables_vec);
  raw_forward_input.num_sequences = num_sequences_;
  raw_forward_input.transfer_kv_infos = std::move(src.transfer_kv_infos);

  // for flashinfer
  raw_forward_input.paged_kv_indptr = std::move(src.paged_kv_indptr);
  raw_forward_input.paged_kv_indices = std::move(src.paged_kv_indices);
  raw_forward_input.paged_kv_last_page_len =
      std::move(src.paged_kv_last_page_len);

  raw_forward_input.embedding_ids = std::move(src.embedding_ids);
  raw_forward_input.extra_token_ids = std::move(src.extra_token_ids);
  // beam search kernel input
  if (!src.acc_logprob_vec.empty()) {
    raw_forward_input.acc_logprob_vec = std::move(src.acc_logprob_vec);
  }

  if (FLAGS_enable_continuous_kvcache) {
    raw_forward_input.new_cache_slot_offsets =
        std::move(src.new_cache_slot_offsets);
    raw_forward_input.kv_cache_start_offsets =
        std::move(src.kv_cache_start_offsets);
  }

  if (!mm_data_vec_.empty()) {
    MMData mm_data = MMData::batch(mm_data_vec_);
    const auto& res = mm_data.get<torch::Tensor>("embedding");
    if (res && res.value().defined()) {
      auto push_embedding = [&](torch::Tensor t) {
        if (!t.defined() || t.numel() == 0) return;
        t = t.to(torch::kCPU).contiguous().to(torch::kFloat32);
        std::vector<float> v(static_cast<size_t>(t.numel()));
        std::memcpy(v.data(), t.data_ptr<float>(), v.size() * sizeof(float));
        raw_forward_input.embeddings.emplace_back(std::move(v));
      };
      if (FLAGS_max_decode_rounds > 0) {
        torch::Tensor embeddings = res.value();
        if (embeddings.dim() == 1) {
          if (embeddings.defined() && embeddings.numel() > 0) {
            push_embedding(embeddings);
          }
        } else if (embeddings.dim() >= 2) {
          for (int64_t output_idx = 0; output_idx < embeddings.size(0);
               ++output_idx) {
            torch::Tensor embedding = embeddings[output_idx];
            if (!embedding.defined() || embedding.numel() == 0) {
              continue;
            }
            push_embedding(embedding);
          }
        }
      } else {
        torch::Tensor embeddings = res.value();
        for (int64_t output_idx = 0; output_idx < embeddings.size(0);
             ++output_idx) {
          push_embedding(embeddings[output_idx]);
        }
      }
    }
  }

  // Swap blocks (optional). Align with BatchInputBuilder's packing behavior:
  // - If block-copy kernel enabled: pack to src/dst indices + cum_sum
  // - Else: keep swap_blocks as a flat list of (src,dst)
  if (swap_block_transfer_infos_ != nullptr &&
      !swap_block_transfer_infos_->empty()) {
    auto& swap_blocks = *swap_block_transfer_infos_;
    if (FLAGS_enable_block_copy_kernel) {
      std::sort(swap_blocks.begin(),
                swap_blocks.end(),
                [](const BlockTransferInfo& a, const BlockTransferInfo& b) {
                  return a.src_block_id < b.src_block_id;
                });
      if (!swap_blocks.empty()) {
        std::vector<int32_t> src_indices, dst_indices, cum_sum;
        int32_t current_src = swap_blocks[0].src_block_id;
        src_indices.reserve(swap_blocks.size());
        dst_indices.reserve(swap_blocks.size());

        src_indices.push_back(swap_blocks[0].src_block_id);
        dst_indices.push_back(swap_blocks[0].dst_block_id);
        for (size_t i = 1; i < swap_blocks.size(); i++) {
          dst_indices.push_back(swap_blocks[i].dst_block_id);
          if (swap_blocks[i].src_block_id != current_src) {
            src_indices.push_back(swap_blocks[i].src_block_id);
            cum_sum.push_back(i);
            current_src = swap_blocks[i].src_block_id;
          }
        }
        cum_sum.push_back(swap_blocks.size());

        raw_forward_input.swap_blocks.clear();
        raw_forward_input.src_block_indices = std::move(src_indices);
        raw_forward_input.dst_block_indices = std::move(dst_indices);
        raw_forward_input.cum_sum = std::move(cum_sum);
      }
    } else {
      raw_forward_input.swap_blocks.insert(raw_forward_input.swap_blocks.end(),
                                           swap_blocks.begin(),
                                           swap_blocks.end());
    }
  }

  // Add multi-step specific data using existing RawForwardInput fields
  auto& multi_step_state = multi_step_state_;
  multi_step_state.total_steps = FLAGS_max_decode_rounds;

  // Set step-level decode metadata for multi-step processing
  raw_forward_input.beam_width = FLAGS_beam_width;
  raw_forward_input.total_round = multi_step_state.total_steps;

  // Set full_kv_shape if we have multi-step decode data
  if (!multi_step_state.decode_seq_lens.empty() && !sequences_.empty()) {
    // Set decode kv cache shape for step-level decode
    // Format: [batch_size * beam_width, n_kv_heads, step_rounds, head_dim]
    int64_t batch_size = static_cast<int64_t>(sequences_.size());
    int64_t step_rounds = static_cast<int64_t>(multi_step_state.total_steps);

    int64_t n_kv_heads =
        args_ ? args_->n_kv_heads().value_or(args_->n_heads()) : 0;
    int64_t head_dim = args_ ? args_->head_dim() : 0;
    // LOG(INFO) << "batch_size:" << batch_size;
    // LOG(INFO) << "FLAGS_max_token_per_req:" << FLAGS_max_token_per_req;
    // LOG(INFO) << "FLAGS_beam_width:" << FLAGS_beam_width;
    // LOG(INFO) << "FLAGS_max_decode_rounds:" << FLAGS_max_decode_rounds;
    raw_forward_input.full_kv_shape = {
        batch_size * FLAGS_max_token_per_req +
            batch_size * FLAGS_beam_width * (FLAGS_max_decode_rounds - 1),
        n_kv_heads,
        head_dim};
  }

  if (!multi_step_state.decode_seq_lens.empty()) {
    raw_forward_input.decode_seq_lens.insert(
        raw_forward_input.decode_seq_lens.end(),
        multi_step_state.decode_seq_lens.begin(),
        multi_step_state.decode_seq_lens.end());
    raw_forward_input.decode_q_seq_lens.insert(
        raw_forward_input.decode_q_seq_lens.end(),
        multi_step_state.decode_q_seq_lens.begin(),
        multi_step_state.decode_q_seq_lens.end());
  }

  if (!multi_step_state.decode_positions_vec.empty()) {
    raw_forward_input.decode_positions_vec.insert(
        raw_forward_input.decode_positions_vec.end(),
        multi_step_state.decode_positions_vec.begin(),
        multi_step_state.decode_positions_vec.end());
  }

  // append multi-step decode state into raw_forward_input
  if (!multi_step_state.decode_selected_token_idxes.empty()) {
    raw_forward_input.decode_selected_token_idxes.insert(
        raw_forward_input.decode_selected_token_idxes.end(),
        multi_step_state.decode_selected_token_idxes.begin(),
        multi_step_state.decode_selected_token_idxes.end());
    raw_forward_input.decode_sample_idxes.insert(
        raw_forward_input.decode_sample_idxes.end(),
        multi_step_state.decode_sample_idxes.begin(),
        multi_step_state.decode_sample_idxes.end());
    raw_forward_input.decode_unique_token_ids_vec.insert(
        raw_forward_input.decode_unique_token_ids_vec.end(),
        multi_step_state.decode_unique_token_ids_vec.begin(),
        multi_step_state.decode_unique_token_ids_vec.end());
    raw_forward_input.decode_unique_token_counts_vec.insert(
        raw_forward_input.decode_unique_token_counts_vec.end(),
        multi_step_state.decode_unique_token_counts_vec.begin(),
        multi_step_state.decode_unique_token_counts_vec.end());
    raw_forward_input.decode_unique_token_lens_vec.insert(
        raw_forward_input.decode_unique_token_lens_vec.end(),
        multi_step_state.decode_unique_token_lens_vec.begin(),
        multi_step_state.decode_unique_token_lens_vec.end());
    raw_forward_input.decode_sampling_params.insert(
        raw_forward_input.decode_sampling_params.end(),
        multi_step_state.decode_sampling_params.begin(),
        multi_step_state.decode_sampling_params.end());
  }
  raw_forward_input.batch_id = batch_id_;
  return raw_forward_input;
}

}  // namespace xllm
