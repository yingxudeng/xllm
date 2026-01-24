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

#include "cuda_graph_executor_impl.h"

#include <c10/core/Device.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <numeric>
#include <thread>

#include "core/common/global_flags.h"
#include "core/common/metrics.h"
#include "core/kernels/cuda/attention_runner.h"
#include "core/kernels/cuda/global_capture_instance.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "core/layers/cuda/flashinfer_planinfo.h"
#include "core/platform/device.h"
#include "core/platform/stream.h"
#include "core/util/utils.h"
#include "kernels/cuda/utils.h"

namespace xllm::runtime::cuda {

DEFINE_bool(force_graph_eager, false, "force_graph_eager");

// CudaGraphPersistentParam implementation
CudaGraphPersistentParam::CudaGraphPersistentParam(
    const ModelArgs& args,
    const torch::Device& device,
    const runtime::Options& options)
    : args_(args), device_(device), options_(options) {
  // Use max_tokens_per_batch for first dimension size
  const int64_t max_tokens_per_batch = FLAGS_max_tokens_per_batch;
  // num_sequences
  const int64_t max_seqs_per_batch =
      options.max_seqs_per_batch() * FLAGS_beam_width;
  auto tensor_options = torch::TensorOptions().device(device);

  const int64_t max_seq_len = FLAGS_max_seq_len_for_graph_mode > 0
                                  ? FLAGS_max_seq_len_for_graph_mode
                                  : args_.max_position_embeddings();

  // Create persistent tensors with max_tokens_per_batch as first dimension
  persistent_tokens_ = torch::zeros({max_tokens_per_batch},
                                    torch::dtype(torch::kInt).device(device));
  persistent_positions_ = torch::zeros(
      {max_tokens_per_batch}, torch::dtype(torch::kInt).device(device));
  persistent_new_cache_slots_ = torch::zeros(
      {max_tokens_per_batch}, torch::dtype(torch::kInt).device(device));

  // q_seq_lens is q_cu_seq_lens in GPU Model.
  // kv_seq_lens is kv_cu_seq_lens in GPU Model.
  q_seq_lens_ = torch::zeros({max_seqs_per_batch + 1},
                             torch::dtype(torch::kInt).device(device));
  kv_seq_lens_ = torch::zeros({max_seqs_per_batch + 1},
                              torch::dtype(torch::kInt).device(device));

  // Block table tensors with maximum possible size
  const auto block_size = options.block_size();
  const int64_t max_block_table_len =
      (max_seq_len + block_size - 1) / block_size + 1;
  persistent_block_tables_ =
      torch::zeros({max_seqs_per_batch, max_block_table_len},
                   torch::dtype(torch::kInt).device(device));

  // Output tensor for hidden states
  torch::Dtype dtype = util::parse_dtype(args.dtype(), device);
  if (args.dtype() == "float" || args.dtype() == "float32") {
    LOG(WARNING)
        << "Cuda graph executor init hidden_states compatible with float32 "
           "dtype: float32. This should not happen in production but for test.";
    dtype = torch::kFloat32;
  }
  hidden_states_ = torch::zeros({max_tokens_per_batch, args.hidden_size()},
                                torch::dtype(dtype).device(device));

  // FlashInfer decode mode parameters
  // paged_kv_indptr: shape [max_seqs_per_batch + 1]
  persistent_paged_kv_indptr_ = torch::zeros(
      {max_seqs_per_batch + 1}, torch::dtype(torch::kInt).device(device));

  // paged_kv_indices: maximum size based on max blocks
  // Estimate max blocks: max_seqs_per_batch * max_block_table_len
  const int64_t max_paged_kv_indices_size =
      max_seqs_per_batch * max_block_table_len;
  persistent_paged_kv_indices_ = torch::zeros(
      {max_paged_kv_indices_size}, torch::dtype(torch::kInt).device(device));

  // paged_kv_last_page_len: shape [max_seqs_per_batch]
  persistent_paged_kv_last_page_len_ = torch::zeros(
      {max_seqs_per_batch}, torch::dtype(torch::kInt).device(device));

  // For decode mode, each sequence has 1 token, so qo_indptr = [0, 1, 2, ...,
  // max_seqs_per_batch]
  persistent_decode_qo_indptr_ = torch::arange(
      0, max_seqs_per_batch + 1, torch::dtype(torch::kInt).device(device));
  // will be updated by q_cu_seq_lens, q_cu_seq_lens is the cumulative sum of
  // q_seq_lens
  persistent_chunked_prefill_qo_indptr_ = torch::zeros(
      {max_seqs_per_batch + 1}, torch::dtype(torch::kInt).device(device));

  // Pre-allocate two-stage decode cache tensors (stable pointers for CUDA
  // graph)
  const int64_t max_total_beam = FLAGS_max_seqs_per_batch * FLAGS_beam_width;
  const int64_t n_heads = args_.n_heads();
  const int64_t head_dim = args_.head_dim();
  auto fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto model_options = torch::TensorOptions().dtype(dtype).device(device);
  auto int32_options =
      torch::TensorOptions().dtype(torch::kInt32).device(device);

  // Output tensors (shape fixed, values computed per layer)
  persistent_two_decode_cache_.shared_lse =
      torch::zeros({max_total_beam, n_heads, 1}, fp32_options);
  persistent_two_decode_cache_.shared_o =
      torch::zeros({max_total_beam, n_heads, head_dim}, model_options);
  persistent_two_decode_cache_.unshared_lse =
      torch::zeros({max_total_beam, n_heads, 1}, fp32_options);
  persistent_two_decode_cache_.unshared_o =
      torch::zeros({max_total_beam, n_heads, head_dim}, model_options);

  // Fixed tensors (values updated per call)
  // q_cu_seq_lens_shared shape is [batch_size + 1], not (batch_size + 1) *
  // beam_width
  persistent_two_decode_cache_.q_cu_seq_lens_shared =
      torch::zeros({max_seqs_per_batch + 1}, int32_options);
  persistent_two_decode_cache_.paged_kv_indptr_expanded =
      torch::zeros({max_total_beam + 1}, int32_options);
  persistent_two_decode_cache_.paged_kv_indices_expanded =
      torch::zeros({max_total_beam * FLAGS_max_decode_rounds}, int32_options);
  persistent_two_decode_cache_.paged_kv_last_page_len_expanded =
      torch::zeros({max_total_beam}, int32_options);

  // Initialize unshared workspace buffers for two-stage decode
  // These buffers are independent from shared stage to avoid plan_info conflict
  if (FLAGS_enable_xattention_two_stage_decode) {
    unshared_float_workspace_buffer_ =
        torch::empty({FLAGS_flashinfer_workspace_buffer_size},
                     torch::dtype(torch::kUInt8).device(device));
    unshared_int_workspace_buffer_ = torch::empty(
        {8 * 1024 * 1024}, torch::dtype(torch::kUInt8).device(device));
    unshared_page_locked_int_workspace_buffer_ = torch::empty(
        {unshared_int_workspace_buffer_.size(0)},
        torch::dtype(torch::kUInt8).device(torch::kCPU).pinned_memory(true));
  }
}

std::optional<ModelInputParams> CudaGraphPersistentParam::update(
    const torch::Tensor& tokens,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& positions,
    const ModelInputParams& params,
    uint32_t padded_num_tokens,
    bool return_capture_params) {
  std::optional<ModelInputParams> params_for_capture;
  if (return_capture_params) {
    CHECK_GT(padded_num_tokens, 0)
        << "padded_num_tokens must be > 0 when return_capture_params is true";
    params_for_capture = std::make_optional<ModelInputParams>(params);
  }
  // Build attn_metadata with original model_input_params. So we can set actual
  // batch size in plan_info.
  std::shared_ptr<layer::AttentionMetadata> attn_metadata;
  attn_metadata = std::make_shared<layer::AttentionMetadata>(
      layer::AttentionMetadataBuilder::build(params));
  CHECK(attn_metadata) << "attn_metadata should not be null";
  attn_metadata->enable_cuda_graph = true;

  const uint32_t actual_num_tokens = tokens.size(0);
  const int64_t actual_batch_size = params.paged_kv_last_page_len.numel();
  const int64_t request_batch_size =
      params.kv_seq_lens.defined() ? (params.kv_seq_lens.numel() - 1) : 0;

  // Copy data from input parameters to persistent graph tensors
  VLOG(kGraphExecutorLogVerboseLevel)
      << "copy_ tokens: src shape=" << tokens.sizes() << ", dst slice shape=["
      << actual_num_tokens << "]";
  persistent_tokens_.slice(/*dim=*/0, /*start=*/0, /*end=*/actual_num_tokens)
      .copy_(tokens, /*non_blocking=*/true);

  // Zero out padding region for tokens to avoid stale data
  // This is needed for both capture and replay when using padded tensors
  if (padded_num_tokens > actual_num_tokens) {
    VLOG(kGraphExecutorLogVerboseLevel)
        << "fill_ tokens padding: [" << actual_num_tokens << ", "
        << padded_num_tokens << "] with 0";
    persistent_tokens_
        .slice(
            /*dim=*/0, /*start=*/actual_num_tokens, /*end=*/padded_num_tokens)
        .fill_(0);
  }

  VLOG(kGraphExecutorLogVerboseLevel)
      << "copy_ positions: src shape=" << positions.sizes()
      << ", dst slice shape=[" << actual_num_tokens << "]";
  persistent_positions_.slice(/*dim=*/0, /*start=*/0, /*end=*/actual_num_tokens)
      .copy_(positions, /*non_blocking=*/true);

  // Zero out padding region for positions to avoid index out-of-bounds errors
  // This is critical: stale position values can exceed vocabulary size
  // Needed for both capture and replay when using padded tensors
  if (padded_num_tokens > actual_num_tokens) {
    VLOG(kGraphExecutorLogVerboseLevel)
        << "fill_ positions padding: [" << actual_num_tokens << ", "
        << padded_num_tokens << "] with 0";
    persistent_positions_
        .slice(
            /*dim=*/0, /*start=*/actual_num_tokens, /*end=*/padded_num_tokens)
        .fill_(0);
  }

  // Validate tokens and positions range only when verbose logging is enabled
  // to avoid CPU synchronization overhead in production
  if (VLOG_IS_ON(10) && padded_num_tokens > 0) {
    // Validate tokens range to detect out-of-bounds before embedding lookup
    auto tokens_cpu = persistent_tokens_.slice(0, 0, padded_num_tokens).cpu();
    auto tokens_min = tokens_cpu.min().item<int32_t>();
    auto tokens_max = tokens_cpu.max().item<int32_t>();
    auto vocab_size = args_.vocab_size();

    if (tokens_min < 0 || tokens_max >= vocab_size) {
      LOG(ERROR) << "Token index out of bounds detected! "
                 << "min_token=" << tokens_min << ", max_token=" << tokens_max
                 << ", valid_range=[0, " << vocab_size - 1 << "]"
                 << ", actual_num_tokens=" << actual_num_tokens
                 << ", padded_num_tokens=" << padded_num_tokens;

      // Clamp to valid range [0, vocab_size - 1]
      persistent_tokens_.slice(0, 0, padded_num_tokens)
          .clamp_(0, vocab_size - 1);

      LOG(WARNING) << "Clamped tokens to valid range [0, " << vocab_size - 1
                   << "]";
    }

    // Validate positions range similarly
    auto positions_cpu =
        persistent_positions_.slice(0, 0, padded_num_tokens).cpu();
    auto positions_min = positions_cpu.min().item<int32_t>();
    auto positions_max = positions_cpu.max().item<int32_t>();
    auto max_position = args_.max_position_embeddings();

    if (positions_min < 0 || positions_max >= max_position) {
      LOG(ERROR) << "Position index out of bounds detected! "
                 << "min_position=" << positions_min
                 << ", max_position=" << positions_max << ", valid_range=[0, "
                 << max_position - 1 << "]"
                 << ", actual_num_tokens=" << actual_num_tokens
                 << ", padded_num_tokens=" << padded_num_tokens;

      // Clamp to valid range [0, max_position - 1]
      persistent_positions_.slice(0, 0, padded_num_tokens)
          .clamp_(0, max_position - 1);

      LOG(WARNING) << "Clamped positions to valid range [0, "
                   << max_position - 1 << "]";
    }
  }

  // q_seq_lens is q_cu_seq_lens in GPU Model.
  // kv_seq_lens is kv_cu_seq_lens in GPU Model.
  // VLOG(kGraphExecutorLogVerboseLevel)
  //     << "copy_ q_seq_lens: src shape=" << params.q_seq_lens.sizes()
  //     << ", dst slice shape=[" << actual_batch_size + 1 << "]";
  // q_seq_lens_.slice(/*dim=*/0, /*start=*/0, /*end=*/actual_batch_size + 1)
  //     .copy_(params.q_seq_lens, /*non_blocking=*/true);
  // VLOG(kGraphExecutorLogVerboseLevel)
  //     << "copy_ kv_seq_lens: src shape=" << params.kv_seq_lens.sizes()
  //     << ", dst slice shape=[" << actual_batch_size + 1 << "]";
  // kv_seq_lens_.slice(/*dim=*/0, /*start=*/0, /*end=*/actual_batch_size + 1)
  //     .copy_(params.kv_seq_lens, /*non_blocking=*/true);
  if (params.new_cache_slots.numel() > 0) {
    VLOG(kGraphExecutorLogVerboseLevel)
        << "copy_ new_cache_slots: src shape=" << params.new_cache_slots.sizes()
        << ", dst slice shape=[" << actual_num_tokens << "]";
    persistent_new_cache_slots_
        .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_num_tokens)
        .copy_(params.new_cache_slots, /*non_blocking=*/true);
  }

  // Zero out padding region for new_cache_slots to avoid stale data
  // Needed for both capture and replay when using padded tensors
  if (padded_num_tokens > actual_num_tokens) {
    VLOG(kGraphExecutorLogVerboseLevel)
        << "fill_ new_cache_slots padding: [" << actual_num_tokens << ", "
        << padded_num_tokens << "] with 0";
    persistent_new_cache_slots_
        .slice(
            /*dim=*/0, /*start=*/actual_num_tokens, /*end=*/padded_num_tokens)
        .fill_(0);
  }

  // Persist q/kv cu_seq_lens so CUDA graph replay can see updated values.
  // NOTE: In step-level (PureDevice) decode, the number of tokens can be
  // batch_size * beam_width, while q/kv cu_seq_lens are per-request (size
  // request_batch_size + 1).
  if (params.q_seq_lens.defined() && params.q_seq_lens.numel() > 0 &&
      request_batch_size > 0) {
    q_seq_lens_
        .slice(/*dim=*/0,
               /*start=*/0,
               /*end=*/request_batch_size + 1)
        .copy_(params.q_seq_lens, /*non_blocking=*/true);
    attn_metadata->q_cu_seq_lens = q_seq_lens_.slice(
        /*dim=*/0, /*start=*/0, /*end=*/request_batch_size + 1);
  }
  if (params.kv_seq_lens.defined() && params.kv_seq_lens.numel() > 0 &&
      request_batch_size > 0) {
    kv_seq_lens_
        .slice(/*dim=*/0,
               /*start=*/0,
               /*end=*/request_batch_size + 1)
        .copy_(params.kv_seq_lens, /*non_blocking=*/true);
    attn_metadata->kv_cu_seq_lens = kv_seq_lens_.slice(
        /*dim=*/0, /*start=*/0, /*end=*/request_batch_size + 1);
  }

  // Copy block table data
  if (params.block_tables.numel() > 0) {
    const int64_t actual_block_table_len = params.block_tables.size(1);
    auto slice_persistent_block_tables =
        persistent_block_tables_
            .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_batch_size)
            .slice(/*dim=*/1, /*start=*/0, /*end=*/actual_block_table_len);
    VLOG(kGraphExecutorLogVerboseLevel)
        << "copy_ block_tables: src shape=" << params.block_tables.sizes()
        << ", dst slice shape=" << slice_persistent_block_tables.sizes();
    slice_persistent_block_tables.copy_(params.block_tables,
                                        /*non_blocking=*/true);
  }

  // Update persistent embedding from input_embedding if available
  const auto& embedding = params.input_embedding;
  if (embedding.defined()) {
    const int64_t embedding_tokens = embedding.size(0);

    // Initialize persistent_embedding_ if needed and not already initialized
    if (persistent_embedding_.numel() == 0) {
      const int64_t max_tokens_per_batch = FLAGS_max_tokens_per_batch;
      const int64_t embedding_dim = embedding.size(1);
      torch::Dtype dtype = util::parse_dtype(args_.dtype(), device_);
      persistent_embedding_ =
          torch::zeros({max_tokens_per_batch, embedding_dim},
                       torch::dtype(dtype).device(device_));
    }

    // Copy embedding data to persistent buffer
    VLOG(kGraphExecutorLogVerboseLevel)
        << "copy_ embedding: src shape=" << embedding.sizes()
        << ", dst slice shape=[" << embedding_tokens << ", "
        << embedding.size(1) << "]";
    persistent_embedding_
        .slice(/*dim=*/0, /*start=*/0, /*end=*/embedding_tokens)
        .copy_(embedding, /*non_blocking=*/true);
  }

  // FlashInfer decode parameters update (if present)
  CHECK(params.paged_kv_indptr.defined())
      << "paged_kv_indptr should not be null";
  VLOG(kGraphExecutorLogVerboseLevel)
      << "copy_ paged_kv_indptr: src shape=" << params.paged_kv_indptr.sizes()
      << ", dst slice shape=[" << (actual_batch_size + 1) << "]";
  if (VLOG_IS_ON(kGraphExecutorLogVerboseLevel)) {
    torch::Tensor paged_kv_indptr_cpu = params.paged_kv_indptr.to(torch::kCPU);
    VLOG(kGraphExecutorLogVerboseLevel)
        << "copy_ paged_kv_indptr: src values=" << paged_kv_indptr_cpu;
  }
  persistent_paged_kv_indptr_
      .slice(/*dim=*/0,
             /*start=*/0,
             /*end=*/actual_batch_size + 1)
      .copy_(params.paged_kv_indptr, /*non_blocking=*/true);

  CHECK(params.paged_kv_indices.defined())
      << "paged_kv_indices should not be null";
  const int64_t actual_indices_size = params.paged_kv_indices.size(0);
  VLOG(kGraphExecutorLogVerboseLevel)
      << "copy_ paged_kv_indices: src shape=" << params.paged_kv_indices.sizes()
      << ", dst slice shape=[" << actual_indices_size << "]";
  persistent_paged_kv_indices_
      .slice(/*dim=*/0,
             /*start=*/0,
             /*end=*/actual_indices_size)
      .copy_(params.paged_kv_indices, /*non_blocking=*/true);
  CHECK(params.paged_kv_last_page_len.defined())
      << "paged_kv_last_page_len should not be null";
  VLOG(kGraphExecutorLogVerboseLevel)
      << "copy_ paged_kv_last_page_len: src shape="
      << params.paged_kv_last_page_len.sizes() << ", dst slice shape=["
      << actual_batch_size << "]";
  persistent_paged_kv_last_page_len_
      .slice(/*dim=*/0,
             /*start=*/0,
             /*end=*/actual_batch_size)
      .copy_(params.paged_kv_last_page_len, /*non_blocking=*/true);

  // Convert cumulative lengths to individual sequence lengths using torch::diff
  // This matches the behavior in attention_metadata_builder.cpp for decode mode
  attn_metadata->kv_seq_lens =
      torch::diff(kv_seq_lens(/*actual_batch_size=*/actual_batch_size + 1));
  // Set FlashInfer decode parameters (always update, not just for capture)
  // This ensures attn_metadata points to updated persistent buffers for
  // plan_info calculation
  attn_metadata->paged_kv_indptr =
      persistent_paged_kv_indptr(actual_batch_size);
  attn_metadata->paged_kv_indices =
      persistent_paged_kv_indices(actual_indices_size);
  attn_metadata->paged_kv_last_page_len =
      persistent_paged_kv_last_page_len(actual_batch_size);
  // qo_indptr is q_cu_seq_lens in GPU Model.
  attn_metadata->qo_indptr = persistent_decode_qo_indptr(actual_batch_size);

  const bool enable_two_stage = FLAGS_enable_xattention_two_stage_decode &&
                                (params.batch_forward_type.is_decode());

  if (enable_two_stage) {
    const int64_t total_beam = request_batch_size * FLAGS_beam_width;
    CHECK_GT(request_batch_size, 0)
        << "request_batch_size must be > 0 for two-stage xattention";
    CHECK_EQ(total_beam % request_batch_size, 0)
        << "total_beam must be divisible by request_batch_size";
    const int64_t beam_width = FLAGS_beam_width;

    // Get attention parameters from ModelArgs
    const int64_t n_heads = args_.n_heads();
    const int32_t head_dim = args_.head_dim();

    // Get two-stage decode cache with sliced tensors from persistent buffers
    layer::TwoStageDecodeCache cache = get_two_stage_decode_cache(
        total_beam, request_batch_size, beam_width, n_heads, head_dim);

    // Update dynamic values in persistent buffers
    // q_cu_seq_lens_shared: generate values using arange and copy to slice
    // Shape is [request_batch_size + 1]: [0, beam_width, 2*beam_width, ...,
    // request_batch_size*beam_width]
    const int64_t q_cu_seq_lens_size = request_batch_size + 1;
    auto q_cu_seq_lens_values = torch::arange(
        0,
        (request_batch_size + 1) * beam_width,
        beam_width,
        torch::TensorOptions().dtype(torch::kInt32).device(tokens.device()));
    cache.q_cu_seq_lens_shared.copy_(q_cu_seq_lens_values,
                                     /*non_blocking=*/true);

    // paged_kv_indptr_expanded: generate values using arange and copy to slice
    auto paged_kv_indptr_values = torch::arange(
        total_beam + 1,
        torch::TensorOptions().dtype(torch::kInt32).device(tokens.device()));
    cache.paged_kv_indptr_expanded.copy_(paged_kv_indptr_values,
                                         /*non_blocking=*/true);

    // paged_kv_indices_expanded: generate values using arange and copy to slice
    auto paged_kv_indices_values = torch::arange(
        total_beam,
        torch::TensorOptions().dtype(torch::kInt32).device(tokens.device()));
    cache.paged_kv_indices_expanded.copy_(paged_kv_indices_values,
                                          /*non_blocking=*/true);

    // paged_kv_last_page_len_expanded: fill with current_round + 1
    // This value changes per round, so must be updated on every call (capture
    // and replay)
    int32_t current_round_value =
        params.current_round.defined() && params.current_round.numel() > 0
            ? params.current_round.item<int32_t>()
            : 0;
    cache.paged_kv_last_page_len_expanded.fill_(current_round_value + 1);

    attn_metadata->two_stage_decode_cache = cache;
  }

  // Update plan_info only before capture. Replay does not invoke model forward,
  // so updating plan_info here has no effect on graph replay.
  // Get attention parameters from ModelArgs
  const int32_t head_dim = args_.head_dim();
  const int64_t n_heads = args_.n_heads();
  const int64_t n_kv_heads = args_.n_kv_heads().value_or(n_heads);
  const int64_t block_size = options_.block_size();

  // Get sliding_window from ModelArgs (default to -1 if not available)
  // Note: sliding_window in ModelArgs is the actual window size, but in
  // attention it's used as window_size_left which is typically sliding_window
  // - 1. This matches the behavior in attention.cpp where sliding_window_ is
  // initialized as sliding_window - 1 regardless of the value.
  int32_t sliding_window = args_.sliding_window();
  sliding_window =
      sliding_window - 1;  // Convert to window_size_left (always subtract 1)

  // Get dtype from k_cache
  const auto dtype = k_cache.scalar_type();

  if (enable_two_stage) {
    // Get cache that was already updated above
    const layer::TwoStageDecodeCache& cache =
        attn_metadata->two_stage_decode_cache.value();

    // 1) shared stage (prefill, causal) plan
    layer::AttentionMetadata shared_attn_meta = *attn_metadata;
    shared_attn_meta.q_cu_seq_lens = cache.q_cu_seq_lens_shared;
    attn_metadata->plan_info->layer_id = 0;
    layer::flashinfer::update_plan_info(
        attn_metadata->plan_info,
        xllm::kernel::cuda::determine_attention_backend(
            /*pos_encoding_mode=*/0,
            /*use_fp16_qk_reduction=*/false,
            /*use_custom_mask=*/false),
        shared_attn_meta,
        dtype,
        dtype,
        dtype,
        head_dim,
        head_dim,
        static_cast<int32_t>(n_heads),
        static_cast<int32_t>(n_kv_heads),
        /*block_size*/ 1,
        -1,
        /*enable_cuda_graph*/ true,
        /*causal*/ false,
        /*use_tensor_core*/ true);

    // 2) unshared stage (decode, non-tensor-core) plan
    layer::AttentionMetadata unshared_attn_meta = *attn_metadata;
    unshared_attn_meta.plan_info = attn_metadata->unshared_plan_info;
    unshared_attn_meta.paged_kv_indptr = cache.paged_kv_indptr_expanded;
    unshared_attn_meta.paged_kv_indices = cache.paged_kv_indices_expanded;
    unshared_attn_meta.paged_kv_last_page_len =
        cache.paged_kv_last_page_len_expanded;
    unshared_attn_meta.use_tensor_core = false;

    // Use independent workspace buffer for unshared stage to avoid conflict
    // with shared stage plan_info during CUDA graph capture
    unshared_attn_meta.float_workspace_buffer =
        unshared_float_workspace_buffer_;
    unshared_attn_meta.int_workspace_buffer = unshared_int_workspace_buffer_;
    unshared_attn_meta.page_locked_int_workspace_buffer =
        unshared_page_locked_int_workspace_buffer_;

    const int64_t max_decode_step =
        params.unshared_k_caches.empty()
            ? 0
            : static_cast<int64_t>(params.unshared_k_caches[0].size(2));
    CHECK_GT(max_decode_step, 0)
        << "max_decode_step must be > 0 for two-stage unshared plan";

    attn_metadata->unshared_plan_info->layer_id = 0;
    layer::flashinfer::update_plan_info(attn_metadata->unshared_plan_info,
                                        /*backend*/ "fa3",
                                        unshared_attn_meta,
                                        dtype,
                                        dtype,
                                        dtype,
                                        head_dim,
                                        head_dim,
                                        static_cast<int32_t>(n_heads),
                                        static_cast<int32_t>(n_kv_heads),
                                        static_cast<int32_t>(max_decode_step),
                                        sliding_window,
                                        /*enable_cuda_graph*/ true,
                                        /*causal*/ false,
                                        /*use_tensor_core*/ false);
  } else {
    // For piecewise capture (prefill), causal should be true
    // For normal capture (decode), causal should be false
    const bool causal =
        attn_metadata->is_prefill || attn_metadata->is_chunked_prefill;

    // Determine backend based on causal mode
    const std::string backend =
        causal ? xllm::kernel::cuda::determine_attention_backend(
                     /*pos_encoding_mode=*/0,
                     /*use_fp16_qk_reduction=*/false,
                     /*use_custom_mask=*/false)
               : "fa2";

    // Update plan_info
    // Note: plan_info is only updated at layer 0, so we set layer_id to 0
    attn_metadata->plan_info->layer_id = 0;

    VLOG(kGraphExecutorLogVerboseLevel)
        << "CudaGraphPersistentParam::update() calling update_plan_info: "
        << "is_prefill=" << attn_metadata->is_prefill
        << ", is_chunked_prefill=" << attn_metadata->is_chunked_prefill
        << ", causal=" << causal << ", backend=" << backend
        << ", enable_cuda_graph=" << attn_metadata->enable_cuda_graph;

    layer::flashinfer::update_plan_info(
        attn_metadata->plan_info,
        backend,
        *attn_metadata,
        dtype,                             // query_dtype
        dtype,                             // key_dtype
        dtype,                             // output_dtype
        head_dim,                          // head_dim_qk
        head_dim,                          // head_dim_vo
        static_cast<int32_t>(n_heads),     // num_qo_heads
        static_cast<int32_t>(n_kv_heads),  // num_kv_heads
        static_cast<int32_t>(block_size),  // block_size
        sliding_window,                    // window_size_left
        true,                              // enable_cuda_graph
        causal,                            // causal
        attn_metadata->use_tensor_core);   // use_tensor_core
  }

  // Return ModelInputParams with persistent buffer references if requested
  if (return_capture_params) {
    CHECK_GT(padded_num_tokens, 0)
        << "padded_num_tokens must be > 0 when return_capture_params is true";
    // Set persistent embedding if available
    if (params.input_embedding.defined()) {
      params_for_capture->input_embedding =
          persistent_embedding(padded_num_tokens);
    }
    params_for_capture->attn_metadata = attn_metadata;
    return params_for_capture;
  }

  return std::nullopt;
}

layer::TwoStageDecodeCache CudaGraphPersistentParam::get_two_stage_decode_cache(
    int64_t total_beam,
    int64_t request_batch_size,
    int64_t beam_width,
    int64_t n_heads,
    int64_t head_dim) const {
  layer::TwoStageDecodeCache cache;

  // Validate bounds
  const int64_t max_total_beam = FLAGS_max_tokens_per_batch * FLAGS_beam_width;
  CHECK_LE(total_beam, max_total_beam)
      << "total_beam (" << total_beam << ") exceeds max_total_beam ("
      << max_total_beam << ")";
  const int64_t max_seqs_per_batch = options_.max_seqs_per_batch();
  CHECK_LE(request_batch_size, max_seqs_per_batch)
      << "request_batch_size (" << request_batch_size
      << ") exceeds max_seqs_per_batch (" << max_seqs_per_batch << ")";
  CHECK_LE(beam_width, FLAGS_beam_width)
      << "beam_width (" << beam_width << ") exceeds FLAGS_beam_width ("
      << FLAGS_beam_width << ")";

  // Get sliced tensors from persistent buffers
  cache.shared_lse = persistent_two_decode_cache_.shared_lse.slice(
      /*dim=*/0, /*start=*/0, /*end=*/total_beam);
  cache.shared_o = persistent_two_decode_cache_.shared_o.slice(
      /*dim=*/0, /*start=*/0, /*end=*/total_beam);
  cache.unshared_lse = persistent_two_decode_cache_.unshared_lse.slice(
      /*dim=*/0, /*start=*/0, /*end=*/total_beam);
  cache.unshared_o = persistent_two_decode_cache_.unshared_o.slice(
      /*dim=*/0, /*start=*/0, /*end=*/total_beam);

  // q_cu_seq_lens_shared: slice from persistent buffer
  // Size is [request_batch_size + 1], not (request_batch_size + 1) * beam_width
  const int64_t q_cu_seq_lens_size = request_batch_size + 1;
  cache.q_cu_seq_lens_shared =
      persistent_two_decode_cache_.q_cu_seq_lens_shared.slice(
          /*dim=*/0, /*start=*/0, /*end=*/q_cu_seq_lens_size);

  cache.paged_kv_indptr_expanded =
      persistent_two_decode_cache_.paged_kv_indptr_expanded.slice(
          /*dim=*/0, /*start=*/0, /*end=*/total_beam + 1);
  cache.paged_kv_indices_expanded =
      persistent_two_decode_cache_.paged_kv_indices_expanded.slice(
          /*dim=*/0, /*start=*/0, /*end=*/total_beam);
  cache.paged_kv_last_page_len_expanded =
      persistent_two_decode_cache_.paged_kv_last_page_len_expanded.slice(
          /*dim=*/0, /*start=*/0, /*end=*/total_beam);

  // Set cached parameters
  cache.cached_batch_size = static_cast<int32_t>(request_batch_size);
  cache.cached_beam_size = static_cast<int32_t>(beam_width);
  cache.cached_num_heads = static_cast<int32_t>(n_heads);
  cache.cached_head_size = head_dim;

  // Set unshared workspace buffers for CUDA graph mode
  cache.unshared_float_workspace_buffer = unshared_float_workspace_buffer_;
  cache.unshared_int_workspace_buffer = unshared_int_workspace_buffer_;
  cache.unshared_page_locked_int_workspace_buffer =
      unshared_page_locked_int_workspace_buffer_;

  return cache;
}

// CudaGraph implementation
bool CudaGraph::capture(CausalLM* model,
                        const ModelArgs& args,
                        const runtime::Options& options,
                        const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        const ModelInputParams& params,
                        std::vector<KVCache>& kv_cache,
                        uint32_t bucket_num_tokens,
                        const at::cuda::MempoolId_t& pool) {
  padded_num_tokens_ = bucket_num_tokens;
  const uint32_t actual_num_tokens = tokens.size(0);
  CHECK_GE(padded_num_tokens_, actual_num_tokens)
      << "bucket_num_tokens >= actual_num_tokens";

  // Update persistent parameters with input data before capture
  const torch::Tensor& k_cache = kv_cache[0].get_k_cache();
  const torch::Tensor& v_cache = kv_cache[0].get_v_cache();
  auto graph_params_opt =
      persistent_param_.update(tokens,
                               k_cache,
                               v_cache,
                               positions,
                               params,
                               padded_num_tokens_,
                               /*return_capture_params=*/true);

  // Use the returned ModelInputParams for graph capture
  CHECK(graph_params_opt.has_value())
      << "update() should return ModelInputParams when "
         "return_capture_params=true";
  // Synchronize to ensure all data is copied to graph persistent buffers
  c10::cuda::getCurrentCUDAStream(device_index_).synchronize();

  LOG(INFO) << "CUDA graph capture begin, bucket_num_tokens: "
            << bucket_num_tokens << ", actual_num_tokens: " << actual_num_tokens
            << ", is_piecewise: " << is_piecewise_;

  // Use capture stream for graph capture (managed by CudaGraphExecutorImpl)
  // Use optional CUDAStreamGuard for RAII-based stream management
  std::optional<c10::cuda::CUDAStreamGuard> stream_guard;

  // Check if current stream is default stream, if so switch to capture stream
  if (c10::cuda::getCurrentCUDAStream(device_index_) ==
      c10::cuda::getDefaultCUDAStream(device_index_)) {
    c10::cuda::getCurrentCUDAStream(device_index_).synchronize();
    capture_stream_.synchronize();
    stream_guard.emplace(capture_stream_);
  }

  if (is_piecewise_) {
    // Piecewise capture mode (for prefill)
    // Warmup: execute forward once without capture to initialize cuBLAS handles
    // and other CUDA resources. This is necessary because these resources
    // cannot be created during CUDA graph capture mode.
    model->forward(persistent_param_.persistent_tokens(padded_num_tokens_),
                   persistent_param_.persistent_positions(padded_num_tokens_),
                   kv_cache,
                   graph_params_opt.value());

    // Begin piecewise capture via GlobalCaptureInstance
    GlobalCaptureInstance::get_instance().begin_capture(pool);

    // Execute forward pass - attention operations will be captured separately
    auto forward_result = model->forward(
        persistent_param_.persistent_tokens(padded_num_tokens_),
        persistent_param_.persistent_positions(padded_num_tokens_),
        kv_cache,
        graph_params_opt.value());

    // Store result in persistent buffer
    persistent_param_.set_hidden_states(forward_result);
    VLOG(kGraphExecutorLogVerboseLevel)
        << "Piecewise capture forward_result shape: " << forward_result.sizes();

    // End capture and get piecewise graphs
    auto piecewise_graphs = GlobalCaptureInstance::get_instance().end_capture();

    if (!piecewise_graphs || piecewise_graphs->empty()) {
      LOG(WARNING) << "Failed to capture piecewise graph: no graphs captured";
      return false;
    }

    // Move piecewise graphs to member
    piecewise_graph_ = std::move(*piecewise_graphs);

    LOG(INFO) << "Piecewise graph capture end, bucket_num_tokens: "
              << bucket_num_tokens
              << ", num_graphs: " << piecewise_graph_.size()
              << ", num_runners: " << piecewise_graph_.num_runners();
  } else {
    // Normal capture mode (for decode)
    // Begin graph capture (capture_mode defaults to
    // cudaStreamCaptureModeGlobal)
    if (!FLAGS_force_graph_eager) {
      // graph_.capture_begin(pool);
      graph_.capture_begin(pool, cudaStreamCaptureModeThreadLocal);
    }

    // Execute forward pass - CUDA graph will capture this
    auto forward_result = model->forward(
        persistent_param_.persistent_tokens(padded_num_tokens_),
        persistent_param_.persistent_positions(padded_num_tokens_),
        kv_cache,
        graph_params_opt.value());

    // Store result in persistent buffer
    persistent_param_.set_hidden_states(forward_result);

    // End graph capture
    if (!FLAGS_force_graph_eager) {
      graph_.capture_end();
    }
  }

  // Stream guard will automatically restore stream when going out of scope
  stream_guard.reset();

  if (!is_piecewise_ && FLAGS_force_graph_eager) {
    // capture failed. next time will enter this function again.
    return false;
  }

  if (is_piecewise_) {
    // replay piecewise graph
    CHECK(graph_params_opt->attn_metadata)
        << "attn_metadata is required for piecewise";
    CHECK(graph_params_opt->attn_metadata->plan_info)
        << "plan_info is required for piecewise";

    ::xllm::kernel::cuda::AttentionReplayParams replay_params;
    replay_params.actual_num_tokens = actual_num_tokens;
    replay_params.plan_info =
        graph_params_opt->attn_metadata->plan_info->plan_info;
    replay_params.q_cu_seq_lens =
        graph_params_opt->attn_metadata->q_cu_seq_lens;
    replay_params.kv_cu_seq_lens =
        graph_params_opt->attn_metadata->kv_cu_seq_lens;
    replay_params.is_causal = graph_params_opt->attn_metadata->is_causal;

    piecewise_graph_.replay(replay_params);
  } else {
    graph_.replay();
  }

  LOG(INFO) << "CUDA graph capture end, bucket_num_tokens: "
            << bucket_num_tokens;
  return true;
}

torch::Tensor CudaGraph::replay(const torch::Tensor& tokens,
                                const torch::Tensor& positions,
                                std::vector<KVCache>& kv_cache,
                                const ModelInputParams& params) {
  const uint32_t actual_num_tokens = tokens.size(0);
  CHECK_LE(actual_num_tokens, padded_num_tokens_)
      << "num_tokens mismatch: expected <= " << padded_num_tokens_ << ", got "
      << actual_num_tokens;

  // Update persistent parameters with new input data
  // This updates attn_metadata including plan_info via update_plan_info
  const torch::Tensor& k_cache = kv_cache[0].get_k_cache();
  const torch::Tensor& v_cache = kv_cache[0].get_v_cache();

  if (is_piecewise_) {
    // Piecewise replay mode (for prefill)
    // Need to get updated params with attn_metadata for attention replay
    auto updated_params_opt =
        persistent_param_.update(tokens,
                                 k_cache,
                                 v_cache,
                                 positions,
                                 params,
                                 padded_num_tokens_,
                                 /*return_capture_params=*/true);
    CHECK(updated_params_opt.has_value())
        << "update() should return ModelInputParams for piecewise replay";

    const auto& updated_params = updated_params_opt.value();
    CHECK(piecewise_graph_.num_runners() > 0)
        << "Piecewise graph must have attention runners";
    CHECK(updated_params.attn_metadata)
        << "attn_metadata is required for piecewise replay";
    CHECK(updated_params.attn_metadata->plan_info)
        << "plan_info is required for piecewise replay";

    // Build AttentionReplayParams from updated attn_metadata
    ::xllm::kernel::cuda::AttentionReplayParams replay_params;
    replay_params.actual_num_tokens = actual_num_tokens;
    replay_params.plan_info =
        updated_params.attn_metadata->plan_info->plan_info;
    replay_params.q_cu_seq_lens = updated_params.attn_metadata->q_cu_seq_lens;
    replay_params.kv_cu_seq_lens = updated_params.attn_metadata->kv_cu_seq_lens;
    replay_params.is_causal = updated_params.attn_metadata->is_causal;

    // Replay piecewise graphs and attention runners
    piecewise_graph_.replay(replay_params);
  } else {
    // Normal replay mode (for decode)
    persistent_param_.update(tokens,
                             k_cache,
                             v_cache,
                             positions,
                             params,
                             padded_num_tokens_,
                             /*return_capture_params=*/false);
    graph_.replay();
  }

  // Return only the actual num_tokens portion of hidden states
  return get_hidden_states(actual_num_tokens);
}

// CudaGraphExecutorImpl implementation
CudaGraphExecutorImpl::CudaGraphExecutorImpl(CausalLM* model,
                                             const ModelArgs& args,
                                             const torch::Device& device,
                                             const runtime::Options& options)
    : model_(model),
      args_(args),
      device_(device),
      options_(options),
      enable_prefill_piecewise_graph_(FLAGS_enable_prefill_piecewise_graph) {
  // Create single persistent parameter object shared by all CudaGraph instances
  persistent_param_ =
      std::make_unique<CudaGraphPersistentParam>(args_, device_, options_);
}

// Static method to get graph memory pool for current thread
// Each thread gets its own graph memory pool, similar to
// GlobalCaptureInstance::get_instance()
at::cuda::MempoolId_t CudaGraphExecutorImpl::get_mem_pool() {
  // Use thread_local to ensure each thread has its own graph pool
  // This follows the same pattern as GlobalCaptureInstance::get_instance()
  // but provides per-thread instances instead of a global singleton
  thread_local at::cuda::MempoolId_t thread_graph_pool =
      at::cuda::graph_pool_handle();

  // Thread-local counter to log initialization only once per thread
  thread_local bool initialized = false;
  if (!initialized) {
    LOG(INFO) << "Initialized graph_pool for thread: "
              << std::this_thread::get_id();
    initialized = true;
  }

  return thread_graph_pool;
}

// Static method to get CUDA capture stream for current thread
// Each thread gets its own high-priority capture stream
c10::cuda::CUDAStream CudaGraphExecutorImpl::get_capture_stream(
    c10::DeviceIndex device_index) {
  // Use thread_local to ensure each thread has its own capture stream
  // This is required because CUDA graphs must be captured on a non-default
  // stream. We use high-priority streams for better performance.
  thread_local c10::cuda::CUDAStream thread_capture_stream =
      c10::cuda::getStreamFromPool(/*isHighPriority=*/true, device_index);

  // Thread-local counter to log initialization only once per thread
  thread_local bool initialized = false;
  if (!initialized) {
    LOG(INFO) << "Initialized capture_stream for thread: "
              << std::this_thread::get_id()
              << ", stream: " << thread_capture_stream
              << ", device_index: " << device_index;
    initialized = true;
  }

  return thread_capture_stream;
}

ForwardInput CudaGraphExecutorImpl::prepare_inputs(Batch& batch) {
  // Prepare inputs for workers
  return batch.prepare_forward_input(options_.num_decoding_tokens(), 0, args_);
}

torch::Tensor CudaGraphExecutorImpl::run(const torch::Tensor& tokens,
                                         const torch::Tensor& positions,
                                         std::vector<KVCache>& kv_caches,
                                         const ModelInputParams& params) {
  const bool is_prefill = params.batch_forward_type.is_prefill();
  const bool is_decode = params.batch_forward_type.is_decode();

  // Get actual num_tokens from tokens shape
  const uint32_t n_tokens = tokens.size(/*dim=*/0);
  const uint32_t bucket_num_tokens =
      get_bucket_num_tokens(n_tokens, is_prefill);

  // Prefill phase with piecewise graph
  if (is_prefill && enable_prefill_piecewise_graph_) {
    // Check if token count is within limit
    const bool tokens_supported =
        FLAGS_max_tokens_for_graph_mode_prefill == 0 ||
        n_tokens <= FLAGS_max_tokens_for_graph_mode_prefill;

    if (!tokens_supported) {
      VLOG(kGraphExecutorLogVerboseLevel)
          << "Token count " << n_tokens
          << " exceeds max_tokens_for_graph_mode_prefill ("
          << FLAGS_max_tokens_for_graph_mode_prefill
          << "), falling back to eager mode";
      COUNTER_INC(num_model_execution_total_eager);
      return model_->forward(tokens, positions, kv_caches, params);
    }

    // Check if piecewise graph exists for this bucket
    auto it = prefill_graphs_.find(bucket_num_tokens);
    if (it != prefill_graphs_.end()) {
      // Replay existing piecewise graph
      VLOG(kGraphExecutorLogVerboseLevel)
          << "CudaGraphExecutorImpl::run() in prefill piecewise replay mode";
      return it->second->replay(tokens, positions, kv_caches, params);
    }

    // Graph doesn't exist, try to create it lazily with piecewise capture
    auto graph =
        std::make_unique<CudaGraph>(*persistent_param_,
                                    device_.index(),
                                    get_capture_stream(device_.index()),
                                    /*is_piecewise=*/true);
    VLOG(kGraphExecutorLogVerboseLevel)
        << "CudaGraphExecutorImpl::run() in prefill piecewise capture mode";
    bool capture_success = graph->capture(model_,
                                          args_,
                                          options_,
                                          tokens,
                                          positions,
                                          params,
                                          kv_caches,
                                          bucket_num_tokens,
                                          get_mem_pool());

    if (capture_success) {
      LOG(INFO) << "Lazy capturing piecewise CUDA graph for bucket num_tokens: "
                << bucket_num_tokens << " (actual num_tokens: " << n_tokens
                << ") done";

      // Save the graph for future reuse
      prefill_graphs_[bucket_num_tokens] = std::move(graph);

      // Return the output from capture
      return prefill_graphs_[bucket_num_tokens]->get_hidden_states(n_tokens);
    }

    // Fallback to eager mode if capture fails
    LOG(WARNING)
        << "Failed to capture piecewise graph, falling back to eager mode";
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }

  // Prefill without piecewise graph: use eager mode
  if (is_prefill) {
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }

  // Decode phase with full graph
  if (is_decode) {
    // Check if conditions are suitable for graph execution (replay or capture)
    const auto max_seq_len = FLAGS_max_seq_len_for_graph_mode > 0
                                 ? FLAGS_max_seq_len_for_graph_mode
                                 : args_.max_position_embeddings();
    const bool seq_len_supported = params.kv_max_seq_len <= max_seq_len;

    // Early return if conditions are not suitable for graph operations
    if (!seq_len_supported) {
      LOG(WARNING) << "Not suitable for CUDA graph operations, falling back to "
                      "eager mode.";
      COUNTER_INC(num_model_execution_total_eager);
      return model_->forward(tokens, positions, kv_caches, params);
    }

    // Check if captured graph exists for this bucket num_tokens
    auto it = graphs_.find(bucket_num_tokens);
    if (it != graphs_.end()) {
      // Replay the existing graph
      VLOG(kGraphExecutorLogVerboseLevel)
          << "CudaGraphExecutorImpl::run() in decode replay mode";
      return it->second->replay(tokens, positions, kv_caches, params);
    }

    // Graph doesn't exist for this bucket num_tokens, try to create it lazily
    auto graph =
        std::make_unique<CudaGraph>(*persistent_param_,
                                    device_.index(),
                                    get_capture_stream(device_.index()));
    VLOG(kGraphExecutorLogVerboseLevel)
        << "CudaGraphExecutorImpl::run() in decode capture mode";
    bool capture_success = graph->capture(model_,
                                          args_,
                                          options_,
                                          tokens,
                                          positions,
                                          params,
                                          kv_caches,
                                          bucket_num_tokens,
                                          get_mem_pool());

    if (capture_success) {
      LOG(INFO) << "Lazy capturing CUDA graph for bucket num_tokens: "
                << bucket_num_tokens << " (actual num_tokens: " << n_tokens
                << ") done";

      // Save the graph for future reuse
      graphs_[bucket_num_tokens] = std::move(graph);

      // Return the output from capture (no need to replay since capture
      // already executed)
      return graphs_[bucket_num_tokens]->get_hidden_states(n_tokens);
    } else if (FLAGS_force_graph_eager) {
      return graph->get_hidden_states(n_tokens);
    }

    // Fallback to eager mode if capture fails
    LOG(ERROR) << "Failed to capture CUDA graph for bucket num_tokens: "
               << bucket_num_tokens;
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }

  // Fallback to eager for unknown batch type
  COUNTER_INC(num_model_execution_total_eager);
  return model_->forward(tokens, positions, kv_caches, params);
}

// bucket will be [1, 2, 4, 8, 16, 32, 48, 64, ..., max_seqs_per_batch]
uint32_t CudaGraphExecutorImpl::get_bucket_num_tokens(uint32_t num_tokens,
                                                      bool is_prefill) const {
  // no_padding only works for decode, prefill requires padding for graph reuse
  if (FLAGS_enable_graph_mode_decode_no_padding && !is_prefill) {
    return num_tokens;
  }
  if (num_tokens <= 1) {
    return 1;
  } else if (num_tokens <= 2) {
    return 2;
  } else if (num_tokens <= 4) {
    return 4;
  } else if (num_tokens <= 8) {
    return 8;
  } else {
    // For num_tokens > 8, use multiples of 16
    return ((num_tokens + 31) / 32) * 32;
  }
}

}  // namespace xllm::runtime::cuda
