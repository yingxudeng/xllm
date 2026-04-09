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

#include "qwen3_next_attention.h"

#include <glog/logging.h>

#include <algorithm>
#include <array>
#include <tuple>

#include "kernels/ops_api.h"
#include "layers/common/rotary_embedding_util.h"

namespace xllm {
namespace layer {

namespace {

using torch::indexing::None;
using ISlice = torch::indexing::Slice;
constexpr int64_t kTritonMropeHeadSize = 256;
constexpr int64_t kTritonMropeRotaryDim = 64;
constexpr std::array<int64_t, 3> kNpuMropeSection = {16, 24, 24};

bool matches_mrope_section(const std::vector<int64_t>& mrope_section,
                           const std::array<int64_t, 3>& expected) {
  return mrope_section.size() == expected.size() &&
         std::equal(
             mrope_section.begin(), mrope_section.end(), expected.begin());
}

bool apply_triton_mrope(const torch::Tensor& positions,
                        int64_t num_heads,
                        int64_t num_kv_heads,
                        int64_t head_dim,
                        int64_t q_size,
                        int64_t kv_size,
                        int64_t rotary_dim,
                        const std::vector<int64_t>& mrope_section,
                        bool rotary_interleaved,
                        const torch::Tensor& mrope_cos_sin_cache,
                        torch::Tensor& q_normed,
                        torch::Tensor& k_normed,
                        torch::Tensor& q,
                        torch::Tensor& k) {
  auto cache = mrope_cos_sin_cache;
  if (cache.device() != q_normed.device() ||
      cache.dtype() != q_normed.dtype()) {
    cache = cache.to(q_normed.device(), q_normed.dtype());
  }

  auto cos_sin = cache.index({positions.to(torch::kLong)});
  auto cos_sin_split = cos_sin.chunk(2, -1);
  q = q_normed.reshape({q_normed.size(0), q_size});
  k = k_normed.reshape({k_normed.size(0), kv_size});
  xllm::kernel::MropeParams mrope_params;
  mrope_params.positions = positions;
  mrope_params.query = q;
  mrope_params.key = k;
  mrope_params.cos_sin_cache = cache;
  mrope_params.cos = cos_sin_split[0].contiguous();
  mrope_params.sin = cos_sin_split[1].contiguous();
  mrope_params.head_size = head_dim;
  mrope_params.rotary_dim = rotary_dim;
  mrope_params.mrope_section = mrope_section;
  mrope_params.interleaved = rotary_interleaved;
  std::tie(q, k) = xllm::kernel::apply_mrope(mrope_params);
  q_normed = q.view({q_normed.size(0), num_heads, head_dim});
  k_normed = k.view({k_normed.size(0), num_kv_heads, head_dim});
  return true;
}

bool apply_npu_mrope(const torch::Tensor& positions,
                     int64_t q_size,
                     int64_t kv_size,
                     int64_t head_dim,
                     int64_t rotary_dim,
                     const std::vector<int64_t>& mrope_section,
                     const torch::Tensor& mrope_cos_sin_cache,
                     torch::Tensor& q_normed,
                     torch::Tensor& k_normed,
                     torch::Tensor& q,
                     torch::Tensor& k) {
  static const std::vector<int64_t> kDecodeMropeSection = {0, 0, 0};
  const auto& mrope_section_for_call =
      positions.dim() == 1 ? kDecodeMropeSection : mrope_section;

  auto q_rotary = q_normed.index({"...", ISlice(0, rotary_dim)});
  auto k_rotary = k_normed.index({"...", ISlice(0, rotary_dim)});
  xllm::kernel::MropeParams mrope_params;
  mrope_params.positions = positions;
  mrope_params.query = q_rotary;
  mrope_params.key = k_rotary;
  mrope_params.cos_sin_cache = mrope_cos_sin_cache;
  mrope_params.head_size = head_dim;
  mrope_params.mrope_section = mrope_section_for_call;
  mrope_params.interleaved = false;
  std::tie(q_rotary, k_rotary) = xllm::kernel::apply_mrope(mrope_params);
  q_normed = torch::cat(
      {q_rotary, q_normed.index({"...", ISlice(rotary_dim, None)})}, -1);
  k_normed = torch::cat(
      {k_rotary, k_normed.index({"...", ISlice(rotary_dim, None)})}, -1);
  q = q_normed.reshape({q_normed.size(0), q_size});
  k = k_normed.reshape({k_normed.size(0), kv_size});
  return true;
}

bool apply_native_mrope(const torch::Tensor& positions,
                        const AttentionMetadata& attn_metadata,
                        int64_t q_size,
                        int64_t kv_size,
                        int64_t rotary_dim,
                        bool rotary_interleaved,
                        torch::Tensor& q_normed,
                        torch::Tensor& k_normed,
                        torch::Tensor& q,
                        torch::Tensor& k) {
  if (positions.dim() != 2 || !attn_metadata.mrope_cos.defined() ||
      !attn_metadata.mrope_sin.defined()) {
    return false;
  }

  auto q_rotary = q_normed.index({"...", ISlice(0, rotary_dim)});
  auto k_rotary = k_normed.index({"...", ISlice(0, rotary_dim)});
  std::tie(q_rotary, k_rotary) =
      layer::rotary::apply_rotary_pos_emb(q_rotary,
                                          k_rotary,
                                          attn_metadata.mrope_cos.unsqueeze(1),
                                          attn_metadata.mrope_sin.unsqueeze(1),
                                          rotary_interleaved);
  q_normed = torch::cat(
      {q_rotary, q_normed.index({"...", ISlice(rotary_dim, None)})}, -1);
  k_normed = torch::cat(
      {k_rotary, k_normed.index({"...", ISlice(rotary_dim, None)})}, -1);
  q = q_normed.reshape({q_normed.size(0), q_size});
  k = k_normed.reshape({k_normed.size(0), kv_size});
  return true;
}

}  // namespace

Qwen3NextAttentionImpl::Qwen3NextAttentionImpl(
    const ModelArgs& args,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options,
    int32_t layer_id) {
  const int64_t tp_size = parallel_args.tp_group_->world_size();
  const int64_t total_num_heads = args.n_heads();
  const int64_t total_num_kv_heads = args.n_kv_heads().value_or(args.n_heads());
  layer_id_ = layer_id;
  rank_ = parallel_args.tp_group_->rank();
  CHECK(total_num_heads % tp_size == 0);
  num_heads_ = total_num_heads / tp_size;

  if (total_num_kv_heads >= tp_size) {
    CHECK(total_num_kv_heads % tp_size == 0);
    num_kv_heads_ = total_num_kv_heads / tp_size;
    num_kv_head_replicas_ = 1;
  } else {
    CHECK(tp_size % total_num_kv_heads == 0);
    num_kv_heads_ = 1;
    num_kv_head_replicas_ = tp_size / total_num_kv_heads;
  }

  head_dim_ = args.head_dim();
  q_size_ = num_heads_ * head_dim_;
  kv_size_ = num_kv_heads_ * head_dim_;
  scaling_ = 1.0f / std::sqrt(static_cast<float>(head_dim_));
  attn_output_gate_ = args.attn_output_gate();
  // 1. QKV linear
  qkv_proj_ = register_module(
      "qkv_proj",
      QKVParallelLinear(args.hidden_size(),
                        attn_output_gate_ ? num_heads_ * 2 : num_heads_,
                        num_kv_heads_,
                        args.head_dim(),
                        num_kv_head_replicas_,
                        /*bias=*/args.attention_bias(),
                        /*gather_output=*/false,
                        parallel_args,
                        options));

  // 2. O proj
  o_proj_ = register_module("o_proj",
                            RowParallelLinear(total_num_heads * head_dim_,
                                              args.hidden_size(),
                                              /*bias=*/false,
                                              /*input_is_parallelized=*/true,
                                              /*if_reduce_results=*/true,
                                              quant_args,
                                              parallel_args.tp_group_,
                                              options));

  // 3. Q norm
  q_norm_ = register_module(
      "q_norm", Qwen3NextRMSNorm(head_dim_, args.rms_norm_eps(), options));

  // 4. K norm
  k_norm_ = register_module(
      "k_norm", Qwen3NextRMSNorm(head_dim_, args.rms_norm_eps(), options));

  // 5. Rotary embedding
  rotary_dim_ = static_cast<int64_t>(head_dim_ * args.partial_rotary_factor());
  CHECK_GT(rotary_dim_, 0) << "rotary_dim must be positive";
  partial_rotary_emb_ =
      register_module("partial_rotary_emb",
                      PartialRotaryEmbedding(rotary_dim_,
                                             args.max_position_embeddings(),
                                             args.rope_theta(),
                                             head_dim_,
                                             /*is_neox_style=*/true,
                                             /*interleaved=*/false,
                                             options));
  mrope_section_ = args.rope_scaling_mrope_section();
  rotary_interleaved_ =
      !mrope_section_.empty() && args.rope_scaling_mrope_interleaved();
  // prefer Triton for 2D interleaved positions, use npu_mrope for the
  // non-interleaved [16, 24, 24] fast path, otherwise fall back to native
  // MRoPE.
  triton_mrope_aligned_enabled_ =
      !mrope_section_.empty() && rotary_interleaved_ &&
      head_dim_ == kTritonMropeHeadSize && rotary_dim_ == kTritonMropeRotaryDim;
  if (triton_mrope_aligned_enabled_) {
    auto inv_freq = layer::rotary::compute_inv_freq(
        rotary_dim_, args.rope_theta(), options);
    const int64_t triton_cache_max_position_embeddings =
        args.max_position_embeddings() * 4;
    auto inv_freq_fp32 = inv_freq.to(torch::kFloat32).contiguous();
    auto seq = torch::arange(triton_cache_max_position_embeddings,
                             torch::TensorOptions()
                                 .dtype(torch::kFloat32)
                                 .device(inv_freq_fp32.device()));
    auto freqs = torch::einsum("i,j->ij", {seq, inv_freq_fp32});
    auto cache_options = torch::TensorOptions()
                             .dtype(options.dtype())
                             .device(inv_freq_fp32.device());
    mrope_cos_sin_cache_ = register_buffer(
        "mrope_cos_sin_cache",
        torch::cat({freqs.cos(), freqs.sin()}, -1).to(cache_options));
  }
  npu_mrope_aligned_enabled_ =
      matches_mrope_section(mrope_section_, kNpuMropeSection) &&
      !rotary_interleaved_;
  if (npu_mrope_aligned_enabled_ && !mrope_cos_sin_cache_.defined()) {
    auto inv_freq = layer::rotary::compute_inv_freq(
        rotary_dim_, args.rope_theta(), options);
    mrope_cos_sin_cache_ = register_buffer(
        "mrope_cos_sin_cache",
        layer::rotary::compute_cos_sin_cache(rotary_dim_,
                                             args.max_position_embeddings(),
                                             rotary_interleaved_,
                                             inv_freq,
                                             options));
  }

  // 6. Attention
  attn_ = register_module("attn",
                          Attention(num_heads_,
                                    head_dim_,
                                    scaling_,
                                    num_kv_heads_,
                                    args.sliding_window()));
}

bool Qwen3NextAttentionImpl::apply_mrope(const torch::Tensor& positions,
                                         const AttentionMetadata& attn_metadata,
                                         torch::Tensor& q_normed,
                                         torch::Tensor& k_normed,
                                         torch::Tensor& q,
                                         torch::Tensor& k) {
  if (mrope_section_.empty()) {
    return false;
  }

  if (triton_mrope_aligned_enabled_ && positions.dim() == 2) {
    return apply_triton_mrope(positions,
                              num_heads_,
                              num_kv_heads_,
                              head_dim_,
                              q_size_,
                              kv_size_,
                              rotary_dim_,
                              mrope_section_,
                              rotary_interleaved_,
                              mrope_cos_sin_cache_,
                              q_normed,
                              k_normed,
                              q,
                              k);
  }

  if (npu_mrope_aligned_enabled_ &&
      (positions.dim() == 1 || positions.dim() == 2)) {
    return apply_npu_mrope(positions,
                           q_size_,
                           kv_size_,
                           head_dim_,
                           rotary_dim_,
                           mrope_section_,
                           mrope_cos_sin_cache_,
                           q_normed,
                           k_normed,
                           q,
                           k);
  }

  return apply_native_mrope(positions,
                            attn_metadata,
                            q_size_,
                            kv_size_,
                            rotary_dim_,
                            rotary_interleaved_,
                            q_normed,
                            k_normed,
                            q,
                            k);
}

torch::Tensor Qwen3NextAttentionImpl::forward(
    const torch::Tensor& positions,
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache) {
  // 1. qkv projection
  auto qkv = qkv_proj_->forward(hidden_states);
  torch::Tensor q, k, v;
  torch::Tensor gate;

  if (attn_output_gate_) {
    // Split qkv for attn_output_gate case: [q_size*2, kv_size, kv_size]
    auto q_gate = qkv.slice(/*dim=*/-1, 0, q_size_ * 2);
    k = qkv.slice(/*dim=*/-1, q_size_ * 2, q_size_ * 2 + kv_size_);
    v = qkv.slice(
        /*dim=*/-1, q_size_ * 2 + kv_size_, q_size_ * 2 + kv_size_ * 2);
    v = v.contiguous();

    std::vector<int64_t> orig_shape;
    int64_t q_gate_dim = q_gate.dim();
    orig_shape =
        std::vector<int64_t>(q_gate.sizes().slice(0, q_gate_dim - 1).begin(),
                             q_gate.sizes().slice(0, q_gate_dim - 1).end());

    std::vector<int64_t> new_shape = orig_shape;
    new_shape.push_back(num_heads_);
    int64_t orig_total = 1;
    for (auto d : orig_shape) orig_total *= d;
    int64_t last_dim = q_gate.numel() / (orig_total * num_heads_);
    new_shape.push_back(last_dim);

    torch::Tensor q_gate_reshaped = q_gate.reshape(new_shape);

    auto chunks = torch::chunk(q_gate_reshaped, 2, /*dim=*/-1);
    q = chunks[0];
    gate = chunks[1];

    std::vector<int64_t> q_new_shape = orig_shape;
    q_new_shape.push_back(q.numel() / orig_total);
    q = q.reshape(q_new_shape);

    std::vector<int64_t> gate_new_shape = orig_shape;
    gate_new_shape.push_back(gate.numel() / orig_total);
    gate = gate.reshape(gate_new_shape);
  } else {
    // Normal case: [q_size, kv_size, kv_size]
    q = qkv.slice(/*dim=*/-1, 0, q_size_);
    k = qkv.slice(/*dim=*/-1, q_size_, q_size_ + kv_size_);
    v = qkv.slice(/*dim=*/-1, q_size_ + kv_size_, q_size_ + 2 * kv_size_);
  }

  const int64_t T = q.size(0);

  auto q_reshaped = q.reshape({T, num_heads_, head_dim_});
  auto k_reshaped = k.reshape({T, num_kv_heads_, head_dim_});
  auto q_normed = q_norm_->forward(q_reshaped);
  auto k_normed = k_norm_->forward(k_reshaped);
  q = q_normed.reshape({T, q_size_});
  k = k_normed.reshape({T, kv_size_});
  if (!apply_mrope(positions, attn_metadata, q_normed, k_normed, q, k)) {
    q = q_normed.reshape({T, q_size_});
    k = k_normed.reshape({T, kv_size_});
    const torch::Tensor rotary_positions =
        positions.dim() == 2 ? positions[0] : positions;
    partial_rotary_emb_->forward(rotary_positions, q, k);
  }
  auto out = std::get<0>(attn_->forward(attn_metadata, q, k, v, kv_cache));

  if (attn_output_gate_) {
    gate = torch::sigmoid(gate);
    out = out * gate;
  }

  out = o_proj_->forward(out);
  return out;
}

void Qwen3NextAttentionImpl::load_state_dict(const StateDict& state_dict) {
  qkv_proj_->load_state_dict(state_dict, {"q_proj.", "k_proj.", "v_proj."});
  o_proj_->load_state_dict(state_dict.get_dict_with_prefix("o_proj."));
  if (auto w = state_dict.get_tensor("q_norm.weight"); w.defined()) {
    q_norm_->load_state_dict(StateDict({{"weight", w}}));
  }
  if (auto w = state_dict.get_tensor("k_norm.weight"); w.defined()) {
    k_norm_->load_state_dict(StateDict({{"weight", w}}));
  }
}

}  // namespace layer
}  // namespace xllm
