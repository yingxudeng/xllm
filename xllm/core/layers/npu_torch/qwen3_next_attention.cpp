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

#include <tuple>

#include "layers/common/rotary_embedding_util.h"

namespace xllm {
namespace layer {

namespace {

using torch::indexing::None;
using ISlice = torch::indexing::Slice;

inline torch::Tensor rotate_every_two(const torch::Tensor& x) {
  auto x1 = x.index({ISlice(), ISlice(), ISlice(0, None, 2)});
  auto x2 = x.index({ISlice(), ISlice(), ISlice(1, None, 2)});
  return torch::stack({-x2, x1}, /*dim=*/-1).flatten(/*start_dim=*/-2);
}

inline torch::Tensor rotate_half(const torch::Tensor& x) {
  auto chunks = x.chunk(2, /*dim=*/-1);
  return torch::cat({-chunks[1], chunks[0]}, /*dim=*/-1);
}

inline std::tuple<torch::Tensor, torch::Tensor>
apply_interleaved_rotary_pos_emb(const torch::Tensor& q,
                                 const torch::Tensor& k,
                                 const torch::Tensor& cos,
                                 const torch::Tensor& sin) {
  auto q_embed = (q * cos) + (rotate_every_two(q) * sin);
  auto k_embed = (k * cos) + (rotate_every_two(k) * sin);
  return std::make_tuple(q_embed, k_embed);
}

inline std::tuple<torch::Tensor, torch::Tensor> apply_rotated_rotary_pos_emb(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& cos,
    const torch::Tensor& sin) {
  auto q_embed = (q * cos) + (rotate_half(q) * sin);
  auto k_embed = (k * cos) + (rotate_half(k) * sin);
  return std::make_tuple(q_embed, k_embed);
}

inline std::tuple<torch::Tensor, torch::Tensor> apply_rotary_pos_emb(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& cos_sin,
    bool interleaved) {
  const auto chunks = cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
  if (interleaved) {
    return apply_interleaved_rotary_pos_emb(q, k, chunks[0], chunks[1]);
  }
  return apply_rotated_rotary_pos_emb(q, k, chunks[0], chunks[1]);
}

torch::Tensor build_mrope_cos_sin(const torch::Tensor& cos_sin_cache,
                                  const torch::Tensor& positions,
                                  bool interleaved,
                                  const std::vector<int64_t>& mrope_section) {
  namespace F = torch::nn::functional;
  auto cos_sin = F::embedding(positions, cos_sin_cache);
  if (positions.dim() != 2 || mrope_section.empty()) {
    return cos_sin.unsqueeze(1);
  }

  auto chunks = cos_sin.chunk(2, -1);
  auto reorder = [&](const torch::Tensor& x) {
    std::vector<int64_t> sections = mrope_section;
    if (interleaved) {
      for (auto& section : sections) {
        section *= 2;
      }
    } else {
      sections.insert(
          sections.end(), mrope_section.begin(), mrope_section.end());
    }

    auto vec = x.split(sections, -1);
    std::vector<torch::Tensor> selects;
    selects.reserve(vec.size());
    for (int64_t i = 0; i < static_cast<int64_t>(vec.size()); ++i) {
      const int64_t pos_axis =
          interleaved ? i : (i % static_cast<int64_t>(mrope_section.size()));
      CHECK_LT(pos_axis, vec[i].size(0))
          << "mRoPE position axis out of range for section " << i;
      selects.push_back(vec[i].select(/*dim=*/0, /*index=*/pos_axis));
    }
    return torch::cat(selects, -1);
  };

  auto cos = reorder(chunks[0]);
  auto sin = reorder(chunks[1]);
  return torch::cat({cos, sin}, -1).unsqueeze(1);
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
  mrope_section_ = args.rope_scaling_mrope_section();
  rotary_interleaved_ =
      !mrope_section_.empty() && args.rope_scaling_mrope_interleaved();
  auto inv_freq =
      layer::rotary::compute_inv_freq(rotary_dim_, args.rope_theta(), options);
  const auto rotary_cos_sin =
      layer::rotary::compute_cos_sin_cache(rotary_dim_,
                                           args.max_position_embeddings(),
                                           rotary_interleaved_,
                                           inv_freq,
                                           options);
  rotary_cos_sin_cache_ =
      register_buffer("rotary_cos_sin_cache", rotary_cos_sin.to(options));

  // 6. Attention
  attn_ = register_module("attn",
                          Attention(num_heads_,
                                    head_dim_,
                                    scaling_,
                                    num_kv_heads_,
                                    args.sliding_window()));
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

  auto q_rotary = q_normed.index({"...", ISlice(0, rotary_dim_)});
  auto k_rotary = k_normed.index({"...", ISlice(0, rotary_dim_)});
  auto rotary_cos_sin = build_mrope_cos_sin(
      rotary_cos_sin_cache_, positions, rotary_interleaved_, mrope_section_);
  std::tie(q_rotary, k_rotary) = apply_rotary_pos_emb(
      q_rotary, k_rotary, rotary_cos_sin, rotary_interleaved_);
  q_normed = torch::cat(
      {q_rotary, q_normed.index({"...", ISlice(rotary_dim_, None)})}, -1);
  k_normed = torch::cat(
      {k_rotary, k_normed.index({"...", ISlice(rotary_dim_, None)})}, -1);

  q = q_normed.reshape({T, q_size_});
  k = k_normed.reshape({T, kv_size_});
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
