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

#include <algorithm>

#include "core/framework/state_dict/utils.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "core/layers/common/rms_norm.h"
#include "core/layers/common/word_embedding.h"
#include "core/layers/deepseek_v4_decoder_layer.h"
#include "layers/common/rotary_embedding_util.h"
#include "llm_model_base.h"

namespace xllm {

class DeepseekV4ModelImpl
    : public LlmModelImplBase<layer::DeepseekV4DecoderLayer> {
 public:
  explicit DeepseekV4ModelImpl(const ModelContext& context)
      : LlmModelImplBase<layer::DeepseekV4DecoderLayer>(
            "deepseek_v4",
            context.get_model_args()) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();

    layers_.reserve(model_args.n_layers());
    norm_ = register_module("norm", layer::RMSNorm(context));
    embed_tokens_ =
        register_module("embed_tokens", layer::WordEmbedding(context));

    hc_mult_ = std::max<int64_t>(model_args.hc_mult(), 1);
    hc_eps_ = static_cast<double>(model_args.hc_eps());
    norm_eps_ = static_cast<double>(model_args.rms_norm_eps());

    const int64_t hc_dim = hc_mult_ * model_args.hidden_size();
    auto hc_options = options.dtype(torch::kFloat32);
    hc_head_fn_ =
        register_parameter("hc_head_fn",
                           torch::empty({hc_mult_, hc_dim}, hc_options),
                           /*requires_grad=*/false);
    hc_head_base_ = register_parameter("hc_head_base",
                                       torch::empty({hc_mult_}, hc_options),
                                       /*requires_grad=*/false);
    hc_head_scale_ = register_parameter("hc_head_scale",
                                        torch::empty({1}, hc_options),
                                        /*requires_grad=*/false);

    const int64_t rope_head_dim = model_args.rope_head_dim();
    const int64_t max_pos = model_args.max_position_embeddings();
    if (rope_head_dim > 0 && max_pos > 0) {
      const int64_t original_max_pos =
          model_args.rope_scaling_original_max_position_embeddings() > 0
              ? model_args.rope_scaling_original_max_position_embeddings()
              : max_pos;
      const float scaling_factor =
          model_args.factor() > 0.0f ? model_args.factor() : 1.0f;
      const float attn_factor = model_args.rope_scaling_attn_factor() > 0.0f
                                    ? model_args.rope_scaling_attn_factor()
                                    : 1.0f;
      dsa_cos_sin_ = layer::rotary::get_deepseek_rotary_embedding(
          /*head_size=*/model_args.head_dim(),
          /*rotary_dim=*/rope_head_dim,
          /*max_position_embeddings=*/max_pos,
          /*rope_scaling_original_max_position_embeddings=*/original_max_pos,
          /*rope_theta=*/model_args.rope_theta(),
          /*interleaved=*/false,
          /*scaling_factor=*/scaling_factor,
          /*extrapolation_factor=*/model_args.rope_extrapolation_factor(),
          /*attn_factor=*/attn_factor,
          /*beta_fast=*/model_args.beta_fast(),
          /*beta_slow=*/model_args.beta_slow(),
          /*mscale=*/model_args.rope_scaling_mscale(),
          /*mscale_all_dim=*/model_args.rope_scaling_mscale_all_dim(),
          options);
    }

    for (int32_t i = 0; i < model_args.n_layers(); ++i) {
      auto layer = layer::DeepseekV4DecoderLayer(context);
      layers_.push_back(layer);
    }
  }

  void load_state_dict(const StateDict& state_dict) override {
    LlmModelImplBase<layer::DeepseekV4DecoderLayer>::load_state_dict(
        state_dict);
    LOAD_WEIGHT(hc_head_fn);
    LOAD_WEIGHT(hc_head_base);
    LOAD_WEIGHT(hc_head_scale);
  }

  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) override {
    if (tokens.numel() == 0) {
      tokens = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
      positions = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
    }

    auto inputs_embeds = input_params.input_embedding;
    torch::Tensor h =
        inputs_embeds.defined() ? inputs_embeds : embed_tokens_(tokens);

    if (h.dim() == 2) {
      h = h.unsqueeze(1).repeat({1, hc_mult_, 1});
    }

    auto modified_input_params = input_params;
    auto& dp_token_nums = modified_input_params.dp_global_token_nums;
    // DP helper: keep zero entries at least 1 to avoid empty slices/padding
    // in xllm DP utilities. DeepSeek V4 not use DP today.
    std::replace(dp_token_nums.begin(), dp_token_nums.end(), 0, 1);

    if (!modified_input_params.attn_metadata) {
      modified_input_params.attn_metadata =
          std::make_shared<layer::AttentionMetadata>(
              layer::AttentionMetadataBuilder::build(modified_input_params));
    }
    auto& attn_metadata = *(modified_input_params.attn_metadata);

    if (dsa_cos_sin_.defined() && positions.defined()) {
      torch::Tensor cos_sin = dsa_cos_sin_;
      if (cos_sin.device() != positions.device()) {
        cos_sin = cos_sin.to(positions.device());
      }
      auto target = cos_sin.index({positions});
      auto chunks = target.chunk(/*chunks=*/2, /*dim=*/-1);
      attn_metadata.dsa_cos = chunks[0].contiguous();
      attn_metadata.dsa_sin = chunks[1].contiguous();
    }

    std::optional<torch::Tensor> residual;
    for (size_t i = 0; i < layers_.size(); i++) {
      h = layers_[i](h,
                     residual,
                     positions,
                     attn_metadata,
                     kv_caches[i],
                     modified_input_params);
    }
    h = hc_head(h);
    auto [hidden_states, residual_out] = norm_(h, std::nullopt);
    return ModelOutput(hidden_states, residual_out);
  }

 private:
  torch::Tensor hc_head(const torch::Tensor& x) {
    auto x_float = x.to(torch::kFloat32);
    auto x_flatten = x_float.flatten(-2, -1);
    auto rsqrt = torch::rsqrt(x_flatten.pow(2).mean(-1, true) + norm_eps_);
    auto mixes = torch::matmul(x_flatten, hc_head_fn_.transpose(0, 1));
    mixes = mixes * rsqrt;
    auto pre = torch::sigmoid(mixes * hc_head_scale_ + hc_head_base_) + hc_eps_;
    auto y = (pre.unsqueeze(-1) * x_float).sum(-2);
    return y.to(x.dtype());
  }

  torch::Tensor dsa_cos_sin_;

  int64_t hc_mult_ = 1;
  double hc_eps_ = 0.0;
  double norm_eps_ = 1e-6;

  DEFINE_WEIGHT(hc_head_fn);
  DEFINE_WEIGHT(hc_head_base);
  DEFINE_WEIGHT(hc_head_scale);
};
TORCH_MODULE(DeepseekV4Model);

class DeepseekV4ForCausalLMImpl
    : public LlmForCausalLMImplBase<DeepseekV4Model> {
 public:
  explicit DeepseekV4ForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<DeepseekV4Model>(context) {}
};
TORCH_MODULE(DeepseekV4ForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL(deepseek_v4, DeepseekV4ForCausalLM);

// register the model args
REGISTER_MODEL_ARGS(deepseek_v4, [&] {
  LOAD_ARG_OR(model_type, "model_type", "deepseek_v4");
  LOAD_ARG_OR(dtype, "torch_dtype", "");

  // Basic model structure
  LOAD_ARG_OR_FUNC(hidden_size, "dim", [&] { return args->hidden_size(); });
  LOAD_ARG_OR_FUNC(
      hidden_size, "hidden_size", [&] { return args->hidden_size(); });
  LOAD_ARG_OR_FUNC(
      n_layers, "num_hidden_layers", [&] { return args->n_layers(); });
  LOAD_ARG_OR_FUNC(n_heads, "n_heads", [&] { return args->n_heads(); });
  LOAD_ARG_OR_FUNC(
      n_heads, "num_attention_heads", [&] { return args->n_heads(); });
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 1);
  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    if (args->head_dim() > 0) {
      return args->head_dim();
    }
    if (args->hidden_size() > 0 && args->n_heads() > 0) {
      return args->hidden_size() / args->n_heads();
    }
    return int64_t{0};
  });
  LOAD_ARG_OR_FUNC(
      vocab_size, "vocab_size", [&] { return args->vocab_size(); });
  LOAD_ARG_OR_FUNC(max_position_embeddings, "max_position_embeddings", [&] {
    return args->max_position_embeddings();
  });
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR_FUNC(intermediate_size, "intermediate_size", [&] {
    if (args->intermediate_size() > 0) {
      return args->intermediate_size();
    }
    if (args->moe_intermediate_size() > 0) {
      return static_cast<int64_t>(args->moe_intermediate_size());
    }
    if (args->hidden_size() > 0) {
      return args->hidden_size() * 4;
    }
    return int64_t{0};
  });

  // Norm / RoPE
  LOAD_ARG_OR_FUNC(
      rms_norm_eps, "norm_eps", [&] { return args->rms_norm_eps(); });
  LOAD_ARG_OR_FUNC(
      rms_norm_eps, "rms_norm_eps", [&] { return args->rms_norm_eps(); });
  LOAD_ARG_OR_FUNC(
      rope_theta, "rope_theta", [&] { return args->rope_theta(); });
  LOAD_ARG_OR_FUNC(
      rope_head_dim, "rope_head_dim", [&] { return args->rope_head_dim(); });

  // LoRA / groups
  LOAD_ARG_OR_FUNC(
      q_lora_rank, "q_lora_rank", [&] { return args->q_lora_rank(); });
  LOAD_ARG_OR_FUNC(
      o_lora_rank, "o_lora_rank", [&] { return args->o_lora_rank(); });
  LOAD_ARG_OR_FUNC(o_groups, "o_groups", [&] { return args->o_groups(); });

  // KV compression / windowing
  LOAD_ARG(compress_ratios, "compress_ratios");
  LOAD_ARG_OR_FUNC(compress_rope_theta, "compress_rope_theta", [&] {
    return args->compress_rope_theta();
  });
  LOAD_ARG_OR_FUNC(
      window_size, "window_size", [&] { return args->window_size(); });

  // MoE routing (DeepSeek V4)
  LOAD_ARG_OR_FUNC(n_routed_experts, "n_routed_experts", [&] {
    return args->n_routed_experts();
  });
  LOAD_ARG_OR_FUNC(n_activated_experts, "n_activated_experts", [&] {
    return args->n_activated_experts();
  });
  LOAD_ARG_OR_FUNC(
      n_hash_layers, "n_hash_layers", [&] { return args->n_hash_layers(); });
  LOAD_ARG_OR_FUNC(
      route_scale, "route_scale", [&] { return args->route_scale(); });
  LOAD_ARG_OR_FUNC(
      score_func, "score_func", [&] { return args->score_func(); });

  // Indexer
  LOAD_ARG_OR_FUNC(
      index_head_dim, "index_head_dim", [&] { return args->index_head_dim(); });
  LOAD_ARG_OR_FUNC(
      index_n_heads, "index_n_heads", [&] { return args->index_n_heads(); });
  LOAD_ARG_OR_FUNC(
      index_topk, "index_topk", [&] { return args->index_topk(); });

  // HC / DSA helpers
  LOAD_ARG_OR_FUNC(hc_mult, "hc_mult", [&] { return args->hc_mult(); });
  LOAD_ARG_OR_FUNC(hc_sinkhorn_iters, "hc_sinkhorn_iters", [&] {
    return args->hc_sinkhorn_iters();
  });
  LOAD_ARG_OR_FUNC(hc_eps, "hc_eps", [&] { return args->hc_eps(); });
  LOAD_ARG_OR_FUNC(factor, "factor", [&] { return args->factor(); });
  LOAD_ARG_OR_FUNC(beta_fast, "beta_fast", [&] { return args->beta_fast(); });
  LOAD_ARG_OR_FUNC(beta_slow, "beta_slow", [&] { return args->beta_slow(); });
  LOAD_ARG_OR_FUNC(scale_fmt, "scale_fmt", [&] { return args->scale_fmt(); });

  // Runtime sizing hints
  LOAD_ARG_OR_FUNC(
      max_batch_size, "max_batch_size", [&] { return args->max_batch_size(); });
  LOAD_ARG_OR_FUNC(
      max_seq_len, "max_seq_len", [&] { return args->max_seq_len(); });

  // Token ids
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 0);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 1);

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
});

}  // namespace xllm
