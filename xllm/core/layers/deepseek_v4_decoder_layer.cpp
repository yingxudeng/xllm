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

#include "deepseek_v4_decoder_layer.h"

#include <torch/nn/functional.h>

#include <algorithm>

namespace xllm {
namespace layer {

DeepseekV4DecoderLayerImpl::DeepseekV4DecoderLayerImpl(
    const ModelContext& context) {
  const auto& args = context.get_model_args();
  const auto& quant_args = context.get_quant_args();
  const auto& parallel_args = context.get_parallel_args();
  const auto& options = context.get_tensor_options();

  int64_t hidden_size = args.hidden_size();
  int64_t intermediate_size = args.intermediate_size();
  if (intermediate_size <= 0) {
    if (args.moe_intermediate_size() > 0) {
      intermediate_size = args.moe_intermediate_size();
    } else if (hidden_size > 0) {
      intermediate_size = hidden_size * 4;
    }
  }
  std::string hidden_act =
      args.hidden_act().empty() ? "silu" : args.hidden_act();

  hc_mult_ = std::max<int64_t>(args.hc_mult(), 1);
  hc_sinkhorn_iters_ = args.hc_sinkhorn_iters();
  hc_eps_ = static_cast<double>(args.hc_eps());
  norm_eps_ = static_cast<double>(args.rms_norm_eps());

  attention_ = register_module("attn", Qwen2Attention(context));
  attn_norm_ = register_module(
      "attn_norm", RMSNorm(hidden_size, args.rms_norm_eps(), options));
  ffn_norm_ = register_module(
      "ffn_norm", RMSNorm(hidden_size, args.rms_norm_eps(), options));
  mlp_ = register_module("ffn",
                         DenseMLP(hidden_size,
                                  intermediate_size,
                                  /*is_gated=*/true,
                                  /*has_bias=*/false,
                                  hidden_act,
                                  /*enable_result_reduction=*/true,
                                  quant_args,
                                  parallel_args.tp_group_,
                                  options));

  const int64_t mix_hc = (2 + hc_mult_) * hc_mult_;
  const int64_t hc_dim = hc_mult_ * hidden_size;
  auto hc_options = options.dtype(torch::kFloat32);
  hc_attn_fn_ = register_parameter("hc_attn_fn",
                                   torch::empty({mix_hc, hc_dim}, hc_options),
                                   /*requires_grad=*/false);
  hc_ffn_fn_ = register_parameter("hc_ffn_fn",
                                  torch::empty({mix_hc, hc_dim}, hc_options),
                                  /*requires_grad=*/false);
  hc_attn_base_ = register_parameter("hc_attn_base",
                                     torch::empty({mix_hc}, hc_options),
                                     /*requires_grad=*/false);
  hc_ffn_base_ = register_parameter("hc_ffn_base",
                                    torch::empty({mix_hc}, hc_options),
                                    /*requires_grad=*/false);
  hc_attn_scale_ = register_parameter("hc_attn_scale",
                                      torch::empty({3}, hc_options),
                                      /*requires_grad=*/false);
  hc_ffn_scale_ = register_parameter("hc_ffn_scale",
                                     torch::empty({3}, hc_options),
                                     /*requires_grad=*/false);
}

void DeepseekV4DecoderLayerImpl::load_state_dict(const StateDict& state_dict) {
  auto attn_state = state_dict.get_dict_with_prefix("attn.");
  if (attn_state.size() == 0) {
    attn_state = state_dict.get_dict_with_prefix("self_attn.");
  }
  if (attn_state.size() > 0) {
    attention_->load_state_dict(attn_state);
  }

  auto attn_norm_state = state_dict.get_dict_with_prefix("attn_norm.");
  if (attn_norm_state.size() == 0) {
    attn_norm_state = state_dict.get_dict_with_prefix("input_layernorm.");
  }
  if (attn_norm_state.size() > 0) {
    attn_norm_->load_state_dict(attn_norm_state);
  }

  auto ffn_norm_state = state_dict.get_dict_with_prefix("ffn_norm.");
  if (ffn_norm_state.size() == 0) {
    ffn_norm_state =
        state_dict.get_dict_with_prefix("post_attention_layernorm.");
  }
  if (ffn_norm_state.size() > 0) {
    ffn_norm_->load_state_dict(ffn_norm_state);
  }

  auto ffn_state = state_dict.get_dict_with_prefix("ffn.");
  if (ffn_state.size() > 0) {
    mlp_->load_state_dict(ffn_state, {"w1.", "w3."}, "w2.");
  } else {
    auto mlp_state = state_dict.get_dict_with_prefix("mlp.");
    if (mlp_state.size() > 0) {
      mlp_->load_state_dict(mlp_state);
    }
  }

  LOAD_WEIGHT(hc_attn_fn);
  LOAD_WEIGHT(hc_ffn_fn);
  LOAD_WEIGHT(hc_attn_base);
  LOAD_WEIGHT(hc_ffn_base);
  LOAD_WEIGHT(hc_attn_scale);
  LOAD_WEIGHT(hc_ffn_scale);
}

torch::Tensor DeepseekV4DecoderLayerImpl::forward(
    torch::Tensor& x,
    std::optional<torch::Tensor>& residual,
    torch::Tensor& positions,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params) {
  (void)input_params;

  residual = std::nullopt;

  auto residual_attn = x;
  auto [attn_input, post_attn, comb_attn] =
      hc_pre(x, hc_attn_fn_, hc_attn_scale_, hc_attn_base_);
  attn_input = std::get<0>(attn_norm_->forward(attn_input));
  attn_input =
      attention_->forward(positions, attn_input, attn_metadata, kv_cache);
  x = hc_post(attn_input, residual_attn, post_attn, comb_attn);

  auto residual_ffn = x;
  auto [ffn_input, post_ffn, comb_ffn] =
      hc_pre(x, hc_ffn_fn_, hc_ffn_scale_, hc_ffn_base_);
  ffn_input = std::get<0>(ffn_norm_->forward(ffn_input));
  ffn_input = mlp_->forward(ffn_input);
  x = hc_post(ffn_input, residual_ffn, post_ffn, comb_ffn);

  return x;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
DeepseekV4DecoderLayerImpl::hc_pre(const torch::Tensor& x,
                                   const torch::Tensor& hc_fn,
                                   const torch::Tensor& hc_scale,
                                   const torch::Tensor& hc_base) {
  auto x_float = x.to(torch::kFloat32);
  auto x_flatten = x_float.flatten(-2, -1);
  auto rsqrt = torch::rsqrt(x_flatten.pow(2).mean(-1, true) + norm_eps_);

  auto mixes = torch::matmul(x_flatten, hc_fn.transpose(0, 1));
  mixes = mixes * rsqrt;

  const int64_t hc_mult = hc_mult_;
  const int64_t mix_hc = (2 + hc_mult) * hc_mult;

  auto mixes_pre = mixes.slice(-1, 0, hc_mult);
  auto mixes_post = mixes.slice(-1, hc_mult, 2 * hc_mult);
  auto mixes_comb = mixes.slice(-1, 2 * hc_mult, mix_hc);

  auto hc_scale_pre = hc_scale.index({0});
  auto hc_scale_post = hc_scale.index({1});
  auto hc_scale_comb = hc_scale.index({2});

  auto hc_base_pre = hc_base.slice(0, 0, hc_mult);
  auto hc_base_post = hc_base.slice(0, hc_mult, 2 * hc_mult);
  auto hc_base_comb = hc_base.slice(0, 2 * hc_mult, mix_hc);

  auto pre = torch::sigmoid(mixes_pre * hc_scale_pre + hc_base_pre) + hc_eps_;
  auto post = torch::sigmoid(mixes_post * hc_scale_post + hc_base_post) * 2.0f;

  auto comb_shape = mixes_comb.sizes().vec();
  comb_shape.back() = hc_mult;
  comb_shape.push_back(hc_mult);
  auto comb = (mixes_comb * hc_scale_comb + hc_base_comb).reshape(comb_shape);
  comb = torch::softmax(comb, -1) + hc_eps_;
  for (int64_t iter = 0; iter < hc_sinkhorn_iters_; ++iter) {
    comb = comb / (comb.sum(-1, true) + hc_eps_);
    comb = comb / (comb.sum(-2, true) + hc_eps_);
  }

  auto y = (pre.unsqueeze(-1) * x_float).sum(-2);
  y = y.to(x.dtype());
  return {y, post, comb};
}

torch::Tensor DeepseekV4DecoderLayerImpl::hc_post(const torch::Tensor& x,
                                                  const torch::Tensor& residual,
                                                  const torch::Tensor& post,
                                                  const torch::Tensor& comb) {
  auto x_float = x.to(torch::kFloat32);
  auto residual_float = residual.to(torch::kFloat32);
  auto post_float = post.to(torch::kFloat32);
  auto comb_float = comb.to(torch::kFloat32);

  auto residual_mix =
      torch::matmul(comb_float.transpose(-1, -2), residual_float);
  auto x_scaled = x_float.unsqueeze(-2) * post_float.unsqueeze(-1);
  auto y = residual_mix + x_scaled;
  return y.to(residual.dtype());
}

}  // namespace layer
}  // namespace xllm
