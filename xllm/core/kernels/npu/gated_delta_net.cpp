/* Copyright 2026 The xLLM Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://github.com/jd-opensource/xllm/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================
 */

#include <glog/logging.h>
#include <torch/torch.h>

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "npu_ops_api.h"
#include "triton_npu/torch_api/triton_ops_api.h"

namespace xllm::kernel::npu {

namespace {

bool should_log_qwen35_fast_fused_sigmoid_gdn_update_hit() {
  static std::atomic<bool> should_log{true};
  return should_log.exchange(false);
}

bool should_log_qwen35_fast_fused_sigmoid_gdn_update_fallback() {
  static std::atomic<bool> should_log{true};
  return should_log.exchange(false);
}

bool use_reference_fused_sigmoid_gating_delta_rule_update() {
  const char* env = std::getenv("XLLM_DEBUG_REF_FUSED_SIGMOID_GDN_UPDATE");
  return env != nullptr && std::string(env) == "1";
}

bool enable_qwen35_fast_fused_sigmoid_gdn_update() {
  const char* env =
      std::getenv("XLLM_ENABLE_QWEN35_FAST_FUSED_SIGMOID_GDN_UPDATE");
  return env != nullptr && std::string(env) == "1";
}

torch::Tensor l2norm(const torch::Tensor& x, int64_t dim, double eps = 1e-6) {
  torch::Tensor norm =
      torch::sqrt(torch::sum(torch::square(x), dim, true) + eps);
  return x / norm;
}

std::tuple<torch::Tensor, torch::Tensor> chunk_gated_delta_rule_impl(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor g,
    torch::Tensor beta,
    int64_t chunk_size,
    const std::optional<torch::Tensor>& initial_state,
    bool use_qk_l2norm_in_kernel) {
  torch::ScalarType initial_dtype = query.scalar_type();
  if (use_qk_l2norm_in_kernel) {
    query = l2norm(query, -1, 1e-6);
    key = l2norm(key, -1, 1e-6);
  }
  auto to_float32 = [](torch::Tensor x) {
    return x.transpose(1, 2).contiguous().to(torch::kFloat32);
  };

  query = to_float32(query);
  key = to_float32(key);
  value = to_float32(value);
  beta = to_float32(beta);
  g = to_float32(g);

  int64_t batch_size = query.size(0);
  int64_t num_k_heads = query.size(1);
  int64_t num_v_heads = value.size(1);
  int64_t sequence_length = query.size(2);
  int64_t k_head_dim = key.size(-1);
  int64_t v_head_dim = value.size(-1);
  CHECK(num_k_heads > 0) << "num_k_heads must be positive.";
  CHECK(num_v_heads > 0) << "num_v_heads must be positive.";
  CHECK_EQ(num_v_heads % num_k_heads, 0)
      << "num_v_heads must be divisible by num_k_heads.";
  int64_t kv_head_ratio = num_v_heads / num_k_heads;
  if (kv_head_ratio > 1) {
    query = query.repeat_interleave(kv_head_ratio, 1);
    key = key.repeat_interleave(kv_head_ratio, 1);
  }

  int64_t pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size;
  query = torch::nn::functional::pad(
      query, torch::nn::functional::PadFuncOptions({0, 0, 0, pad_size}));
  key = torch::nn::functional::pad(
      key, torch::nn::functional::PadFuncOptions({0, 0, 0, pad_size}));
  value = torch::nn::functional::pad(
      value, torch::nn::functional::PadFuncOptions({0, 0, 0, pad_size}));
  beta = torch::nn::functional::pad(
      beta, torch::nn::functional::PadFuncOptions({0, pad_size}));
  g = torch::nn::functional::pad(
      g, torch::nn::functional::PadFuncOptions({0, pad_size}));

  int64_t total_sequence_length = sequence_length + pad_size;
  float scale = 1.0f / std::sqrt(static_cast<float>(query.size(-1)));
  query = query * scale;
  torch::Tensor v_beta = value * beta.unsqueeze(-1);
  torch::Tensor k_beta = key * beta.unsqueeze(-1);
  auto reshape_to_chunks = [chunk_size](torch::Tensor x) {
    auto shape = x.sizes();
    std::vector<int64_t> new_shape = {
        shape[0], shape[1], shape[2] / chunk_size, chunk_size, shape[3]};
    return x.reshape(new_shape);
  };

  query = reshape_to_chunks(query);
  key = reshape_to_chunks(key);
  value = reshape_to_chunks(value);
  k_beta = reshape_to_chunks(k_beta);
  v_beta = reshape_to_chunks(v_beta);

  auto g_shape = g.sizes();
  std::vector<int64_t> g_new_shape = {
      g_shape[0], g_shape[1], g_shape[2] / chunk_size, chunk_size};
  g = g.reshape(g_new_shape);
  torch::Tensor mask = torch::triu(
      torch::ones(
          {chunk_size, chunk_size},
          torch::TensorOptions().dtype(torch::kBool).device(query.device())),
      0);

  g = g.cumsum(-1);
  torch::Tensor g_diff = g.unsqueeze(-1) - g.unsqueeze(-2);
  torch::Tensor decay_mask = g_diff.tril().exp().to(torch::kFloat32);
  decay_mask = decay_mask.tril();
  torch::Tensor attn =
      -(torch::matmul(k_beta, key.transpose(-1, -2)) * decay_mask)
           .masked_fill(mask, 0.0);
  for (int64_t i = 1; i < chunk_size; ++i) {
    if (!attn.is_contiguous()) {
      attn = attn.contiguous();
    }
    torch::Tensor row = attn.slice(-2, i, i + 1)
                            .slice(-1, 0, i)
                            .squeeze(-2)
                            .clone()
                            .contiguous();
    torch::Tensor sub =
        attn.slice(-2, 0, i).slice(-1, 0, i).clone().contiguous();
    torch::Tensor row_unsq = row.unsqueeze(-1).contiguous();
    torch::Tensor row_sub_mul = (row_unsq * sub).contiguous();
    torch::Tensor row_sub_sum = row_sub_mul.sum(-2).contiguous();
    torch::Tensor row_final = (row + row_sub_sum).contiguous();
    attn.index_put_({torch::indexing::Ellipsis,
                     torch::indexing::Slice(i, i + 1),
                     torch::indexing::Slice(0, i)},
                    row_final.unsqueeze(-2));
  }

  attn = attn +
         torch::eye(
             chunk_size,
             torch::TensorOptions().dtype(attn.dtype()).device(attn.device()));
  value = torch::matmul(attn, v_beta);
  torch::Tensor k_cumdecay =
      torch::matmul(attn, (k_beta * g.exp().unsqueeze(-1)));
  torch::Tensor last_recurrent_state;
  if (!initial_state.has_value()) {
    last_recurrent_state = torch::zeros(
        {batch_size, num_v_heads, k_head_dim, v_head_dim},
        torch::TensorOptions().dtype(value.dtype()).device(value.device()));
  } else {
    last_recurrent_state = initial_state.value().to(value);
  }
  torch::Tensor core_attn_out = torch::zeros_like(value);
  mask = torch::triu(
      torch::ones(
          {chunk_size, chunk_size},
          torch::TensorOptions().dtype(torch::kBool).device(query.device())),
      1);
  int64_t num_chunks = total_sequence_length / chunk_size;
  for (int64_t i = 0; i < num_chunks; ++i) {
    torch::Tensor q_i = query.select(2, i);
    torch::Tensor k_i = key.select(2, i);
    torch::Tensor v_i = value.select(2, i);
    torch::Tensor attn_i =
        (torch::matmul(q_i, k_i.transpose(-1, -2)) * decay_mask.select(2, i))
            .masked_fill_(mask, 0.0);
    torch::Tensor v_prime =
        torch::matmul(k_cumdecay.select(2, i), last_recurrent_state);
    torch::Tensor v_new = v_i - v_prime;
    torch::Tensor attn_inter = torch::matmul(
        q_i * g.select(2, i).unsqueeze(-1).exp(), last_recurrent_state);
    core_attn_out.select(2, i) = attn_inter + torch::matmul(attn_i, v_new);
    torch::Tensor g_i_last = g.select(2, i).select(-1, -1).unsqueeze(-1);
    torch::Tensor g_exp_term = (g_i_last - g.select(2, i)).exp().unsqueeze(-1);
    torch::Tensor k_g_exp = (k_i * g_exp_term).transpose(-1, -2).contiguous();
    last_recurrent_state = last_recurrent_state * g_i_last.unsqueeze(-1).exp() +
                           torch::matmul(k_g_exp, v_new);
  }
  auto core_attn_out_shape = core_attn_out.sizes();
  std::vector<int64_t> reshape_shape = {
      core_attn_out_shape[0],
      core_attn_out_shape[1],
      core_attn_out_shape[2] * core_attn_out_shape[3],
      core_attn_out_shape[4]};
  core_attn_out = core_attn_out.reshape(reshape_shape);
  core_attn_out = core_attn_out.slice(2, 0, sequence_length);
  core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype);
  return std::make_tuple(core_attn_out, last_recurrent_state);
}

torch::Tensor invert_unit_lower_triangular(const torch::Tensor& lower) {
  CHECK_GE(lower.dim(), 2)
      << "invert_unit_lower_triangular expects at least 2 dims.";
  CHECK_EQ(lower.size(-1), lower.size(-2))
      << "invert_unit_lower_triangular expects square matrices.";

  int64_t size = lower.size(-1);
  torch::Tensor inverse = torch::zeros_like(lower);
  inverse.diagonal(0, -2, -1).fill_(1.0f);
  for (int64_t row_idx = 1; row_idx < size; ++row_idx) {
    torch::Tensor lower_row =
        lower.select(-2, row_idx).slice(-1, 0, row_idx).unsqueeze(-2);
    torch::Tensor inverse_prefix =
        inverse.slice(-2, 0, row_idx).slice(-1, 0, row_idx);
    inverse.select(-2, row_idx)
        .slice(-1, 0, row_idx)
        .copy_(-torch::matmul(lower_row, inverse_prefix).squeeze(-2));
  }
  return inverse;
}

std::tuple<torch::Tensor, torch::Tensor> recurrent_gated_delta_rule_impl(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor g,
    torch::Tensor beta,
    const std::optional<torch::Tensor>& initial_state,
    const std::optional<float>& scale_value,
    bool use_qk_l2norm_in_kernel) {
  torch::ScalarType initial_dtype = query.scalar_type();

  if (use_qk_l2norm_in_kernel) {
    query = l2norm(query, -1, 1e-6);
    key = l2norm(key, -1, 1e-6);
  }

  auto to_float32_and_transpose = [](torch::Tensor x) {
    return x.transpose(1, 2).contiguous().to(torch::kFloat32);
  };
  query = to_float32_and_transpose(query);
  key = to_float32_and_transpose(key);
  value = to_float32_and_transpose(value);
  beta = to_float32_and_transpose(beta);
  g = to_float32_and_transpose(g);

  int64_t batch_size = key.size(0);
  int64_t num_k_heads = key.size(1);
  int64_t num_v_heads = value.size(1);
  int64_t sequence_length = key.size(2);
  int64_t k_head_dim = key.size(3);
  int64_t v_head_dim = value.size(3);
  TORCH_CHECK(num_k_heads > 0, "num_k_heads must be positive.");
  TORCH_CHECK(num_v_heads > 0, "num_v_heads must be positive.");
  TORCH_CHECK(num_v_heads % num_k_heads == 0,
              "num_v_heads must be divisible by num_k_heads, got num_v_heads=",
              num_v_heads,
              ", num_k_heads=",
              num_k_heads);
  int64_t kv_head_ratio = num_v_heads / num_k_heads;

  float scale_val = scale_value.has_value()
                        ? scale_value.value()
                        : 1.0 / std::sqrt(static_cast<float>(query.size(-1)));
  torch::Tensor scale = torch::tensor(scale_val, query.options());
  query = query * scale;

  torch::Tensor core_attn_out = torch::zeros(
      {batch_size, num_v_heads, sequence_length, v_head_dim},
      torch::TensorOptions().dtype(torch::kFloat32).device(value.device()));
  torch::Tensor last_recurrent_state;
  if (!initial_state.has_value()) {
    last_recurrent_state = torch::zeros(
        {batch_size, num_v_heads, k_head_dim, v_head_dim},
        torch::TensorOptions().dtype(torch::kFloat32).device(value.device()));
  } else {
    last_recurrent_state =
        initial_state.value().to(value.device(), torch::kFloat32);
  }

  for (int64_t i = 0; i < sequence_length; ++i) {
    torch::Tensor q_t = query.select(2, i);
    torch::Tensor k_t = key.select(2, i);
    if (kv_head_ratio > 1) {
      q_t = q_t.repeat_interleave(kv_head_ratio, 1);
      k_t = k_t.repeat_interleave(kv_head_ratio, 1);
    }
    torch::Tensor v_t = value.select(2, i);
    torch::Tensor g_t = g.select(2, i).exp().unsqueeze(-1).unsqueeze(-1);
    torch::Tensor beta_t = beta.select(2, i).unsqueeze(-1);
    last_recurrent_state = last_recurrent_state * g_t;
    torch::Tensor kv_mem =
        torch::sum(last_recurrent_state * k_t.unsqueeze(-1), -2);
    torch::Tensor delta = (v_t - kv_mem) * beta_t;
    last_recurrent_state =
        last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2);
    core_attn_out.select(2, i) =
        torch::sum(last_recurrent_state * q_t.unsqueeze(-1), -2);
  }

  core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype);
  return std::make_tuple(core_attn_out, last_recurrent_state);
}

std::tuple<torch::Tensor, torch::Tensor> causal_conv1d_ref_single(
    torch::Tensor x,
    torch::Tensor weight,
    const std::optional<torch::Tensor>& bias,
    const std::optional<torch::Tensor>& initial_state,
    bool activation) {
  torch::ScalarType input_dtype = x.scalar_type();
  int64_t width = weight.size(1);
  int64_t state_len = width - 1;
  torch::Tensor x_conv = x.to(weight.dtype()).contiguous();
  torch::Tensor prefix =
      initial_state.has_value()
          ? initial_state.value().to(x_conv.dtype()).contiguous()
          : torch::zeros({x.size(0), x.size(1), state_len}, x_conv.options());
  torch::Tensor padded_input = torch::cat({prefix, x_conv}, -1).contiguous();

  std::vector<torch::Tensor> windows;
  windows.reserve(width);
  for (int64_t tap_idx = 0; tap_idx < width; ++tap_idx) {
    windows.push_back(
        padded_input.slice(-1, tap_idx, tap_idx + x.size(-1)).contiguous());
  }
  torch::Tensor stacked_windows = torch::stack(windows, -1);
  torch::Tensor out = torch::sum(
      stacked_windows * weight.view({1, weight.size(0), 1, width}), -1);
  if (bias.has_value()) {
    out = out + bias.value().view({1, bias.value().size(0), 1});
  }
  if (activation) {
    out = torch::silu(out);
  }

  torch::Tensor final_state =
      padded_input
          .slice(-1, padded_input.size(-1) - state_len, padded_input.size(-1))
          .to(input_dtype)
          .contiguous();
  return std::make_tuple(out.to(input_dtype), final_state);
}

}  // namespace

torch::Tensor causal_conv1d(
    torch::Tensor& x,
    torch::Tensor& weight,
    const std::optional<torch::Tensor>& bias,
    const std::optional<torch::Tensor>& seq_lens,
    std::optional<torch::Tensor> conv_state_source,
    const std::optional<torch::Tensor>& conv_state_indices,
    const std::optional<torch::Tensor>& has_initial_state,
    bool activation,
    int64_t pad_slot_id) {
  CHECK(x.dim() == 3) << "causal_conv1d expects x to be [B, D, T].";
  CHECK(weight.dim() == 2) << "causal_conv1d expects weight to be [D, W].";
  CHECK_EQ(x.size(1), weight.size(0)) << "causal_conv1d channel size mismatch.";

  int64_t batch_size = x.size(0);
  int64_t max_seq_len = x.size(2);
  torch::Tensor output = torch::zeros_like(x);
  torch::Tensor weight_contiguous = weight.contiguous();

  for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    int64_t valid_len = seq_lens.has_value()
                            ? seq_lens.value()[batch_idx].item<int64_t>()
                            : max_seq_len;
    if (valid_len <= 0) {
      continue;
    }

    int64_t cache_idx = batch_idx;
    if (conv_state_indices.has_value()) {
      cache_idx = conv_state_indices.value()[batch_idx].item<int64_t>();
      if (cache_idx == pad_slot_id) {
        continue;
      }
    }

    std::optional<torch::Tensor> initial_state = std::nullopt;
    if (conv_state_source.has_value() && has_initial_state.has_value() &&
        has_initial_state.value()[batch_idx].item<bool>()) {
      initial_state = conv_state_source.value()
                          .index({cache_idx})
                          .unsqueeze(0)
                          .contiguous();
    }

    torch::Tensor x_slice =
        x.index({batch_idx}).slice(-1, 0, valid_len).unsqueeze(0).contiguous();
    auto [out_slice, final_state] = causal_conv1d_ref_single(
        x_slice, weight_contiguous, bias, initial_state, activation);

    output.index_put_({batch_idx,
                       torch::indexing::Slice(),
                       torch::indexing::Slice(0, valid_len)},
                      out_slice.squeeze(0));
    if (conv_state_source.has_value()) {
      conv_state_source.value().index_put_(
          {cache_idx},
          final_state.squeeze(0).to(conv_state_source.value().scalar_type()));
    }
  }
  return output;
}

std::tuple<torch::Tensor, torch::Tensor> chunk_gated_delta_rule(
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& v,
    torch::Tensor& g,
    torch::Tensor& beta,
    const std::optional<torch::Tensor>& seq_lens,
    int64_t chunk_size,
    const std::optional<torch::Tensor>& initial_state,
    bool output_final_state,
    bool use_qk_l2norm_in_kernel) {
  (void)chunk_size;
  CHECK(q.dim() == 4) << "chunk_gated_delta_rule expects q to be [B, T, H, K].";
  CHECK_EQ(q.size(0), k.size(0))
      << "chunk_gated_delta_rule q/k batch mismatch.";
  CHECK_EQ(q.size(1), k.size(1)) << "chunk_gated_delta_rule q/k seq mismatch.";
  CHECK_EQ(q.size(2), k.size(2)) << "chunk_gated_delta_rule q/k head mismatch.";
  CHECK_EQ(q.size(0), v.size(0)) << "chunk_gated_delta_rule batch mismatch.";
  CHECK_EQ(q.size(1), v.size(1)) << "chunk_gated_delta_rule seq mismatch.";

  int64_t batch_size = q.size(0);
  int64_t max_seq_len = q.size(1);
  int64_t num_v_heads = v.size(2);
  int64_t v_head_dim = v.size(3);
  int64_t k_head_dim = k.size(3);
  torch::Tensor output = torch::zeros(
      {batch_size, max_seq_len, num_v_heads, v_head_dim}, v.options());
  torch::Tensor final_state;
  if (output_final_state) {
    final_state = torch::zeros(
        {batch_size, num_v_heads, k_head_dim, v_head_dim},
        torch::TensorOptions().dtype(torch::kFloat32).device(v.device()));
  }

  for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    int64_t valid_len = seq_lens.has_value()
                            ? seq_lens.value()[batch_idx].item<int64_t>()
                            : max_seq_len;
    if (valid_len <= 0) {
      continue;
    }

    std::optional<torch::Tensor> initial_state_slice = std::nullopt;
    if (initial_state.has_value()) {
      initial_state_slice =
          initial_state.value().index({batch_idx}).unsqueeze(0).contiguous();
    }
    auto q_slice =
        q.index({batch_idx}).slice(0, 0, valid_len).unsqueeze(0).contiguous();
    auto k_slice =
        k.index({batch_idx}).slice(0, 0, valid_len).unsqueeze(0).contiguous();
    auto v_slice =
        v.index({batch_idx}).slice(0, 0, valid_len).unsqueeze(0).contiguous();
    auto g_slice =
        g.index({batch_idx}).slice(0, 0, valid_len).unsqueeze(0).contiguous();
    auto beta_slice = beta.index({batch_idx})
                          .slice(0, 0, valid_len)
                          .unsqueeze(0)
                          .contiguous();
    auto chunk_result = chunk_gated_delta_rule_impl(q_slice,
                                                    k_slice,
                                                    v_slice,
                                                    g_slice,
                                                    beta_slice,
                                                    chunk_size,
                                                    initial_state_slice,
                                                    use_qk_l2norm_in_kernel);
    auto [out_slice, final_state_slice] = std::move(chunk_result);

    output.index_put_({batch_idx, torch::indexing::Slice(0, valid_len)},
                      out_slice.squeeze(0));
    if (output_final_state) {
      final_state.index_put_({batch_idx}, final_state_slice.squeeze(0));
    }
  }

  if (!output_final_state) {
    final_state = torch::Tensor();
  }
  return std::make_tuple(output, final_state);
}

torch::Tensor fused_sigmoid_gating_delta_rule_update(
    torch::Tensor& A_log,
    torch::Tensor& a,
    torch::Tensor& dt_bias,
    float softplus_beta,
    float softplus_threshold,
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& v,
    torch::Tensor& b,
    torch::Tensor& initial_state_source,
    torch::Tensor& initial_state_indices,
    const std::optional<float>& scale,
    bool use_qk_l2norm_in_kernel,
    const std::optional<torch::Tensor>& cu_seqlens) {
  CHECK(initial_state_indices.is_contiguous())
      << "initial_state_indices must be contiguous.";
  if (cu_seqlens.has_value() &&
      !use_reference_fused_sigmoid_gating_delta_rule_update() &&
      enable_qwen35_fast_fused_sigmoid_gdn_update()) {
    const int64_t original_batch = q.size(0);
    const int64_t original_seq = q.size(1);
    torch::Tensor packed_q = q;
    torch::Tensor packed_k = k;
    torch::Tensor packed_v = v;
    torch::Tensor packed_a = a;
    torch::Tensor packed_b = b;
    const bool need_pack_batch = original_batch > 1;
    if (need_pack_batch) {
      const int64_t total_tokens = original_batch * original_seq;
      // Keep the Triton entry aligned with vllm-ascend varlen decode:
      // q/k/v are packed as [1, total_tokens, ...] and cu_seqlens maps tokens.
      packed_q =
          q.contiguous().reshape({1, total_tokens, q.size(2), q.size(3)});
      packed_k =
          k.contiguous().reshape({1, total_tokens, k.size(2), k.size(3)});
      packed_v =
          v.contiguous().reshape({1, total_tokens, v.size(2), v.size(3)});
      packed_a = a.contiguous().reshape({1, total_tokens, a.size(2)});
      packed_b = b.contiguous().reshape({1, total_tokens, b.size(2)});
    }
    if (should_log_qwen35_fast_fused_sigmoid_gdn_update_hit()) {
      LOG(INFO) << "Qwen3.5 decode fused_sigmoid_gating_delta_rule_update "
                   "uses fused_sigmoid_gating_delta_rule_update_kernel.";
    }
    auto output =
        npu_fused_sigmoid_gating_delta_rule_update(A_log,
                                                   packed_a,
                                                   dt_bias,
                                                   softplus_beta,
                                                   softplus_threshold,
                                                   packed_q,
                                                   packed_k,
                                                   packed_v,
                                                   packed_b,
                                                   initial_state_source,
                                                   initial_state_indices,
                                                   scale,
                                                   use_qk_l2norm_in_kernel,
                                                   cu_seqlens);
    if (need_pack_batch) {
      output = output.contiguous().reshape(
          {original_batch, original_seq, v.size(2), v.size(3)});
    }
    return output;
  }
  if (cu_seqlens.has_value() &&
      should_log_qwen35_fast_fused_sigmoid_gdn_update_fallback()) {
    LOG(INFO) << "Qwen3.5 decode fused_sigmoid_gating_delta_rule_update "
                 "fell back to recurrent reference path."
              << " fast_enabled="
              << enable_qwen35_fast_fused_sigmoid_gdn_update() << ", debug_ref="
              << use_reference_fused_sigmoid_gating_delta_rule_update();
  }

  torch::Tensor beta = torch::sigmoid(b);
  torch::Tensor a_plus_dt = a.to(torch::kFloat32) + dt_bias;
  torch::Tensor g =
      -A_log.exp() * torch::nn::functional::softplus(
                         a_plus_dt,
                         torch::nn::functional::SoftplusFuncOptions()
                             .beta(softplus_beta)
                             .threshold(softplus_threshold));
  g = g.to(a.dtype()).contiguous();
  torch::Tensor initial_state =
      torch::index_select(initial_state_source, 0, initial_state_indices)
          .contiguous();
  auto [output, final_state] = recurrent_gated_delta_rule_impl(
      q, k, v, g, beta, initial_state, scale, use_qk_l2norm_in_kernel);
  initial_state_source.index_put_({initial_state_indices},
                                  final_state.to(initial_state_source.dtype()));
  return output;
}

}  // namespace xllm::kernel::npu
