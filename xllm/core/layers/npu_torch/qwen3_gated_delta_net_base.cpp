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

#include "qwen3_gated_delta_net_base.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <tuple>

#include "xllm/core/kernels/ops_api.h"

namespace xllm {
namespace layer {

Qwen3GatedDeltaNetBaseImpl::Qwen3GatedDeltaNetBaseImpl(
    const ModelArgs& args,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options) {
  tp_size_ = parallel_args.tp_group_->world_size();
  rank_ = parallel_args.tp_group_->rank();
  num_k_heads_ = args.linear_num_key_heads();
  num_v_heads_ = args.linear_num_value_heads();
  head_k_dim_ = args.linear_key_head_dim();
  head_v_dim_ = args.linear_value_head_dim();
  k_size_ = num_k_heads_ * head_k_dim_;
  v_size_ = num_v_heads_ * head_v_dim_;
  conv_kernel_size_ = args.linear_conv_kernel_dim();

  // Shared causal conv projection over mixed QKV states.
  conv1d_ = register_module("conv1d",
                            ColumnParallelLinear(args.linear_conv_kernel_dim(),
                                                 k_size_ * 2 + v_size_,
                                                 /*bias=*/false,
                                                 /*gather_output=*/false,
                                                 quant_args,
                                                 parallel_args.tp_group_,
                                                 options));

  auto opts = options.dtype(torch::kFloat32);
  dt_bias_ = register_parameter("dt_bias",
                                torch::ones({num_v_heads_ / tp_size_}, opts),
                                /*requires_grad=*/false);

  A_log_ = register_parameter("A_log",
                              torch::empty({num_v_heads_ / tp_size_}, opts),
                              /*requires_grad=*/false);

  // Output projection and gated RMSNorm shared by hybrid variants.
  o_proj_ = register_module("out_proj",
                            RowParallelLinear(v_size_,
                                              args.hidden_size(),
                                              /*bias=*/false,
                                              /*input_is_parallelized=*/true,
                                              /*if_reduce_results=*/true,
                                              quant_args,
                                              parallel_args.tp_group_,
                                              options));

  norm_ = register_module(
      "norm", RmsNormGated(head_v_dim_, args.rms_norm_eps(), options));
}

void Qwen3GatedDeltaNetBaseImpl::load_common_state_dict(
    const StateDict& state_dict) {
  const int64_t rank = rank_;
  const int64_t world_size = tp_size_;
  const int32_t shard_tensor_count = 3;
  const std::vector<int64_t> shard_sizes = {
      k_size_ / tp_size_, k_size_ / tp_size_, v_size_ / tp_size_};

  if (auto w = state_dict.get_tensor("conv1d.weight"); w.defined()) {
    conv1d_->load_state_dict(
        StateDict({{"weight", w.squeeze(1)}}), shard_tensor_count, shard_sizes);
  }
  o_proj_->load_state_dict(state_dict.get_dict_with_prefix("out_proj."));
  if (auto w = state_dict.get_tensor("norm.weight"); w.defined()) {
    norm_->load_state_dict(StateDict({{"weight", w}}));
  }
  LOAD_SHARDED_WEIGHT(dt_bias, 0);
  LOAD_SHARDED_WEIGHT(A_log, 0);
}

void Qwen3GatedDeltaNetBaseImpl::verify_common_loaded_weights(
    const std::string& prefix) const {
  CHECK(dt_bias_is_loaded_)
      << "Missing required weight after all shards loaded: " << prefix
      << "dt_bias";
  CHECK(A_log_is_loaded_) << "Missing required weight after all shards loaded: "
                          << prefix << "A_log";
}

torch::Tensor Qwen3GatedDeltaNetBaseImpl::forward(
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params) {
  auto [qkvz_padded, ba_padded] =
      project_padded_inputs(hidden_states, attn_metadata);

  torch::Tensor q, k, v, z, b, a;
  std::tie(q, k, v, z) = process_qkvz_tensor(qkvz_padded);
  std::tie(b, a) = process_ba_tensor(ba_padded);

  auto rearrange_merge = [](const torch::Tensor& t) {
    TORCH_CHECK(
        t.dim() > 2, "Tensor must have at least 2 dims! but got ", t.dim());
    std::vector<int64_t> new_shape;
    int64_t slice_end = t.dim() - 2;
    auto valid_slice = t.sizes().slice(0, slice_end);
    new_shape = std::vector<int64_t>(valid_slice.begin(), valid_slice.end());
    int64_t last_two_dim = t.size(slice_end) * t.size(slice_end + 1);
    new_shape.push_back(last_two_dim);
    return t.reshape(new_shape);
  };

  q = rearrange_merge(q);
  k = rearrange_merge(k);
  v = rearrange_merge(v);

  // Run the causal conv update on the mixed QKV states.
  torch::Tensor mixed_qkv = torch::cat({q, k, v}, q.dim() - 1);
  mixed_qkv = mixed_qkv.transpose(1, 2);
  torch::Tensor conv_cache = kv_cache.get_conv_cache();
  torch::Tensor ssm_cache = kv_cache.get_ssm_cache();
  torch::Tensor core_attn_out;
  auto device = mixed_qkv.device();
  auto conv_weight = conv1d_->weight();
  torch::Tensor cache_indices;
  torch::Tensor has_initial_state;
  if (attn_metadata.is_prefill) {
    CHECK(input_params.block_tables.defined())
        << "Qwen3.5 GDN prefill requires input_params.block_tables.";
    cache_indices = input_params.block_tables.select(1, 0).contiguous();
    has_initial_state =
        (attn_metadata.kv_seq_lens > attn_metadata.q_seq_lens).contiguous();
  } else {
    CHECK(attn_metadata.block_table.defined())
        << "Qwen3.5 GDN decode requires attn_metadata.block_table.";
    cache_indices = attn_metadata.block_table.select(1, 0)
                        .to(device, torch::kInt32)
                        .contiguous();
  }

  if (attn_metadata.is_prefill) {
    xllm::kernel::CausalConv1dParams conv_params;
    conv_params.x = mixed_qkv;
    conv_params.weight = conv_weight.to(device);
    conv_params.seq_lens = attn_metadata.q_seq_lens;
    conv_params.conv_state_source = conv_cache;
    conv_params.conv_state_indices = cache_indices;
    conv_params.has_initial_state = has_initial_state;
    conv_params.activation = true;
    mixed_qkv = xllm::kernel::causal_conv1d(conv_params);
  } else {
    xllm::kernel::CausalConv1dUpdateParams params;
    params.x = mixed_qkv;
    params.conv_state = conv_cache;
    params.weight = conv_weight;
    params.conv_state_indices = cache_indices;
    mixed_qkv = xllm::kernel::causal_conv1d_update(params);
  }

  auto [processed_q, processed_k, processed_v] = process_mixed_qkv(mixed_qkv);
  if (attn_metadata.is_prefill) {
    torch::Tensor beta = torch::sigmoid(b).contiguous();
    torch::Tensor a_plus_dt = a.to(torch::kFloat32) + dt_bias_.view({1, 1, -1});
    torch::Tensor g =
        -A_log_.exp().view({1, 1, -1}) *
        torch::nn::functional::softplus(
            a_plus_dt,
            torch::nn::functional::SoftplusFuncOptions().beta(1.0f).threshold(
                20.0f));
    g = g.contiguous();

    torch::Tensor initial_state =
        ssm_cache.index_select(0, cache_indices).contiguous();
    initial_state.index_put_({torch::logical_not(has_initial_state)}, 0);
    torch::Tensor last_recurrent_state;
    xllm::kernel::ChunkGatedDeltaRuleParams chunk_params;
    chunk_params.q = processed_q;
    chunk_params.k = processed_k;
    chunk_params.v = processed_v;
    chunk_params.g = g;
    chunk_params.beta = beta;
    chunk_params.seq_lens = attn_metadata.q_seq_lens;
    chunk_params.chunk_size = 64;
    chunk_params.initial_state = initial_state;
    chunk_params.output_final_state = true;
    chunk_params.use_qk_l2norm_in_kernel = true;
    std::tie(core_attn_out, last_recurrent_state) =
        xllm::kernel::chunk_gated_delta_rule(chunk_params);
    ssm_cache.index_put_({cache_indices},
                         last_recurrent_state.to(ssm_cache.dtype()));
  } else {
    xllm::kernel::FusedSigmoidGatingDeltaRuleUpdateParams recurrent_params;
    recurrent_params.A_log = A_log_.contiguous();
    recurrent_params.a = a.contiguous();
    recurrent_params.dt_bias = dt_bias_.contiguous();
    recurrent_params.softplus_beta = 1.0f;
    recurrent_params.softplus_threshold = 20.0f;
    recurrent_params.q = processed_q.contiguous();
    recurrent_params.k = processed_k.contiguous();
    recurrent_params.v = processed_v.contiguous();
    recurrent_params.b = b.contiguous();
    recurrent_params.initial_state_source = ssm_cache;
    recurrent_params.initial_state_indices = cache_indices;
    recurrent_params.use_qk_l2norm_in_kernel = true;
    recurrent_params.cu_seqlens = attn_metadata.q_cu_seq_lens;
    core_attn_out =
        xllm::kernel::fused_sigmoid_gating_delta_rule_update(recurrent_params);
  }

  auto z_reshaped = z.view({-1, z.size(-1)});
  auto core_attn_out_reshaped =
      core_attn_out.view({-1, core_attn_out.size(-1)});
  auto norm_out = norm_->forward(core_attn_out_reshaped, z_reshaped);
  auto z_shape_og = z.sizes().vec();
  norm_out = norm_out.view(z_shape_og);
  norm_out = norm_out.view({-1, norm_out.size(2), norm_out.size(3)});

  // Project the normalized attention output back to hidden size.
  auto rearranged_norm = rearrange_merge(norm_out);
  rearranged_norm = reshape_qkvz_unpad(attn_metadata, rearranged_norm);
  return o_proj_->forward(rearranged_norm);
}

torch::Tensor Qwen3GatedDeltaNetBaseImpl::reshape_qkvz_unpad(
    const AttentionMetadata& attn_metadata,
    const torch::Tensor& padded_qkvz) const {
  if (!attn_metadata.is_prefill) {
    return padded_qkvz;
  }
  std::vector<torch::Tensor> valid_batches;
  int64_t bs = attn_metadata.q_seq_lens.size(0);
  int64_t max_len = attn_metadata.max_query_len;
  const auto& ori_seq_lens = attn_metadata.q_seq_lens;
  auto reshaped_qkvz = padded_qkvz.view({bs, max_len, -1});
  for (int64_t b = 0; b < bs; ++b) {
    int64_t ori_len = ori_seq_lens[b].template item<int64_t>();
    torch::Tensor valid_batch = reshaped_qkvz[b].slice(0, 0, ori_len);
    valid_batches.push_back(valid_batch);
  }
  return torch::cat(valid_batches, 0).contiguous();
}

torch::Tensor Qwen3GatedDeltaNetBaseImpl::reshape_qkvz_with_pad(
    const AttentionMetadata& attn_metadata,
    const torch::Tensor& qkvz) const {
  int64_t bs = attn_metadata.q_seq_lens.size(0);
  int64_t max_len = attn_metadata.max_query_len;
  const auto& start_loc = attn_metadata.q_seq_lens;
  if (!attn_metadata.is_prefill) {
    return qkvz.view({bs, -1, qkvz.size(-1)});
  }
  std::vector<torch::Tensor> batches;
  int64_t idx = 0;
  for (int64_t b = 0; b < bs; ++b) {
    int64_t cur_len = start_loc[b].template item<int64_t>();
    torch::Tensor batch = qkvz.slice(0, idx, idx + cur_len).contiguous();
    idx = idx + cur_len;
    if (batch.size(0) != max_len) {
      batch = batch.size(0) > max_len
                  ? batch.slice(0, 0, max_len).contiguous()
                  : torch::nn::functional::pad(
                        batch,
                        torch::nn::functional::PadFuncOptions(
                            {0, 0, 0, max_len - batch.size(0)}))
                        .contiguous();
    }
    batches.push_back(batch);
  }
  auto ret = torch::stack(batches, 0).contiguous();
  return ret;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
Qwen3GatedDeltaNetBaseImpl::process_mixed_qkv(torch::Tensor& mixed_qkv) const {
  mixed_qkv = mixed_qkv.transpose(1, 2);
  int64_t batch_size = mixed_qkv.size(0);
  int64_t seq_len = mixed_qkv.size(1);
  std::vector<int64_t> split_sizes = {
      k_size_ / tp_size_, k_size_ / tp_size_, v_size_ / tp_size_};
  auto processed_qkv = torch::split(mixed_qkv, split_sizes, 2);
  auto processed_q = processed_qkv[0];
  auto processed_k = processed_qkv[1];
  auto processed_v = processed_qkv[2];
  processed_q = processed_q.view(
      {batch_size, seq_len, num_k_heads_ / tp_size_, head_k_dim_});
  processed_k = processed_k.view(
      {batch_size, seq_len, num_k_heads_ / tp_size_, head_k_dim_});
  processed_v = processed_v.view(
      {batch_size, seq_len, num_v_heads_ / tp_size_, head_v_dim_});
  return std::make_tuple(processed_q, processed_k, processed_v);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
Qwen3GatedDeltaNetBaseImpl::process_qkvz_tensor(
    const torch::Tensor& qkvz) const {
  std::vector<int64_t> new_tensor_shape_qkvz = [&]() {
    std::vector<int64_t> dims;
    dims.push_back(qkvz.size(0));
    dims.push_back(qkvz.size(1));
    int64_t dim1 = num_k_heads_ / tp_size_;
    int64_t dim2 = head_k_dim_ + head_k_dim_ +
                   (head_v_dim_ + head_v_dim_) * num_v_heads_ / num_k_heads_;
    dims.push_back(dim1);
    dims.push_back(dim2);
    return dims;
  }();

  auto reshaped_qkvz = qkvz.view(new_tensor_shape_qkvz);
  auto qkvz_split = torch::split(reshaped_qkvz,
                                 {head_k_dim_,
                                  head_k_dim_,
                                  num_v_heads_ / num_k_heads_ * head_v_dim_,
                                  num_v_heads_ / num_k_heads_ * head_v_dim_},
                                 reshaped_qkvz.dim() - 1);

  auto q = qkvz_split[0].contiguous();
  auto k = qkvz_split[1].contiguous();
  auto v = qkvz_split[2].contiguous();
  auto z = qkvz_split[3].contiguous();

  v = v.reshape({v.size(0), v.size(1), num_v_heads_ / tp_size_, head_v_dim_});
  z = z.reshape({z.size(0), z.size(1), num_v_heads_ / tp_size_, head_v_dim_});

  return std::make_tuple(q, k, v, z);
}

std::tuple<torch::Tensor, torch::Tensor>
Qwen3GatedDeltaNetBaseImpl::process_ba_tensor(const torch::Tensor& ba) const {
  std::vector<int64_t> new_tensor_shape_ba = [&]() {
    std::vector<int64_t> dims;
    dims.push_back(ba.size(0));
    dims.push_back(ba.size(1));
    int64_t dim1 = num_k_heads_ / tp_size_;
    int64_t dim2 = 2 * num_v_heads_ / num_k_heads_;
    dims.push_back(dim1);
    dims.push_back(dim2);
    return dims;
  }();

  auto reshaped_ba = ba.view(new_tensor_shape_ba);
  auto ba_split =
      torch::split(reshaped_ba,
                   {num_v_heads_ / num_k_heads_, num_v_heads_ / num_k_heads_},
                   reshaped_ba.dim() - 1);

  auto b = ba_split[0].contiguous();
  auto a = ba_split[1].contiguous();

  b = b.reshape({b.size(0), b.size(1), num_v_heads_ / tp_size_});
  a = a.reshape({a.size(0), a.size(1), num_v_heads_ / tp_size_});

  return std::make_tuple(b, a);
}

}  // namespace layer
}  // namespace xllm
