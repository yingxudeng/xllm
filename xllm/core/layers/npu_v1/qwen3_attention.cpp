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

#include "qwen3_attention.h"

#include <glog/logging.h>

#include <tuple>

#include "core/kernels/linear.h"
#include "core/kernels/rms_norm.h"
#include "core/kernels/rope.h"
#include "core/kernels/split.h"

namespace xllm {
namespace layer {

void print_first_5_4(const torch::Tensor& tensor, const std::string& name) {
  try {
    std::stringstream ss;
    auto float_tensor = tensor.to(torch::kCPU, torch::kFloat);
    auto flattened = float_tensor.flatten();
    int64_t size = flattened.size(0);

    ss << std::setprecision(4) << std::fixed;
    ss << name << ":\n";

    ss << "  Shape: " << tensor.sizes() << "\n";
    ss << "  Data Type: " << tensor.dtype() << "\n";
    ss << "  Device: " << tensor.device() << "\n";

    // 打印前4个元素
    ss << "  First 4 elements: ";
    for (int i = 0; i < std::min(static_cast<int64_t>(4), size); ++i) {
      ss << flattened[i].item<float>() << " ";
    }
    ss << "\n";

    // 打印后4个元素
    ss << "  Last 4 elements:  ";
    for (int i = std::max(static_cast<int64_t>(0), size - 4); i < size; ++i) {
      ss << flattened[i].item<float>() << " ";
    }
    ss << "\n";

    // 计算统计值
    if (size > 0) {
      float max_val = flattened.max().item<float>();
      float min_val = flattened.min().item<float>();
      float mean_val = flattened.mean().item<float>();

      ss << std::fixed << std::setprecision(8);
      ss << "  Max: " << max_val << "\n";
      ss << "  Min: " << min_val << "\n";
      ss << "  Mean: " << mean_val << "\n";
    } else {
      ss << "  Tensor is empty\n";
    }
    ss << std::endl;

    std::cout << ss.str();
    std::cout.flush();
  } catch (const c10::Error& e) {
    std::cerr << "PyTorch Error in print_first_5_1: " << e.what() << std::endl;
  } catch (const std::runtime_error& e) {
    std::cerr << "Runtime Error in print_first_5_1: " << e.what() << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Standard Exception in print_first_5_1: " << e.what()
              << std::endl;
  } catch (...) {
    std::cerr << "Unknown Error in print_first_5_1" << std::endl;
  }
}

Qwen3AttentionImpl::Qwen3AttentionImpl(const ModelArgs& args,
                                       const QuantArgs& quant_args,
                                       const ParallelArgs& parallel_args,
                                       const torch::TensorOptions& options,
                                       const ModelContext& context) {
  const int64_t tp_size = parallel_args.world_size();
  const int64_t total_num_heads = args.n_heads();
  const int64_t total_num_kv_heads = args.n_kv_heads().value_or(args.n_heads());

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
  scaling_ = std::sqrt(1.0f / head_dim_);

  // 1. QKV parallel linear
  qkv_proj_ = register_module("qkv_proj",
                              QKVParallelLinear(args.hidden_size(),
                                                num_heads_,
                                                num_kv_heads_,
                                                args.head_dim(),
                                                num_kv_head_replicas_,
                                                /*bias=*/false,
                                                /*gather_output=*/false,
                                                parallel_args,
                                                options));

  // 2. Output projection
  o_proj_ = register_module("o_proj",
                            RowParallelLinear(total_num_heads * args.head_dim(),
                                              args.hidden_size(),
                                              /*bias=*/false,
                                              /*input_is_parallelized=*/true,
                                              /*if_reduce_results=*/true,
                                              parallel_args,
                                              options));

  // 3. RMSNorm
  std::cerr << "dyx-debug args.head_dim(): " << args.head_dim() << std::endl;
  std::cerr << "dyx-debug args.rms_norm_eps(): " << args.rms_norm_eps()
            << std::endl;
  q_norm_ = register_module(
      "q_norm", RmsNormV1(args.head_dim(), args.rms_norm_eps(), options));
  // q_norm_ = register_module("q_norm", xllm::kernel::RmsNorm(context));

  k_norm_ = register_module(
      "k_norm", RmsNormV1(args.head_dim(), args.rms_norm_eps(), options));
  // k_norm_ = register_module("k_norm", xllm::kernel::RmsNorm(context));

  // 4. Rotary embedding
  rotary_emb_ = register_module("rope",
                                RotaryEmbedding(/*rotary_dim=*/head_dim_,
                                                args.max_position_embeddings(),
                                                args.rope_theta(),
                                                /*interleaved=*/false,
                                                options));
  // rotary_emb_ = register_module("rope", xllm::kernel::Rope(context));

  // 5. Attention
  attn_ = register_module("attn",
                          Attention(num_heads_,
                                    head_dim_,
                                    scaling_,
                                    num_kv_heads_,
                                    args.sliding_window()));
}

torch::Tensor Qwen3AttentionImpl::forward(
    const torch::Tensor& positions,
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache) {
  // 1. qkv projection
  auto qkv = qkv_proj_->forward(hidden_states);
  std::cout << "After qkv projection - qkv shape: " << qkv.sizes() << std::endl;
  print_first_5_4(qkv, "Qwen3AttentionImpl::forward qkv");

  auto q = qkv.slice(/*dim=*/-1, 0, q_size_);
  auto k = qkv.slice(/*dim=*/-1, q_size_, q_size_ + kv_size_);
  auto v = qkv.slice(/*dim=*/-1, q_size_ + kv_size_, q_size_ + 2 * kv_size_);

  std::cout << "After slicing - q shape: " << q.sizes() << std::endl;
  std::cout << "After slicing - k shape: " << k.sizes() << std::endl;
  std::cout << "After slicing - v shape: " << v.sizes() << std::endl;
  print_first_5_4(q, "Qwen3AttentionImpl::forward q before norm");
  print_first_5_4(k, "Qwen3AttentionImpl::forward k before norm");
  print_first_5_4(v, "Qwen3AttentionImpl::forward v before norm");
  const int64_t T = q.size(0);
  std::cout << "Sequence length T: " << T << std::endl;

  auto q_by_head = q.view({q.size(0), q.size(-1) / head_dim_, head_dim_});
  auto k_by_head = k.view({k.size(0), k.size(-1) / head_dim_, head_dim_});
  // 2. q-norm
  std::cout << "Before q_norm - q shape: " << q.sizes() << std::endl;
  q_by_head = q_norm_->forward(q_by_head);
  q = q_by_head.view(q.sizes());
  std::cout << "After q_norm - q shape: " << q.sizes() << std::endl;
  print_first_5_4(q, "Qwen3AttentionImpl::forward q after q_norm");
  // 3. k-norm
  std::cout << "Before k_norm - k shape: " << k.sizes() << std::endl;
  k_by_head = k_norm_->forward(k_by_head);
  k = k_by_head.view(k.sizes());
  std::cout << "After k_norm - k shape: " << k.sizes() << std::endl;
  print_first_5_4(k, "Qwen3AttentionImpl::forward k after k_norm");

  // 4. rope
  rotary_emb_->forward(q,
                       k,
                       positions,
                       attn_metadata.query_start_loc,
                       attn_metadata.max_query_len,
                       attn_metadata.is_prefill);
  // q = q.view({T, q_size_});
  // k = k.view({T, kv_size_});
  std::cerr << "dyx-debug after rotary_emb q" << q.sizes() << std::endl;
  std::cerr << "dyx-debug after rotary_emb k" << k.sizes() << std::endl;
  // auto sizes = q.sizes();

  // std::vector<int64_t> new_shape(sizes.begin(), sizes.end() - 1);
  // new_shape.push_back(head_dim_);
  // new_shape.push_back(head_dim_);

  std::cout << "dyx-debug Before attention - q shape: " << q.sizes()
            << ", k shape: " << k.sizes() << ", v shape: " << v.sizes()
            << std::endl;
  // << ", kv_cache shape: " << kv_cache.sizes() << std::endl;
  print_first_5_4(q, "Qwen3AttentionImpl::forward q before attention");
  print_first_5_4(k, "Qwen3AttentionImpl::forward k before attention");
  print_first_5_4(v, "Qwen3AttentionImpl::forward v before attention");
  // q = q.view(sizes);
  // 5. store k/v cache and do attention
  auto out = std::get<0>(attn_->forward(attn_metadata, q, k, v, kv_cache));
  print_first_5_4(out, "Qwen3AttentionImpl::forward out before o_proj");
  // 6. output projection
  auto output_last = o_proj_->forward(out);
  print_first_5_4(output_last, "Qwen3AttentionImpl::forward output_last");
  return output_last;
}

void Qwen3AttentionImpl::load_state_dict(const StateDict& state_dict) {
  std::cerr << "dyx-debug Loading Qwen3AttentionImpl::load_state_dict"
            << std::endl;
  std::cerr << "dyx-debug StateDict Info:" << std::endl;
  std::cerr << "  - prefix: '" << state_dict.prefix() << "'" << std::endl;
  std::cerr << "  - size: " << state_dict.size() << " tensors" << std::endl;

  std::cerr << "  - keys: ";
  for (const auto& [key, tensor] : state_dict) {
    std::cerr << "  Key: '" << key << "'"
              << " | Shape: " << tensor.sizes()
              << " | Dtype: " << tensor.dtype()
              << " | Device: " << tensor.device() << std::endl;
  }

  qkv_proj_->load_state_dict(state_dict);
  o_proj_->load_state_dict(state_dict.get_dict_with_prefix("o_proj."));
  std::cerr << "dyx-debug Loading Q Norm Weights" << std::endl;
  if (auto w = state_dict.get_tensor("q_norm.weight"); w.defined()) {
    q_norm_->load_state_dict(StateDict({{"weight", w}}));
  }
  std::cerr << "dyx-debug Loading K Norm Weights" << std::endl;
  if (auto w = state_dict.get_tensor("k_norm.weight"); w.defined()) {
    k_norm_->load_state_dict(StateDict({{"weight", w}}));
  }
}

}  // namespace layer
}  // namespace xllm
