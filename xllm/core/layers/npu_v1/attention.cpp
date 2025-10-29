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

#include "attention.h"

#include "core/kernels/npu_v1/ops_npu/PagedAttentionAtb.h"
#include "core/kernels/npu_v1/ops_npu/ReshapeAndCachAtb.h"
#include "core/kernels/npu_v1/ops_npu/SelfAttentionAtb.h"
DECLARE_bool(enable_chunked_prefill);
namespace xllm {
namespace layer {

void print_first_5_2(const torch::Tensor& tensor, const std::string& name) {
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

AttentionMetadata AttentionMetadata::build(const ModelInputParams& params,
                                           bool is_prefill,
                                           const torch::Tensor& attn_mask) {
  return AttentionMetadata::build(params, "float", is_prefill, attn_mask);
}

AttentionMetadata AttentionMetadata::build(const ModelInputParams& params,
                                           const std::string& compute_dtype,
                                           bool is_prefill,
                                           const torch::Tensor& attn_mask) {
  AttentionMetadata attn_metadata;
  attn_metadata.query_start_loc = params.q_seq_lens;
  attn_metadata.seq_start_loc = params.kv_seq_lens;
  attn_metadata.max_query_len = params.q_max_seq_len;
  attn_metadata.max_seq_len = params.kv_max_seq_len;
  attn_metadata.slot_mapping = params.new_cache_slots;
  attn_metadata.compute_dtype = compute_dtype;
  attn_metadata.attn_mask = attn_mask;

  bool is_start_loc_match = (params.q_seq_lens_vec == params.kv_seq_lens_vec);
  attn_metadata.is_chunked_prefill = is_prefill && !is_start_loc_match;
  attn_metadata.is_prefill = is_prefill && !attn_metadata.is_chunked_prefill;

  std::cerr << "params.kv_seq_lens: " << params.kv_seq_lens << std::endl;
  // attn_metadata.seq_lens = torch::diff(params.kv_seq_lens);
  attn_metadata.seq_lens = params.kv_seq_lens.to(torch::kCPU);

  if (!attn_metadata.is_prefill) {
    attn_metadata.block_table = params.block_tables;
  }

  return attn_metadata;
}

AttentionImpl::AttentionImpl(int num_heads,
                             int head_size,
                             float scale,
                             int num_kv_heads,
                             int sliding_window)
    : num_heads_(num_heads),
      head_size_(head_size),
      scale_(scale),
      num_kv_heads_(num_kv_heads),
      sliding_window_(sliding_window - 1) {}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> AttentionImpl::forward(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& query,
    torch::Tensor& key,
    torch::Tensor& value,
    KVCache& kv_cache) {
  std::cerr << "attn_metadata.seq_lens: " << attn_metadata.seq_lens
            << " attn_metadata.seq_lens.sizes():"
            << attn_metadata.seq_lens.sizes() << std::endl;
  auto output = torch::empty_like(query);
  auto output_lse = std::nullopt;

  query = query.view({-1, num_heads_, head_size_});
  key = key.view({-1, num_kv_heads_, head_size_});
  value = value.view({-1, num_kv_heads_, head_size_});
  output = output.view({-1, num_heads_, head_size_});

  torch::Tensor k_cache = kv_cache.get_k_cache();
  torch::Tensor v_cache = kv_cache.get_v_cache();

  atb::_npu_reshape_and_cache(
      key, value, k_cache, v_cache, attn_metadata.slot_mapping);

  std::cerr << "dyx-debug Before attention attn_metadata.is_prefill: "
            << attn_metadata.is_prefill << std::endl;
  if (attn_metadata.is_prefill) {
    std::cerr << "dyx-debug start _npu_flash_attention" << std::endl;
    auto fake_mask =
        torch::ones({attn_metadata.max_seq_len, attn_metadata.max_seq_len},
                    query.options().dtype(query.dtype()));

    std::cout << "dyx-debug  query shape: " << query.sizes() << std::endl;
    std::cout << "dyx-debug  key shape: " << key.sizes() << std::endl;
    std::cout << "dyx-debug  value shape: " << value.sizes() << std::endl;
    std::cout << "dyx-debug  attn_metadata.attn_mask shape: "
              << attn_metadata.attn_mask.sizes() << std::endl;
    std::cout << "dyx-debug  scale_: " << scale_ << std::endl;
    std::cout << "dyx-debug  num_heads_: " << num_heads_ << std::endl;
    std::cout << "dyx-debug  num_kv_heads_: " << num_kv_heads_ << std::endl;
    std::cout << "dyx-debug  output shape before attention: " << output.sizes()
              << std::endl;
    // mask = fake_mask.view({-1, 1, 1, attn_metadata.max_seq_len});

    // auto  seq_len_t = torch::tensor(4);
    // torch::Tensor seq_len_t = torch::tensor({4}, torch::kInt32);
    torch::Tensor seq_len_t =
        torch::tensor({4}, torch::dtype(torch::kInt32).device(torch::kCPU));
    std::cout << "dyx-debug  seq_len_t shape: " << seq_len_t.sizes()
              << std::endl;

    print_first_5_2(query, "_forward_prefill_only query");
    print_first_5_2(key, "_forward_prefill_only key");
    print_first_5_2(value, "_forward_prefill_only value");
    print_first_5_2(attn_metadata.seq_lens,
                    "_forward_prefill_only attn_metadata.seq_lens");
    std::cerr << "dyx-debug num_kv_heads: " << num_kv_heads_
              << ", num_heads: " << num_heads_ << ", scale: " << scale_
              << std::endl;

    atb::_npu_flash_attention(query,
                              key,
                              value,
                              attn_metadata.attn_mask,
                              attn_metadata.seq_lens,
                              // seq_len_t,
                              scale_,
                              num_heads_,
                              num_kv_heads_,
                              output);
    print_first_5_2(output, "_forward_prefill_only output");
  }
  // else if (attn_metadata.is_chunked_prefill) {
  //   tmo::torch_api::flash_attention(query,
  //                                   k_cache,
  //                                   v_cache,
  //                                   output,
  //                                   output_lse,
  //                                   attn_metadata.query_start_loc,
  //                                   attn_metadata.seq_start_loc,
  //                                   std::nullopt /* alibi_slope */,
  //                                   std::nullopt /* attn_bias */,
  //                                   std::nullopt /* q_quant_scale */,
  //                                   std::nullopt /* k_quant_scale */,
  //                                   std::nullopt /* v_quant_scale */,
  //                                   std::nullopt /* out_quant_scale */,
  //                                   attn_metadata.block_table,
  //                                   attn_metadata.max_query_len,
  //                                   attn_metadata.max_seq_len,
  //                                   scale_,
  //                                   true /* is_causal */,
  //                                   sliding_window_,
  //                                   -1,
  //                                   attn_metadata.compute_dtype,
  //                                   false /* return_lse */);
  // }
  else {
    std::cerr << "dyx-debug _npu_paged_attention will start " << std::endl;
    query = query.view({-1, num_heads_, head_size_});
    output = output.view({-1, num_heads_, head_size_});
    // tmo::torch_api::single_query_cached_kv_attn(
    //     query,
    //     k_cache,
    //     output,
    //     attn_metadata.block_table,
    //     attn_metadata.seq_lens,
    //     v_cache,
    //     output_lse,
    //     std::nullopt /* q_quant_scale */,
    //     std::nullopt /* k_cache_quant_scale */,
    //     std::nullopt /* v_cache_quant_scale */,
    //     std::nullopt /* out_quant_scale */,
    //     std::nullopt /* alibi_slope */,
    //     std::nullopt /* mask */,
    //     attn_metadata.compute_dtype,
    //     attn_metadata.max_seq_len,
    //     sliding_window_,
    //     -1 /* always -1 for window size right */,
    //     scale_,
    //     false /* return_lse */,
    //     -1 /* kv_cache_quant_bit_size */);

    print_first_5_2(query, "_forward_decode_only query");
    print_first_5_2(k_cache, "_forward_decode_only k_cache");
    print_first_5_2(v_cache, "_forward_decode_only v_cache");
    print_first_5_2(attn_metadata.block_table,
                    "_forward_decode_only attn_metadata.block_table");
    print_first_5_2(attn_metadata.seq_lens,
                    "_forward_decode_only attn_metadata.seq_lens");
    std::cerr << "dyx-debug num_kv_heads: " << num_kv_heads_
              << ", num_heads: " << num_heads_ << ", scale: " << scale_
              << std::endl;
    atb::_npu_paged_attention(query,
                              k_cache,
                              v_cache,
                              num_kv_heads_,
                              num_heads_,
                              scale_,
                              attn_metadata.block_table,
                              attn_metadata.seq_lens,
                              output);
    print_first_5_2(output, "_forward_decode_only output");
  }

  output = output.view({-1, num_heads_ * head_size_});
  return {output, output_lse};
}

}  // namespace layer
}  // namespace xllm
