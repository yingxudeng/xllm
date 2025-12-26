// #pragma once

// #include "rec.h"

// #include <torch/torch.h>

// namespace xllm::kernel::cuda::triton {

// class RecTritonKernel : public RecKernel {
//  public:
//   RecTritonKernel();

//   ~RecTritonKernel();

//   torch::Tensor xattention(torch::Tensor q,                    // [batch_size, beam_size, num_heads, head_dim]
//                            torch::Tensor shared_k_cache,       // [num_shared_kv_seq_len, kv_heads, head_dim]
//                            torch::Tensor shared_v_cache,       // [num_shared_kv_seq_len, kv_heads, head_dim]
//                            torch::Tensor unshared_k_cache,     // [max_num_request, beam_size, kv_heads, max_decode_step, head_dim]
//                            torch::Tensor unshared_v_cache,     // [max_num_request, beam_size, kv_heads, max_decode_step, head_dim]
//                            torch::Tensor kv_seq_len,     // [batch_size]
//                            torch::Tensor block_table,       // [batch_size, 1]
//                            float sm_scale, // 去掉
//                            uint32_t step) override;

//   void prefill_reshape_and_cache(torch::Tensor proj_k,          // [shared_len, kv_heads, head_dim]
//                                  torch::Tensor proj_v,          // [shared_len, kv_heads, head_dim]
//                                  torch::Tensor shared_k_cache,  // [num_shared_kv_seq_len, kv_heads, head_dim]
//                                  torch::Tensor shared_v_cache   // [num_shared_kv_seq_len, kv_heads, head_dim]
//                                  ) override;

//   void decoder_reshape_and_cache(torch::Tensor proj_k,          // [batch_size, beam_size, kv_heads, head_dim]
//                                  torch::Tensor proj_v,          // [batch_size, beam_size, kv_heads, head_dim]
//                                  torch::Tensor unshared_k_cache,  // [max_num_request, beam_size, kv_heads, max_decode_step, head_dim]
//                                  torch::Tensor unshared_v_cache,   // [max_num_request, beam_size, kv_heads, max_decode_step, head_dim]
//                                  torch::Tensor block_table, 
//                                  uint32_t step) override;
  
//   void beam_search(torch::Tensor acc_logprob, 
//                    torch::Tensor in_sequence_group, 
//                    torch::Tensor top_tokens, 
//                    torch::Tensor top_logprobs, 
//                    torch::Tensor out_acc_logprob, 
//                    torch::Tensor out_token_ids, 
//                    torch::Tensor out_token_index, 
//                    torch::Tensor out_beam_count_prefix_sums, 
//                    torch::Tensor out_sequence_group, 
//                    uint32_t batch_size,
//                    uint32_t current_step) override;
  
//  private:
//   using BeamSearchKernelConfig = GenericKernelConfig<OneDimBlockSize>;

//   GenericKernelConfigs<BeamSearchKernelConfig> fp32_beam_search_configs_;
//   std::vector<std::vector<int>> fp32_beam_search_input_dim_array_;
// };
// } // namespace xllm::kernel::cuda::triton