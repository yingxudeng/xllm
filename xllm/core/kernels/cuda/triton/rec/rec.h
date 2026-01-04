#pragma once

#include <torch/torch.h>

namespace xllm::kernel::cuda::triton {

class RecKernel {
 public:
   virtual torch::Tensor xattention(torch::Tensor q,                    // [batch_size, beam_size, num_heads, head_dim]
                                    torch::Tensor shared_k_cache,       // [num_shared_kv_seq_len, kv_heads, head_dim]
                                    torch::Tensor shared_v_cache,       // [num_shared_kv_seq_len, kv_heads, head_dim]
                                    torch::Tensor unshared_k_cache,     // [max_num_request, beam_size, kv_heads, max_decode_step, head_dim]
                                    torch::Tensor unshared_v_cache,     // [max_num_request, beam_size, kv_heads, max_decode_step, head_dim]
                                    torch::Tensor kv_seq_len,     // [batch_size]
                                    torch::Tensor block_table,       // [total_beams]
                                    float sm_scale,
                                    uint32_t step) = 0;

  virtual void prefill_reshape_and_cache(torch::Tensor proj_k,          // [shared_len, kv_heads, head_dim]
                                         torch::Tensor proj_v,          // [shared_len, kv_heads, head_dim]
                                         torch::Tensor shared_k_cache,  // [num_shared_kv_seq_len, kv_heads, head_dim]
                                         torch::Tensor shared_v_cache) = 0; // [batch_size + 1], cumulative sequence lengths

  virtual void decoder_reshape_and_cache(torch::Tensor proj_k,          // [batch_size, beam_size, kv_heads, head_dim]
                                         torch::Tensor proj_v,          // [batch_size, beam_size, kv_heads, head_dim]
                                         torch::Tensor unshared_k_cache,  // [max_num_request, beam_size, kv_heads, max_decode_step, head_dim]
                                         torch::Tensor unshared_v_cache,  // [max_num_request, beam_size, kv_heads, max_decode_step, head_dim]
                                         torch::Tensor block_table, 
                                         uint32_t step) = 0;
  
  virtual void beam_search(torch::Tensor acc_logprob, 
                           torch::Tensor in_sequence_group, 
                           torch::Tensor top_tokens, 
                           torch::Tensor top_logprobs, 
                           torch::Tensor out_acc_logprob, 
                           torch::Tensor out_token_ids, 
                           torch::Tensor out_token_index, 
                           torch::Tensor out_beam_count_prefix_sums, 
                           torch::Tensor out_sequence_group, 
                           uint32_t batch_size,
                           uint32_t current_step) = 0;

  virtual void cache_select(const torch::Tensor& beam_index,        // [batch * beam, 1] - out_token_index
                            std::vector<torch::Tensor>& unshared_k_cache,  // per layer: [max_num_request, beam_size, max_decode_step, kv_heads, head_dim]
                            std::vector<torch::Tensor>& unshared_v_cache,  // per layer: [max_num_request, beam_size, max_decode_step, kv_heads, head_dim]
                            const torch::Tensor& block_table,        // [batch_size, 1]
                            const torch::Tensor& group_offset,       // [batch * beam, 1] - out_beam_count_prefix_sums
                            int64_t decode_step,                     // current round
                            int64_t beam_size,                       // beam width
                            int64_t layer_num) = 0;                 // number of layers
 private:

};

} // namespace xllm::kernel::cuda::triton