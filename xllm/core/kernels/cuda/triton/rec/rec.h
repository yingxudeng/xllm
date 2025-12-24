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
                                         torch::Tensor shared_v_cache   // [num_shared_kv_seq_len, kv_heads, head_dim]
                                         ) = 0;

  virtual void decoder_reshape_and_cache(torch::Tensor proj_k,          // [batch_size, beam_size, kv_heads, head_dim]
                                         torch::Tensor proj_v,          // [batch_size, beam_size, kv_heads, head_dim]
                                         torch::Tensor unshared_k_cache,  // [max_num_request, beam_size, kv_heads, max_decode_step, head_dim]
                                         torch::Tensor unshared_v_cache,  // [max_num_request, beam_size, kv_heads, max_decode_step, head_dim]
                                         torch::Tensor block_table, 
                                         uint32_t step) = 0;
 private:

};

} // namespace xllm::kernel::cuda::triton