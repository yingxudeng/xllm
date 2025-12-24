#pragma once

#include "rec.h"

#include <torch/torch.h>

namespace xllm::kernel::cuda::triton {

class RecTorchKernel : public RecKernel {
 public:
  RecTorchKernel();

  ~RecTorchKernel();

  torch::Tensor xattention(torch::Tensor q,                    // [batch_size, beam_size, num_heads, head_dim]
                           torch::Tensor shared_k_cache,       // [num_shared_kv_seq_len, kv_heads, head_dim]
                           torch::Tensor shared_v_cache,       // [num_shared_kv_seq_len, kv_heads, head_dim]
                           torch::Tensor unshared_k_cache,     // [max_num_request, beam_size, kv_heads, max_decode_step, head_dim]
                           torch::Tensor unshared_v_cache,     // [max_num_request, beam_size, kv_heads, max_decode_step, head_dim]
                           torch::Tensor kv_seq_len,     // [batch_size]
                           torch::Tensor block_table,       // [batch_size, 1]
                           float sm_scale, // 去掉
                           uint32_t step) override;

  void prefill_reshape_and_cache(torch::Tensor proj_k,          // [shared_len, kv_heads, head_dim]
                                 torch::Tensor proj_v,          // [shared_len, kv_heads, head_dim]
                                 torch::Tensor shared_k_cache,  // [num_shared_kv_seq_len, kv_heads, head_dim]
                                 torch::Tensor shared_v_cache   // [num_shared_kv_seq_len, kv_heads, head_dim]
                                 ) override;

  void decoder_reshape_and_cache(torch::Tensor proj_k,          // [batch_size, beam_size, kv_heads, head_dim]
                                 torch::Tensor proj_v,          // [batch_size, beam_size, kv_heads, head_dim]
                                 torch::Tensor unshared_k_cache,  // [max_num_request, beam_size, kv_heads, max_decode_step, head_dim]
                                 torch::Tensor unshared_v_cache,   // [max_num_request, beam_size, kv_heads, max_decode_step, head_dim]
                                 torch::Tensor block_table, 
                                 uint32_t step) override;
  
  void cache_select();

 private:
  // 全部为静态shape，能padding的都padding到最大长度
  void shared(torch::Tensor q,              // [batch_size, beam_size, num_heads, head_dim]
              torch::Tensor shared_k_cache, // [num_shared_kv_seq_len, kv_heads, head_dim].  num_shared_kv_seq_len为最大长度padding
              torch::Tensor shared_v_cache, // [num_shared_kv_seq_len, kv_heads, head_dim]
              torch::Tensor o,              // [batch_size, beam_size, num_heads, head_dim]
              torch::Tensor kv_seq_len,     // [batch_size]
              torch::Tensor shared_m,       // [batch_size, beam_size, num_heads]
              torch::Tensor shared_l,       // [batch_size, beam_size, num_heads]
              float sm_scale);

  void unshared(torch::Tensor q,                // [batch_size, beam_size, num_heads, head_dim]
                torch::Tensor unshared_k_cache, // [max_num_request, beam_size, kv_heads, max_decode_step, head_dim]
                torch::Tensor unshared_v_cache, // [max_num_request, beam_size, kv_heads, max_decode_step, head_dim]
                torch::Tensor o_unshared,       // [batch_size, beam_size, num_heads, head_dim]
                torch::Tensor block_table,       // [total_beams]
                torch::Tensor unshared_m,       // [batch_size, beam_size, num_heads]
                torch::Tensor unshared_l,       // [batch_size, beam_size, num_heads]
                float sm_scale,
                uint32_t step);
  
  void combine(torch::Tensor shared_o,        // [batch_size, beam_size, num_heads, head_dim]
               torch::Tensor shared_m,        // [batch_size, beam_size, num_heads]
               torch::Tensor shared_l,        // [batch_size, beam_size, num_heads]
               torch::Tensor unshared_o,      // [batch_size, beam_size, num_heads, head_dim]
               torch::Tensor unshared_m,      // [batch_size, beam_size, num_heads]
               torch::Tensor unshared_l,      // [batch_size, beam_size, num_heads]
               torch::Tensor final_o);        // [batch_size, beam_size, num_heads, head_dim]

  // torch::Tensor xattention(torch::Tensor q,                    // [batch_size * beam_size, num_heads, head_dim]
  //                          torch::Tensor shared_k_cache,       // [num_shared_kv_seq_len, kv_heads, head_dim]
  //                          torch::Tensor shared_v_cache,       // [num_shared_kv_seq_len, kv_heads, head_dim]
  //                          torch::Tensor unshared_k_cache,     // [max_num_request, beam_size, kv_heads, max_decode_step, head_dim]
  //                          torch::Tensor unshared_v_cache,     // [max_num_request, beam_size, kv_heads, max_decode_step, head_dim]
  //                          torch::Tensor O,
  //                          torch::Tensor kv_seq_len,     // [batch_size]
  //                          torch::Tensor block_table,       // [batch_size, 1]
  //                          torch::Tensor shared_m,
  //                          torch::Tensor shared_l,
  //                          torch::Tensor unshared_m,
  //                          torch::Tensor unshared_l,
  //                          uint32_t step) override;
  
  // void shared(torch::Tensor q,              // [batch_size * beam_size, num_heads, head_dim]
  //             torch::Tensor shared_k_cache, // [num_shared_kv_seq_len, kv_heads, head_dim].  num_shared_kv_seq_len为最大长度padding
  //             torch::Tensor shared_v_cache, // [num_shared_kv_seq_len, kv_heads, head_dim]
  //             torch::Tensor o,              // [batch_size * beam_size, num_heads, head_dim]
  //             torch::Tensor kv_seq_len,     // [batch_size]
  //             torch::Tensor shared_m,       // [batch_size * beam_size, num_heads]
  //             torch::Tensor shared_l,       // [batch_size * beam_size, num_heads]
  //             );

  // void unshared(torch::Tensor q,                // [batch_size * beam_size, num_heads, head_dim]
  //               torch::Tensor unshared_k_cache, // [max_num_request, beam_size, kv_heads, max_decode_step, head_dim]
  //               torch::Tensor unshared_v_cache, // [max_num_request, beam_size, kv_heads, max_decode_step, head_dim]
  //               torch::Tensor o_unshared,       // [batch_size * beam_size, num_heads, head_dim]
  //               torch::Tensor block_table,       // [total_beams]
  //               torch::Tensor unshared_m,       // [batch_size * beam_size, num_heads]
  //               torch::Tensor unshared_l,       // [batch_size * beam_size, num_heads]
  //               uint32_t step);
  
  // void combine(torch::Tensor shared_o,        // [batch_size * beam_size, num_heads, head_dim]
  //              torch::Tensor shared_m,        // [batch_size * beam_size, num_heads]
  //              torch::Tensor shared_l,        // [batch_size * beam_size, num_heads]
  //              torch::Tensor unshared_o,      // [batch_size * beam_size, num_heads, head_dim]
  //              torch::Tensor unshared_m,      // [batch_size * beam_size, num_heads]
  //              torch::Tensor unshared_l,      // [batch_size * beam_size, num_heads]
  //              torch::Tensor final_o);        // [batch_size * beam_size, num_heads, head_dim]
  };
} // namespace xllm::kernel::cuda::triton