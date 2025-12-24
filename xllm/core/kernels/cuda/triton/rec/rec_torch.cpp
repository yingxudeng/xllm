#include "rec_torch.h"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include "cuda.h"

#include <torch/script.h>
#include <torch/torch.h>

namespace xllm::kernel::cuda::triton {
RecTorchKernel::RecTorchKernel() {

}

RecTorchKernel::~RecTorchKernel() {

}

torch::Tensor RecTorchKernel::xattention(torch::Tensor q,                    // [batch_size, beam_size, num_heads, head_dim]
                                         torch::Tensor shared_k_cache,       // [num_shared_kv_seq_len, kv_heads, head_dim]
                                         torch::Tensor shared_v_cache,       // [num_shared_kv_seq_len, kv_heads, head_dim]
                                         torch::Tensor unshared_k_cache,     // [max_num_request, beam_size, kv_heads, max_decode_step, head_dim]
                                         torch::Tensor unshared_v_cache,     // [max_num_request, beam_size, kv_heads, max_decode_step, head_dim]
                                         torch::Tensor kv_seq_len,           // [batch_size]
                                         torch::Tensor block_table,          // [total_beams]
                                         float sm_scale,
                                         uint32_t step) {
  LOG(INFO) << "inner RecTorchKernel::xattention.";
  int64_t batch_size = q.size(0);
  int64_t beam_size = q.size(1);
  int64_t num_heads = q.size(2);
  int64_t head_dim = q.size(3);
  int64_t num_shared_kv_seq_len = shared_k_cache.size(0);
  int64_t kv_heads = shared_k_cache.size(1);
  
  torch::Tensor shared_o = torch::zeros_like(q); // [batch_size, beam_size, num_heads, head_dim]
  torch::Tensor shared_m = torch::zeros({batch_size, beam_size, num_heads}, q.options()); // [batch_size, beam_size, num_heads]
  torch::Tensor shared_l = torch::zeros({batch_size, beam_size, num_heads}, q.options()); // [batch_size, beam_size, num_heads]
  this->shared(q, shared_k_cache, shared_v_cache, shared_o, 
               kv_seq_len, shared_m, shared_l, sm_scale);
  LOG(INFO) << "shared_m.dtype: " << shared_m.dtype();
  LOG(INFO) << "after shared.";
  torch::Tensor unshared_o = torch::zeros_like(q); // [batch_size, beam_size, num_heads, head_dim]
  torch::Tensor unshared_m = torch::zeros({batch_size, beam_size, num_heads}, q.options()); // [batch_size, beam_size, num_heads]
  torch::Tensor unshared_l = torch::zeros({batch_size, beam_size, num_heads}, q.options()); // [batch_size, beam_size, num_heads]
  this->unshared(q, unshared_k_cache, unshared_v_cache, unshared_o, 
                 block_table, unshared_m, unshared_l, sm_scale, step);
  LOG(INFO) << "after unshared.";
  torch::Tensor final_o = torch::zeros_like(q); // [batch_size, beam_size, num_heads, head_dim]  
  this->combine(shared_o, shared_m, shared_l, 
                unshared_o, unshared_m, unshared_l, final_o);
  LOG(INFO) << "after combine.";
  return final_o;
}


void RecTorchKernel::prefill_reshape_and_cache(torch::Tensor proj_k,          // [shared_len, kv_heads, head_dim]
                                               torch::Tensor proj_v,          // [shared_len, kv_heads, head_dim]
                                               torch::Tensor shared_k_cache,  // [num_shared_kv_seq_len, kv_heads, head_dim]
                                               torch::Tensor shared_v_cache   // [num_shared_kv_seq_len, kv_heads, head_dim]
                                               ) {
  LOG(INFO) << "inner RecTorchKernel::prefill_reshape_and_cache";
  // 获取维度信息
  int64_t shared_len = proj_k.size(0);
  int64_t kv_heads = proj_k.size(1);
  int64_t head_dim = proj_k.size(2);
  int64_t num_shared_kv_seq_len = shared_k_cache.size(0);
  
  // 检查维度兼容性
  CHECK_LE(shared_len, num_shared_kv_seq_len) << 
              "shared_len () must be <= num_shared_kv_seq_len ";
  CHECK_EQ(proj_k.size(1), shared_k_cache.size(1)) <<
              "kv_heads dimension mismatch";
  CHECK_EQ(proj_k.size(2), shared_k_cache.size(2)) << 
              "head_dim dimension mismatch";
  CHECK_EQ(proj_v.sizes(), proj_k.sizes()) << 
              "proj_v and proj_k must have same shape";
  CHECK_EQ(shared_v_cache.sizes(), shared_k_cache.sizes()) << 
              "shared_v_cache and shared_k_cache must have same shape";
  LOG(INFO) << "before copy_.";
  // 将 proj_k 和 proj_v 复制到 cache 的前 shared_len 位置
  // 方法1: 使用 slice 和 copy_
  shared_k_cache.slice(0, 0, shared_len).copy_(proj_k);
  shared_v_cache.slice(0, 0, shared_len).copy_(proj_v);
  LOG(INFO) << "after copy_.";
}

void RecTorchKernel::decoder_reshape_and_cache(torch::Tensor proj_k,          // [batch_size, beam_size, kv_heads, head_dim]
                                               torch::Tensor proj_v,          // [batch_size, beam_size, kv_heads, head_dim]
                                               torch::Tensor unshared_k_cache,  // [max_num_request, beam_size, kv_heads, max_decode_step, head_dim]
                                               torch::Tensor unshared_v_cache,   // [max_num_request, beam_size, kv_heads, max_decode_step, head_dim]
                                               torch::Tensor block_table,     // [batch_size]
                                               uint32_t step) {
  // 维度检查
  TORCH_CHECK(proj_k.dim() == 4, "proj_k must be 4-dimensional");
  TORCH_CHECK(proj_v.dim() == 4, "proj_v must be 4-dimensional");
  TORCH_CHECK(unshared_k_cache.dim() == 5, "unshared_k_cache must be 5-dimensional");
  TORCH_CHECK(unshared_v_cache.dim() == 5, "unshared_v_cache must be 5-dimensional");
  TORCH_CHECK(block_table.dim() == 2, "block_table must be 2-dimensional");
  TORCH_CHECK(block_table.size(1) == 1, "second dim must be 1");
  
  int64_t batch_size = proj_k.size(0);
  int64_t beam_size = proj_k.size(1);
  int64_t kv_heads = proj_k.size(2);
  int64_t head_dim = proj_k.size(3);
  int64_t max_decode_step = unshared_k_cache.size(3);
  
  // 形状兼容性检查
  TORCH_CHECK(proj_v.sizes() == proj_k.sizes(), "proj_v and proj_k must have same shape");
  TORCH_CHECK(block_table.size(0) == batch_size, "block_table size must match batch_size");
  TORCH_CHECK(step >= 0 && step < max_decode_step, "step must be in valid range");
  
  for (int64_t i = 0; i < batch_size; ++i) {
      int64_t batch_id = i;
      
      // 选择当前batch的数据
      torch::Tensor target_batch_proj_k = proj_k.select(0, batch_id);  // [beam_size, kv_heads, head_dim]
      torch::Tensor target_batch_proj_v = proj_v.select(0, batch_id);  // [beam_size, kv_heads, head_dim]
      
      // 获取对应的block索引
      int64_t batch_index = block_table[batch_id][0].item<int64_t>();
      
      // 边界检查
      TORCH_CHECK(batch_index >= 0 && batch_index < unshared_k_cache.size(0), 
                  "batch_index out of range");
      
      // 将数据复制到缓存中
      unshared_k_cache.select(0, batch_index).select(2, step).copy_(target_batch_proj_k);
      unshared_v_cache.select(0, batch_index).select(2, step).copy_(target_batch_proj_v);
  }
}

void RecTorchKernel::shared(torch::Tensor q,              // [batch_size, beam_size, num_heads, head_dim]
                            torch::Tensor shared_k_cache, // [num_shared_kv_seq_len, kv_heads, head_dim].  num_shared_kv_seq_len为最大长度padding
                            torch::Tensor shared_v_cache, // [num_shared_kv_seq_len, kv_heads, head_dim]
                            torch::Tensor o,              // [batch_size, beam_size, num_heads, head_dim]
                            torch::Tensor kv_seq_len,     // [batch_size]
                            torch::Tensor shared_m,       // [batch_size, beam_size, num_heads]
                            torch::Tensor shared_l,       // [batch_size, beam_size, num_heads]
                            float sm_scale) {
  int64_t BLOCK_SIZE_N = 64;

  int64_t batch_size = q.size(0);
  int64_t beam_size = q.size(1);
  int64_t num_heads = q.size(2);
  int64_t head_dim = q.size(3);
  int64_t num_shared_kv_seq_len = shared_k_cache.size(0);
  int64_t kv_heads = shared_k_cache.size(1);

  float scale = sm_scale; // 使用传入的 sm_scale 参数
  
  int64_t kv_seq_len_prefix_sum = 0;

  for (int64_t i = 0; i < batch_size; ++i) {
    torch::Tensor target_Q = q.select(0, i); // [beam_size, num_heads, head_dim]
    LOG(INFO) << "target_Q.shape: " << target_Q.sizes();
    int64_t group_size = num_heads / kv_heads;
    target_Q = target_Q.view({beam_size, kv_heads, group_size, head_dim}); // [beam_size, kv_heads, group_size, head_dim]
    target_Q = target_Q.unsqueeze(-2); // [beam_size, kv_heads, group_size, 1, head_dim]
    
    int64_t batch_id = i;
    int64_t begin = kv_seq_len_prefix_sum;
    int64_t cur_kv_seq_len = kv_seq_len[batch_id].item().to<int64_t>();
    int64_t end = cur_kv_seq_len + begin;
    kv_seq_len_prefix_sum += cur_kv_seq_len;
    
    torch::Tensor target_batch_K = shared_k_cache.slice(0, begin, end); // [cur_kv_seq_len, kv_heads, head_dim]
    torch::Tensor target_batch_V = shared_v_cache.slice(0, begin, end); // [cur_kv_seq_len, kv_heads, head_dim]
    
    torch::Tensor acc = torch::zeros_like(target_Q); // [beam_size, kv_heads, group_size, 1, head_dim]

    torch::Tensor m_i = torch::full({beam_size, kv_heads, group_size, 1}, -std::numeric_limits<float>::infinity(), target_Q.options()); // [beam_size, kv_heads, group_size, 1]
    torch::Tensor l_i = torch::zeros({beam_size, kv_heads, group_size, 1}, target_Q.options()); // [beam_size, kv_heads, group_size, 1]
    
    for (int64_t j = 0; j < cur_kv_seq_len; j += BLOCK_SIZE_N) {
      int64_t seq_begin = j;
      int64_t seq_end = std::min(j + BLOCK_SIZE_N, cur_kv_seq_len);
      
      torch::Tensor target_batch_seq_K = target_batch_K.slice(0, seq_begin, seq_end); // [seq_end - seq_begin, kv_heads, head_dim]
      torch::Tensor target_batch_seq_V = target_batch_V.slice(0, seq_begin, seq_end); // [seq_end - seq_begin, kv_heads, head_dim]
      
      // 调整维度顺序以匹配矩阵乘法要求
      target_batch_seq_K = target_batch_seq_K.transpose(0, 1).unsqueeze(1); // [kv_heads, 1, seq_end - seq_begin, head_dim]
      target_batch_seq_V = target_batch_seq_V.transpose(0, 1).unsqueeze(1); // [kv_heads, 1, seq_end - seq_begin, head_dim]
      
      // Q: [beam_size, kv_heads, group_size, 1, head_dim]
      // K: [kv_heads, 1, seq_end - seq_begin, head_dim]
      // qk: [beam_size, kv_heads, group_size, 1, seq_end - seq_begin]
      torch::Tensor qk = torch::matmul(target_Q, target_batch_seq_K.transpose(-2, -1)) * scale;
      
      torch::Tensor max_qk = std::get<0>(torch::max(qk, -1, true)); // [beam_size, kv_heads, group_size, 1, 1]
      max_qk = max_qk.squeeze(-1); // [beam_size, kv_heads, group_size, 1]
      
      torch::Tensor m_j = max_qk; // [beam_size, kv_heads, group_size, 1]
      
      torch::Tensor m_new = torch::maximum(m_i, m_j); // [beam_size, kv_heads, group_size, 1]
      
      qk = qk - m_new.unsqueeze(-1); // [beam_size, kv_heads, group_size, 1, seq_end - seq_begin]
      
      torch::Tensor p = torch::exp(qk); // [beam_size, kv_heads, group_size, 1, seq_end - seq_begin]
      
      torch::Tensor l_j = torch::sum(p, -1); // [beam_size, kv_heads, group_size, 1]
      
      // p: [beam_size, kv_heads, group_size, 1, seq_end - seq_begin]
      // V: [1, kv_heads, 1, seq_end - seq_begin, head_dim]
      // cur_o: [beam_size, kv_heads, group_size, 1, head_dim]
      torch::Tensor cur_o = torch::matmul(p, target_batch_seq_V);
      
      torch::Tensor alpha = torch::exp(m_i - m_new); // [beam_size, kv_heads, group_size, 1]
      
      acc = acc * alpha.unsqueeze(-1) + cur_o; // [beam_size, kv_heads, group_size, 1, head_dim]
      l_i = l_i * alpha + l_j; // [beam_size, kv_heads, group_size, 1]
      m_i = m_new; // [beam_size, kv_heads, group_size, 1]
    }
    
    acc = acc.view({beam_size, group_size * kv_heads, 1, head_dim}); // [beam_size, num_heads, 1, head_dim]
    l_i = l_i.view({beam_size, group_size * kv_heads, 1}); // [beam_size, num_heads, 1]
    
    // o [batch_size, beam_size, num_heads, head_dim]
    o.select(0, batch_id).copy_(acc.squeeze(-2)); // 去掉seq_len=1的维度
    
    // shared_l [batch_size, beam_size, num_heads]
    shared_l.select(0, batch_id).copy_(l_i.squeeze(-1)); // [beam_size, num_heads]
    
    // shared_m [batch_size, beam_size, num_heads]
    m_i = m_i.view({beam_size, group_size * kv_heads, 1}); // [beam_size, num_heads, 1]
    shared_m.select(0, batch_id).copy_(m_i.squeeze(-1)); // [beam_size, num_heads]
  }
}
void RecTorchKernel::unshared(torch::Tensor q,                 // [batch_size, beam_size, num_heads, head_dim]
                              torch::Tensor unshared_k_cache,  // [max_num_request, beam_size, kv_heads, max_decode_step, head_dim]
                              torch::Tensor unshared_v_cache,  // [max_num_request, beam_size, kv_heads, max_decode_step, head_dim]
                              torch::Tensor o_unshared,        // [batch_size, beam_size, num_heads, head_dim]
                              torch::Tensor block_table,       // [total_beams]
                              torch::Tensor unshared_m,        // [batch_size, beam_size, num_heads]
                              torch::Tensor unshared_l,        // [batch_size, beam_size, num_heads]
                              float sm_scale,
                              uint32_t step) {
  int64_t batch_size = q.size(0);
  int64_t beam_size = q.size(1);
  int64_t num_heads = q.size(2);
  int64_t head_dim = q.size(3);
  int64_t max_num_request = unshared_k_cache.size(0);
  int64_t kv_heads = unshared_k_cache.size(2);
  int64_t max_decode_step = unshared_k_cache.size(3);
  
  float scale = sm_scale; // 使用传入的 sm_scale 参数
  
  for (int64_t i = 0; i < batch_size; ++i) {
    torch::Tensor target_Q = q.select(0, i); // [beam_size, num_heads, head_dim]
    int64_t group_size = num_heads / kv_heads;
    
    target_Q = target_Q.view({beam_size, kv_heads, group_size, head_dim}); // [beam_size, kv_heads, group_size, head_dim]
    target_Q = target_Q.unsqueeze(-2); // [beam_size, kv_heads, group_size, 1, head_dim] (添加seq维度)
    
    int64_t batch_id = i;
    int64_t block_id = block_table[batch_id][0].item().to<int64_t>();
    
    torch::Tensor target_request_K = unshared_k_cache.select(0, block_id); // [beam_size, kv_heads, max_decode_step, head_dim]
    torch::Tensor target_request_V = unshared_v_cache.select(0, block_id); // [beam_size, kv_heads, max_decode_step, head_dim]
    
    // 现在K和V已经是正确的维度顺序，只需要添加group维度
    target_request_K = target_request_K.unsqueeze(2); // [beam_size, kv_heads, 1, max_decode_step, head_dim]
    target_request_V = target_request_V.unsqueeze(2); // [beam_size, kv_heads, 1, max_decode_step, head_dim]
    
    // step可能是0和1，第0步是，需要读第1个kv，第1步时需要读2个kv
    torch::Tensor target_request_seq_K = target_request_K.slice(3, 0, step + 1); // [beam_size, kv_heads, 1, step + 1, head_dim]
    torch::Tensor target_request_seq_V = target_request_V.slice(3, 0, step + 1); // [beam_size, kv_heads, 1, step + 1, head_dim]
    
    torch::Tensor qk = torch::matmul(target_Q, target_request_seq_K.transpose(-2, -1)) * scale; // [beam_size, kv_heads, group_size, 1, step + 1]
    
    torch::Tensor tmp_m = std::get<0>(torch::max(qk, -1)); // [beam_size, kv_heads, group_size, 1]
    
    torch::Tensor m = tmp_m.view({beam_size, num_heads, 1}).squeeze(-1); // [beam_size, num_heads]
    unshared_m.select(0, batch_id).copy_(m);
    
    torch::Tensor qk_shifted = qk - tmp_m.unsqueeze(-1); // [beam_size, kv_heads, group_size, 1, step + 1]
    torch::Tensor qk_exp = torch::exp(qk_shifted); // [beam_size, kv_heads, group_size, 1, step + 1]
    
    torch::Tensor tmp_l = torch::sum(qk_exp, -1); // [beam_size, kv_heads, group_size, 1]
    torch::Tensor l = tmp_l.view({beam_size, num_heads, 1}).squeeze(-1); // [beam_size, num_heads]
    unshared_l.select(0, batch_id).copy_(l);
    
    torch::Tensor p = qk_exp / tmp_l.unsqueeze(-1); // [beam_size, kv_heads, group_size, 1, step + 1]
    
    torch::Tensor output = torch::matmul(p, target_request_seq_V); // [beam_size, kv_heads, group_size, 1, head_dim]
    output = output.view({beam_size, num_heads, head_dim}); // [beam_size, num_heads, head_dim]
    
    o_unshared.select(0, batch_id).copy_(output);
  }
}

void RecTorchKernel::combine(torch::Tensor shared_o,   
                              torch::Tensor shared_m, 
                              torch::Tensor shared_l, 
                              torch::Tensor unshared_o, 
                              torch::Tensor unshared_m, 
                              torch::Tensor unshared_l, 
                              torch::Tensor final_o) {
  // 步骤1: 反归一化，恢复累积状态
  torch::Tensor acc_shared = shared_o * shared_l.unsqueeze(-1);   // [batch_size, beam_size, num_heads, head_dim]
  torch::Tensor acc_unshared = unshared_o * unshared_l.unsqueeze(-1); // [batch_size, beam_size, num_heads, head_dim]
  
  // 步骤2: 计算全局max（每个query位置独立）
  torch::Tensor m_global = torch::maximum(shared_m, unshared_m); // [batch_size, beam_size, num_heads]
  
  // 步骤3: 重新缩放到全局max
  torch::Tensor alpha_shared = torch::exp(shared_m - m_global);   // [batch_size, beam_size, num_heads]
  torch::Tensor alpha_unshared = torch::exp(unshared_m - m_global); // [batch_size, beam_size, num_heads]
  
  // 步骤4: 重新缩放累积值
  torch::Tensor acc_shared_rescaled = acc_shared * alpha_shared.unsqueeze(-1);     // [batch_size, beam_size, num_heads, head_dim]
  torch::Tensor acc_unshared_rescaled = acc_unshared * alpha_unshared.unsqueeze(-1); // [batch_size, beam_size, num_heads, head_dim]
  torch::Tensor l_shared_rescaled = shared_l * alpha_shared;       // [batch_size, beam_size, num_heads]
  torch::Tensor l_unshared_rescaled = unshared_l * alpha_unshared; // [batch_size, beam_size, num_heads]
  
  // 步骤5: 合并
  torch::Tensor acc_merged = acc_shared_rescaled + acc_unshared_rescaled; // [batch_size, beam_size, num_heads, head_dim]
  torch::Tensor l_merged = l_shared_rescaled + l_unshared_rescaled;       // [batch_size, beam_size, num_heads]
  
  // 步骤6: 重新归一化并写入final_o
  final_o.copy_(acc_merged / l_merged.unsqueeze(-1));
}

} // namespace xllm::kernel::cuda::triton