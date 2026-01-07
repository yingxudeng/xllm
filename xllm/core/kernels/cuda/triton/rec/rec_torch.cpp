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
                                               torch::Tensor shared_v_cache) {
  // LOG(INFO) << "proj_k.shape: " << proj_k.sizes();
  // LOG(INFO) << "proj_v.shape: " << proj_v.sizes();
  // LOG(INFO) << "shared_k_cache.shape: " << shared_k_cache.sizes();
  // LOG(INFO) << "shared_v_cache.shape: " << shared_v_cache.sizes();
  int64_t shared_len = proj_k.size(0);
  shared_k_cache = shared_k_cache.slice(0, 0, shared_len);
  shared_v_cache = shared_v_cache.slice(0, 0, shared_len);
  shared_k_cache.copy_(proj_k);
  shared_v_cache.copy_(proj_v);
}

void RecTorchKernel::decoder_reshape_and_cache(torch::Tensor proj_k,          // [batch_size, beam_size, kv_heads, head_dim]
                                               torch::Tensor proj_v,          // [batch_size, beam_size, kv_heads, head_dim]
                                               torch::Tensor unshared_k_cache,  // [max_num_request, beam_size, max_decode_step, kv_heads, head_dim]
                                               torch::Tensor unshared_v_cache,   // [max_num_request, beam_size, max_decode_step, kv_heads, head_dim]
                                               torch::Tensor block_table,     // [batch_size, 1]
                                               uint32_t step) {
  // 获取维度信息
  int64_t batch_size = proj_k.size(0);
  int64_t beam_size = proj_k.size(1);
  int64_t kv_heads = proj_k.size(2);
  int64_t head_dim = proj_k.size(3);
  int64_t max_num_request = unshared_k_cache.size(0);
  int64_t max_decode_step = unshared_k_cache.size(2);
  // LOG(INFO) << "proj_k.shape: " << proj_k.sizes();
  // LOG(INFO) << "proj_v.shape: " << proj_v.sizes();
  // LOG(INFO) << "unshared_k_cache.shape: " << unshared_k_cache.sizes();
  // LOG(INFO) << "unshared_v_cache.shape: " << unshared_v_cache.sizes();
  // LOG(INFO) << "block_table.shape: " << block_table.sizes();
  // LOG(INFO) << "step: " << step;
  // 检查维度兼容性
  CHECK_EQ(proj_v.sizes(), proj_k.sizes()) << 
              "proj_v and proj_k must have same shape";
  CHECK_EQ(block_table.size(0), batch_size) <<
              "block_table size must match batch_size";
  CHECK_EQ(block_table.size(1), 1) <<
              "block_table second dim must be 1";
  CHECK_LT(step, max_decode_step) <<
              "step must be less than max_decode_step";
  CHECK_EQ(unshared_k_cache.size(1), beam_size) <<
              "unshared_k_cache beam_size mismatch";
  CHECK_EQ(unshared_k_cache.size(3), kv_heads) <<
              "unshared_k_cache kv_heads mismatch";
  CHECK_EQ(unshared_k_cache.size(4), head_dim) <<
              "unshared_k_cache head_dim mismatch";
  CHECK_EQ(unshared_v_cache.sizes(), unshared_k_cache.sizes()) <<
              "unshared_v_cache and unshared_k_cache must have same shape";
  
  // 将 block_table 移到 CPU 以便访问
  auto block_table_cpu = block_table.select(1, 0).to(torch::kCPU);  // [batch_size]
  
  // 遍历每个 batch，将 proj_k 和 proj_v 写入到对应的位置
  for (int64_t b = 0; b < batch_size; ++b) {
    int64_t request_id = block_table_cpu[b].item<int64_t>();
    
    // 检查 request_id 是否有效
    CHECK_GE(request_id, 0) << "Invalid request_id: " << request_id;
    CHECK_LT(request_id, max_num_request) << 
                "request_id (" << request_id << ") >= max_num_request (" 
                << max_num_request << ")";
    
    // 提取当前 batch 的 proj_k 和 proj_v: [beam_size, kv_heads, head_dim]
    auto proj_k_batch = proj_k[b];  // [beam_size, kv_heads, head_dim]
    auto proj_v_batch = proj_v[b];  // [beam_size, kv_heads, head_dim]
    
    // 写入到 unshared_k_cache 和 unshared_v_cache 的对应位置
    // unshared_k_cache[request_id, :, step, :, :] = proj_k_batch
    unshared_k_cache[request_id].select(1, step).copy_(proj_k_batch);
    unshared_v_cache[request_id].select(1, step).copy_(proj_v_batch);
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


// log_probs_ptr,       # [B*BEAM_SIZE, 1] - 当前beam的对数概率
      // in_sequence_ptr,     # [B, BEAM_SIZE, MAX_DECODE_STEP] - 输入序列（只读）
      // top_tokens_ptr,      # [B*BEAM_SIZE, TOP_K] - 每个beam的top K个token
      // top_probs_ptr,       # [B*BEAM_SIZE, TOP_K] - 每个beam的top K个概率

      // out_log_probs_ptr,   # [B*BEAM_SIZE, 1] - 输出对数概率
      // out_token_ids_ptr,   # [B*BEAM_SIZE, 1] - 输出token ID
      // out_token_index_ptr, # [B*BEAM_SIZE, 1] - 输出token在top K中的索引
      // out_beam_count_prefix_sums_ptr,  # [B*BEAM_SIZE, 1] - beam计数前缀和（未使用）
      // out_sequence_ptr,    # [B, BEAM_SIZE, MAX_DECODE_STEP] - 输出序列（写入）
void RecTorchKernel::beam_search(torch::Tensor acc_logprob, 
                                 torch::Tensor in_sequence_group, 
                                 torch::Tensor top_tokens, 
                                 torch::Tensor top_logprobs, 
                                 torch::Tensor out_acc_logprob, 
                                 torch::Tensor out_token_ids, 
                                 torch::Tensor out_token_index, 
                                 torch::Tensor out_beam_count_prefix_sums, 
                                 torch::Tensor out_sequence_group, 
                                 uint32_t batch_size,
                                 uint32_t current_step) {

  torch::Device device = acc_logprob.device();
  
  uint32_t beam_size = in_sequence_group.size(1);

  uint32_t top_k = top_tokens.size(1);
  uint32_t total_rounds = in_sequence_group.size(2);
  
  CHECK_EQ(beam_size, top_k) << "beam_size must be equal with top_k.";
  
  if (current_step == 0) {

    // [batch_size, beam_size]
    auto tokens = top_tokens.view({batch_size * 1, top_k}).slice(1, 0, beam_size);
    // [batch_size * beam_size]

    tokens = tokens.reshape({batch_size * beam_size, 1});
    // [batch_size, beam_size]

    auto init_probs = top_logprobs.view({batch_size * 1, top_k}).slice(1, 0, beam_size);
    // [batch_size * beam_size]

    init_probs = init_probs.reshape({batch_size * beam_size, 1});

    out_acc_logprob.copy_(init_probs);
    out_token_ids.copy_(tokens);

    auto indices = torch::arange(beam_size, torch::kInt32).to(device); // [beam_size]
    indices = indices.unsqueeze(0).expand({batch_size, -1}).reshape({-1, 1}); // [batch * beam, 1]
    out_token_index.copy_(indices);

    auto sequence_view = out_sequence_group.view({batch_size * beam_size, total_rounds});

    sequence_view.select(1, 0).copy_(tokens.squeeze(1));

  } else {

    auto combined_probs = acc_logprob + top_logprobs;

    combined_probs = combined_probs.view({batch_size, beam_size * top_k});

    auto topk_result = torch::topk(combined_probs, beam_size, -1);
    auto new_probs = std::get<0>(topk_result);    // [batch_size, beam_size]
    auto new_indices = std::get<1>(topk_result);  // [batch_size, beam_size]

    auto parent_beam = (new_indices / top_k).to(torch::kLong);      // [batch_size, beam_size]
    auto token_in_beam = (new_indices % top_k).to(torch::kLong);    // [batch_size, beam_size]

    // [batch_size, 1]
    auto batch_idx = torch::arange(batch_size, torch::kLong).to(device).unsqueeze(1);
    // [batch_size, beam_size, top_k]
    auto top_tokens_reshaped = top_tokens.view({batch_size, beam_size, top_k});

    auto new_tokens = top_tokens_reshaped.index({batch_idx, parent_beam, token_in_beam});

    out_acc_logprob.copy_(new_probs.reshape({-1, 1}));
    out_token_index.copy_(new_indices.reshape({-1, 1}));
    out_token_ids.copy_(new_tokens.reshape({-1, 1}));

    auto batch_range = torch::arange(batch_size, torch::kInt32).to(device).unsqueeze(1).expand({-1, beam_size});
    auto beam_range = torch::arange(beam_size, torch::kInt32).to(device).unsqueeze(0).expand({batch_size, -1});

    out_sequence_group.slice(2, 0, current_step) = 
        in_sequence_group.index({batch_range, parent_beam, torch::indexing::Slice(0, current_step)});

    out_sequence_group.slice(2, current_step, current_step + 1) = new_tokens.unsqueeze(2);
  }
}

void RecTorchKernel::cache_select(const torch::Tensor& beam_index,        // [batch * beam, 1] - out_token_index
                                  std::vector<torch::Tensor>& unshared_k_cache,  // per layer: [max_num_request, beam_size, max_decode_step, kv_heads, head_dim]
                                  std::vector<torch::Tensor>& unshared_v_cache,  // per layer: [max_num_request, beam_size, max_decode_step, kv_heads, head_dim]
                                  const torch::Tensor& block_table,        // [batch_size, 1]
                                  const torch::Tensor& group_offset,       // [batch * beam, 1] - out_beam_count_prefix_sums
                                  int64_t decode_step,                     // current round (step 0, 1, ...)
                                  int64_t beam_size,                       // beam width
                                  int64_t layer_num) {                     // number of layers
  // 获取维度信息
  int64_t batch_size = block_table.size(0);
  int64_t total_beams = beam_index.size(0);
  int64_t kv_heads = unshared_k_cache[0].size(3);
  int64_t head_dim = unshared_k_cache[0].size(4);
  // LOG(INFO) << "block_table: " << block_table;
  // LOG(INFO) << "beam_index: " << beam_index;
  // LOG(INFO) << "unshared_k_cache: " << unshared_k_cache[0].sizes();
  // LOG(INFO) << "unshared_v_cache: " << unshared_v_cache[0].sizes();
  // LOG(INFO) << "layer_num: " << layer_num;
  // LOG(INFO) << "decode_step: " << decode_step;
  // LOG(INFO) << "beam_size: " << beam_size;
  // LOG(INFO) << "total_beams: " << total_beams;
  CHECK_EQ(total_beams, batch_size * beam_size) << "beam_index size mismatch";
  // 检查 unshared cache 的维度
  if (layer_num > 0) {
    int64_t max_num_request = unshared_k_cache[0].size(0);
    int64_t max_decode_step = unshared_k_cache[0].size(2);
    // 将 beam_index reshape 成 [batch_size, beam_size] 并计算 parent_beam
    // beam_index 是 new_indices，parent_beam = new_indices / top_k (top_k == beam_size)
    auto beam_index_reshaped = beam_index.reshape({batch_size, beam_size}).to(torch::kLong);  // [batch_size, beam_size]
    auto parent_beam = (beam_index_reshaped / beam_size).to(torch::kLong);  // [batch_size, beam_size]
    
    // 将 block_table 移到 CPU 以便访问 request_id
    auto _block_table = block_table.select(1, 0);
    
    auto ori_k_cache = torch::zeros({layer_num, batch_size, beam_size, max_decode_step, kv_heads, head_dim});
    auto ori_v_cache = torch::zeros({layer_num, batch_size, beam_size, max_decode_step, kv_heads, head_dim});

    for (int64_t layer = 0; layer < layer_num; ++layer) {
      auto k_cache = unshared_k_cache[layer];
      auto v_cache = unshared_v_cache[layer];
      ori_k_cache[layer].copy_(k_cache.index_select(0, _block_table));
      ori_v_cache[layer].copy_(v_cache.index_select(0, _block_table));
    }

    // 对于每一层，更新 unshared cache
    for (int64_t layer = 0; layer < layer_num; ++layer) {
      auto k_cache = unshared_k_cache[layer];
      auto v_cache = unshared_v_cache[layer];
      
      // 对于每个 batch
      for (int64_t b = 0; b < batch_size; ++b) {
        int64_t request_id = block_table[b].item<int64_t>();
        
        // 检查 request_id 是否有效
        CHECK_GE(request_id, 0) << "Invalid request_id: " << request_id;
        CHECK_LT(request_id, max_num_request) << 
                    "request_id (" << request_id << ") >= max_num_request (" 
                    << max_num_request << ")";
        
        // 获取当前 batch 的 parent_beam: [beam_size]
        auto parent_beam_batch = parent_beam[b];  // [beam_size]
        
        // 对于每个新的 beam，从对应的旧 beam 复制所有历史 step 的 KV cache
        // 需要复制从 0 到 decode_step 的所有 step，因为新的 beam 可能来自不同的旧 beam
        // 例如：在 step 1 时，如果新的 beam 1 来自旧的 beam 0（而不是旧的 beam 1），
        // 那么新的 beam 1 的所有历史（step 0 和 step 1）都应该来自旧的 beam 0
        for (int64_t new_beam = 0; new_beam < beam_size; ++new_beam) {
          int64_t old_beam = parent_beam_batch[new_beam].item<int64_t>();
          
          // 检查 old_beam 是否有效
          CHECK_GE(old_beam, 0) << "Invalid old_beam: " << old_beam;
          CHECK_LT(old_beam, beam_size) << 
                      "old_beam (" << old_beam << ") >= beam_size (" 
                      << beam_size << ")";
          
          // 如果新 beam 和旧 beam 相同，则不需要复制（保持不变）
          if (new_beam == old_beam) {
            continue;
          }
          
          // 复制从 0 到 decode_step 的所有 step 的 KV cache 值
          // 从 unshared_k_cache[request_id, old_beam, 0:decode_step+1, :, :] 
          // 复制到 unshared_k_cache[request_id, new_beam, 0:decode_step+1, :, :]
          for (int64_t step = 0; step <= decode_step; ++step) {
            k_cache[request_id][new_beam][step].copy_(
              ori_k_cache[layer][request_id][old_beam][step]);
            v_cache[request_id][new_beam][step].copy_(
              ori_v_cache[layer][request_id][old_beam][step]);
          }
        }
      }
    }
  }
}

} // namespace xllm::kernel::cuda::triton