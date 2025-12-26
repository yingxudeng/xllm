// #include "rec_triton.h"

// #include <ATen/cuda/CUDAContext.h>
// #include <cuda_runtime.h>
// #include "cuda.h"

// #include <torch/script.h>
// #include <torch/torch.h>

// #include "ptx_kernels/beam_search_sm_90.ptx.h"

// namespace xllm::kernel::cuda::triton {
// RecTritonKernel::RecTritonKernel() {
//   auto arch = get_cuda_arch();
//   if (arch == ARCH::SM_90) {
//     PROCESS_KERNEL(sm_90,
//                    fp32,
//                    BeamSearchKernelConfig,
//                    fp32_beam_search_configs_,
//                    fp32_beam_search_input_dim_array_,
//                    beam_search_kernel,
//                    BLOCK_SINGLE(beam_search_kernel_sm_90_fp32));      
//   } else {
//     LOG(FATAL) << "do not support arch but SM_90.";
//   }
// }

// RecTritonKernel::~RecTritonKernel() {

// }

// torch::Tensor RecTritonKernel::xattention(torch::Tensor q,                    // [batch_size, beam_size, num_heads, head_dim]
//                                          torch::Tensor shared_k_cache,       // [num_shared_kv_seq_len, kv_heads, head_dim]
//                                          torch::Tensor shared_v_cache,       // [num_shared_kv_seq_len, kv_heads, head_dim]
//                                          torch::Tensor unshared_k_cache,     // [max_num_request, beam_size, kv_heads, max_decode_step, head_dim]
//                                          torch::Tensor unshared_v_cache,     // [max_num_request, beam_size, kv_heads, max_decode_step, head_dim]
//                                          torch::Tensor kv_seq_len,           // [batch_size]
//                                          torch::Tensor block_table,          // [total_beams]
//                                          float sm_scale,
//                                          uint32_t step) {
//   LOG(FATAL) << "gpu xattention is not implemented by torch.";
  
// }


// void RecTritonKernel::prefill_reshape_and_cache(torch::Tensor proj_k,          // [shared_len, kv_heads, head_dim]
//                                                torch::Tensor proj_v,          // [shared_len, kv_heads, head_dim]
//                                                torch::Tensor shared_k_cache,  // [num_shared_kv_seq_len, kv_heads, head_dim]
//                                                torch::Tensor shared_v_cache   // [num_shared_kv_seq_len, kv_heads, head_dim]
//                                                ) {
//   LOG(FATAL) << "gpu prefill_reshape_and_cache is not implemented by torch.";
  
// }

// void RecTritonKernel::decoder_reshape_and_cache(torch::Tensor proj_k,          // [batch_size, beam_size, kv_heads, head_dim]
//                                                torch::Tensor proj_v,          // [batch_size, beam_size, kv_heads, head_dim]
//                                                torch::Tensor unshared_k_cache,  // [max_num_request, beam_size, max_decode_step, kv_heads, head_dim]
//                                                torch::Tensor unshared_v_cache,   // [max_num_request, beam_size, max_decode_step, kv_heads, head_dim]
//                                                torch::Tensor block_table,     // [batch_size, 1]
//                                                uint32_t step) {
//   // 维度检查
//   LOG(FATAL) << "gpu decoder_reshape_and_cache is not implemented by torch.";
// }


// void RecTritonKernel::beam_search(torch::Tensor acc_logprob,              // [B*BEAM_SIZE, 1]
//                                   torch::Tensor in_sequence_group,        // [B, BEAM_SIZE, MAX_DECODE_STEP] (Read)
//                                   torch::Tensor top_tokens,               // [B*BEAM_SIZE, TOP_K]
//                                   torch::Tensor top_logprobs,             // [B*BEAM_SIZE, TOP_K]
//                                   torch::Tensor out_acc_logprob,          // [B*BEAM_SIZE, 1]
//                                   torch::Tensor out_token_ids,            // [B*BEAM_SIZE, 1]
//                                   torch::Tensor out_token_index,          // [B*BEAM_SIZE, 1]
//                                   torch::Tensor out_beam_count_prefix_sums, // [B*BEAM_SIZE, 1]
//                                   torch::Tensor out_sequence_group,       // [B, BEAM_SIZE, MAX_DECODE_STEP] (Write)
//                                   uint32_t batch_size,
//                                   uint32_t current_step) {
//   CUfunction beam_search_kernel;
//   uint32_t beam_search_shared_mem_bytes = 0;
//   int32_t total_beams = acc_logprob.size(0);
//   int32_t beam_size = total_beams / batch_size;

//   switch (top_logprobs.scalar_type()) {
//     case torch::Float32: {
//       const auto input_dim =
//           std::array<int, 1>{static_cast<int>(beam_size)};
//       const auto& beam_search_config =
//           fp32_beam_search_configs_.find_closest_index(
//               fp32_beam_search_input_dim_array_, input_dim);
//       beam_search_kernel = beam_search_config.kernel;

//       beam_search_shared_mem_bytes =
//           beam_search_config.shared_mem_bytes;
//       break;
//     }
//     default: {
//       LOG(FATAL) << "RecTritonKernel: beam_search_kernel get upsupported dtype "
//                  << top_logprobs.scalar_type();
//       break;
//     }
//   }
//   LOG(INFO) << "beam_search_shared_mem_bytes: " << beam_search_shared_mem_bytes;

//   CUdeviceptr acc_logprob_ptr =
//       reinterpret_cast<CUdeviceptr>(acc_logprob.data_ptr());
//   CUdeviceptr in_sequence_group_ptr =
//       reinterpret_cast<CUdeviceptr>(in_sequence_group.data_ptr());
//   CUdeviceptr top_tokens_ptr =
//       reinterpret_cast<CUdeviceptr>(top_tokens.data_ptr());
//   CUdeviceptr top_logprobs_ptr =
//       reinterpret_cast<CUdeviceptr>(top_logprobs.data_ptr());
//   CUdeviceptr out_acc_logprob_ptr =
//       reinterpret_cast<CUdeviceptr>(out_acc_logprob.data_ptr());
//   CUdeviceptr out_token_ids_ptr =
//       reinterpret_cast<CUdeviceptr>(out_token_ids.data_ptr());
//   CUdeviceptr out_token_index_ptr =
//       reinterpret_cast<CUdeviceptr>(out_token_index.data_ptr());
//   CUdeviceptr out_beam_count_prefix_sums_ptr =
//       reinterpret_cast<CUdeviceptr>(out_beam_count_prefix_sums.data_ptr());
//   CUdeviceptr out_sequence_group_ptr =
//       reinterpret_cast<CUdeviceptr>(out_sequence_group.data_ptr());
// // LOG(INFO) << "before at::cuda::getCurrentCUDAStream().";
  
//   auto stream = at::cuda::getCurrentCUDAStream();
//   // uint32_t batch_size = in_sequence_group.size(0);
// // LOG(INFO) << "batch_size: " << batch_size;
// // LOG(INFO) << "current_step: " << current_step;
//   // uint32_t beam_size = in_sequence_group.size(1);
// // LOG(INFO) << "beam_size: " << beam_size;
//   uint32_t top_k = top_tokens.size(1);
// // LOG(INFO) << "top_k: " << top_k;
//   CHECK_EQ(in_sequence_group.dim(), 3) << "in_sequence_group.dim() must be equal with 3.";
  
//   uint32_t max_decode_step = in_sequence_group.size(2);
//   void* args[] = {(void*)&acc_logprob_ptr,
//                   (void*)&in_sequence_group_ptr,
//                   (void*)&top_tokens_ptr,
//                   (void*)&top_logprobs_ptr,
//                   (void*)&out_acc_logprob_ptr,
//                   (void*)&out_token_ids_ptr,
//                   (void*)&out_token_index_ptr,
//                   (void*)&out_beam_count_prefix_sums_ptr,
//                   (void*)&out_sequence_group_ptr, 
//                   (void*)&batch_size, 
//                   (void*)&max_decode_step, 
//                   (void*)&current_step, 
//                   (void*)&current_step};
//   uint32_t thread_nums = get_default_thread_nums();
// // LOG(INFO) << "thread_nums: " << thread_nums;
// // LOG(INFO) << "before cuLaunchKernel.";
//   CUDA_CHECK(cuLaunchKernel(beam_search_kernel,
//                             batch_size,
//                             1,
//                             1,
//                             thread_nums,
//                             1,
//                             1,
//                             beam_search_shared_mem_bytes,
//                             stream,
//                             args,
//                             NULL));
// }

// } // namespace xllm::kernel::cuda::triton