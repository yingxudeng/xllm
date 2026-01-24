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

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gflags/gflags.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <utility>

#include "common/global_flags.h"
#include "cuda.h"
#include "cuda_utils.h"
#include "topk_last_dim.cuh"

// ensure half type is available (consistent with topk_last_dim.cuh)
using half = __half;

namespace xllm::kernel::cuda {

std::pair<torch::Tensor, torch::Tensor> compute_topk_general(
    torch::Tensor input,
    uint32_t batch_size,
    uint32_t input_length,
    uint32_t k,
    torch::Device device,
    bool sorted) {
  input = input.contiguous();

  auto output_dtype = input.dtype();
  auto new_values =
      torch::empty({batch_size, k},
                   torch::TensorOptions().dtype(output_dtype).device(device));
  auto new_indices =
      torch::empty({batch_size, k},
                   torch::TensorOptions().dtype(torch::kInt32).device(device));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto workspace_size =
      reduce_topk::invokeComputeTopkLastDimWorkspaceSize<float>(
          static_cast<SizeType32>(batch_size),
          static_cast<SizeType32>(input_length),
          static_cast<SizeType32>(k),
          true,
          sorted);

  auto workspace =
      torch::empty({static_cast<int64_t>(workspace_size)},
                   torch::TensorOptions().dtype(torch::kUInt8).device(device));

  reduce_topk::invokeTopkLastDim<float>(static_cast<SizeType32>(batch_size),
                                        static_cast<SizeType32>(input_length),
                                        static_cast<SizeType32>(k),
                                        true,
                                        input.data_ptr<float>(),
                                        new_values.data_ptr<float>(),
                                        new_indices.data_ptr<int32_t>(),
                                        workspace.data_ptr<uint8_t>(),
                                        stream,
                                        sorted);

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return std::make_pair(new_values, new_indices);
}

// template function: wrap topK calculation logic, support different precision
// types
template <typename T>
std::pair<torch::Tensor, torch::Tensor> compute_topk_for_beam_search_impl(
    torch::Tensor combined_probs,
    uint32_t batch_size,
    uint32_t beam_size,
    uint32_t top_k,
    torch::Device device) {
  combined_probs = combined_probs.contiguous();

  // create output tensor, output type is the same as input type
  auto output_dtype = combined_probs.dtype();
  auto new_probs =
      torch::empty({batch_size, beam_size},
                   torch::TensorOptions().dtype(output_dtype).device(device));
  auto new_indices =
      torch::empty({batch_size, beam_size},
                   torch::TensorOptions().dtype(torch::kInt32).device(device));

  // get CUDA stream
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // calculate workspace size
  bool sorted = FLAGS_enable_topk_sorted;
  auto workspace_size = reduce_topk::invokeComputeTopkLastDimWorkspaceSize<T>(
      static_cast<SizeType32>(batch_size),
      static_cast<SizeType32>(beam_size * top_k),
      static_cast<SizeType32>(beam_size),
      true,
      sorted);  // is_largest = true

  // allocate workspace memory
  auto workspace =
      torch::empty({static_cast<int64_t>(workspace_size)},
                   torch::TensorOptions().dtype(torch::kUInt8).device(device));

  // call TensorRT-LLM's topK function
  // note: the data type of combined_probs must match the template parameter T
  reduce_topk::invokeTopkLastDim<T>(static_cast<SizeType32>(batch_size),
                                    static_cast<SizeType32>(beam_size * top_k),
                                    static_cast<SizeType32>(beam_size),
                                    true,  // is_largest = true
                                    combined_probs.data_ptr<T>(),
                                    new_probs.data_ptr<T>(),
                                    new_indices.data_ptr<int32_t>(),
                                    workspace.data_ptr<uint8_t>(),
                                    stream,
                                    sorted);

  // synchronize CUDA stream
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return std::make_pair(new_probs, new_indices);
}

// specialization for half (float16)
template <>
std::pair<torch::Tensor, torch::Tensor> compute_topk_for_beam_search_impl<half>(
    torch::Tensor combined_probs,
    uint32_t batch_size,
    uint32_t beam_size,
    uint32_t top_k,
    torch::Device device) {
  combined_probs = combined_probs.contiguous();

  // create output tensor, output type is the same as input type
  auto output_dtype = combined_probs.dtype();
  auto new_probs =
      torch::empty({batch_size, beam_size},
                   torch::TensorOptions().dtype(output_dtype).device(device));
  auto new_indices =
      torch::empty({batch_size, beam_size},
                   torch::TensorOptions().dtype(torch::kInt32).device(device));

  // get CUDA stream
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // calculate workspace size
  bool sorted = FLAGS_enable_topk_sorted;
  auto workspace_size =
      reduce_topk::invokeComputeTopkLastDimWorkspaceSize<half>(
          static_cast<SizeType32>(batch_size),
          static_cast<SizeType32>(beam_size * top_k),
          static_cast<SizeType32>(beam_size),
          true,
          sorted);  // is_largest = true

  // allocate workspace memory
  auto workspace =
      torch::empty({static_cast<int64_t>(workspace_size)},
                   torch::TensorOptions().dtype(torch::kUInt8).device(device));

  // call TensorRT-LLM's topK function
  // use at::Half for data_ptr, then cast to half* for CUDA kernel
  reduce_topk::invokeTopkLastDim<half>(
      static_cast<SizeType32>(batch_size),
      static_cast<SizeType32>(beam_size * top_k),
      static_cast<SizeType32>(beam_size),
      true,  // is_largest = true
      reinterpret_cast<half const*>(combined_probs.data_ptr<at::Half>()),
      reinterpret_cast<half*>(new_probs.data_ptr<at::Half>()),
      new_indices.data_ptr<int32_t>(),
      workspace.data_ptr<uint8_t>(),
      stream,
      sorted);

  // synchronize CUDA stream
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return std::make_pair(new_probs, new_indices);
}

// specialization for __nv_bfloat16
#ifdef ENABLE_BF16
template <>
std::pair<torch::Tensor, torch::Tensor>
compute_topk_for_beam_search_impl<__nv_bfloat16>(torch::Tensor combined_probs,
                                                 uint32_t batch_size,
                                                 uint32_t beam_size,
                                                 uint32_t top_k,
                                                 torch::Device device) {
  combined_probs = combined_probs.contiguous();

  // create output tensor, output type is the same as input type
  auto output_dtype = combined_probs.dtype();
  auto new_probs =
      torch::empty({batch_size, beam_size},
                   torch::TensorOptions().dtype(output_dtype).device(device));
  auto new_indices =
      torch::empty({batch_size, beam_size},
                   torch::TensorOptions().dtype(torch::kInt32).device(device));

  // get CUDA stream
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // calculate workspace size
  bool sorted = FLAGS_enable_topk_sorted;
  auto workspace_size =
      reduce_topk::invokeComputeTopkLastDimWorkspaceSize<__nv_bfloat16>(
          static_cast<SizeType32>(batch_size),
          static_cast<SizeType32>(beam_size * top_k),
          static_cast<SizeType32>(beam_size),
          true,
          sorted);  // is_largest = true

  // allocate workspace memory
  auto workspace =
      torch::empty({static_cast<int64_t>(workspace_size)},
                   torch::TensorOptions().dtype(torch::kUInt8).device(device));

  // call TensorRT-LLM's topK function
  // use at::BFloat16 for data_ptr, then cast to __nv_bfloat16* for CUDA kernel
  reduce_topk::invokeTopkLastDim<__nv_bfloat16>(
      static_cast<SizeType32>(batch_size),
      static_cast<SizeType32>(beam_size * top_k),
      static_cast<SizeType32>(beam_size),
      true,  // is_largest = true
      reinterpret_cast<__nv_bfloat16 const*>(
          combined_probs.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(new_probs.data_ptr<at::BFloat16>()),
      new_indices.data_ptr<int32_t>(),
      workspace.data_ptr<uint8_t>(),
      stream,
      sorted);

  // synchronize CUDA stream
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return std::make_pair(new_probs, new_indices);
}
#endif

// dispatch to the correct template instantiation based on the dtype of the
// input tensor
std::pair<torch::Tensor, torch::Tensor> compute_topk_for_beam_search(
    torch::Tensor combined_probs,
    uint32_t batch_size,
    uint32_t beam_size,
    uint32_t top_k,
    torch::Device device) {
  auto dtype = combined_probs.dtype();

  if (dtype == torch::kFloat32) {
    // ensure type match
    combined_probs = combined_probs.to(torch::kFloat32).contiguous();
    return compute_topk_for_beam_search_impl<float>(
        combined_probs, batch_size, beam_size, top_k, device);
  } else if (dtype == torch::kFloat16 || dtype == torch::kHalf) {
    // ensure type match, use half (consistent with topk_last_dim.cuh)
    combined_probs = combined_probs.to(torch::kFloat16).contiguous();
    return compute_topk_for_beam_search_impl<half>(
        combined_probs, batch_size, beam_size, top_k, device);
  } else if (dtype == torch::kBFloat16) {
#ifdef ENABLE_BF16
    // ensure type match
    combined_probs = combined_probs.to(torch::kBFloat16).contiguous();
    return compute_topk_for_beam_search_impl<__nv_bfloat16>(
        combined_probs, batch_size, beam_size, top_k, device);
#else
    // if BF16 is not supported, convert to float32
    combined_probs = combined_probs.to(torch::kFloat32).contiguous();
    return compute_topk_for_beam_search_impl<float>(
        combined_probs, batch_size, beam_size, top_k, device);
#endif
  } else {
    // default convert to float32
    combined_probs = combined_probs.to(torch::kFloat32).contiguous();
    return compute_topk_for_beam_search_impl<float>(
        combined_probs, batch_size, beam_size, top_k, device);
  }
}

template <typename T>
__global__ void beam_search_init_kernel(
    const int32_t* __restrict__ top_tokens,    // [batch_size, top_k]
    const T* __restrict__ top_logprobs,        // [batch_size, top_k]
    int32_t* __restrict__ out_token_ids,       // [batch_size, beam_size]
    T* __restrict__ out_acc_logprob,           // [batch_size, beam_size]
    int32_t* __restrict__ out_token_index,     // [batch_size * beam_size, 1]
    int32_t* __restrict__ out_sequence_group,  // [batch_size, beam_size,
                                               // total_rounds]
    uint32_t batch_size,
    uint32_t beam_size,
    uint32_t top_k,
    uint32_t total_rounds) {
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t total_elements = batch_size * beam_size;

  if (tid >= total_elements) return;

  const uint32_t batch_idx = tid / beam_size;
  const uint32_t beam_idx = tid % beam_size;

  // source index: read the first beam_size tokens from [batch_size, top_k]
  const uint32_t src_idx = batch_idx * top_k + beam_idx;

  // destination index
  const uint32_t dst_idx = batch_idx * beam_size + beam_idx;

  // copy all the tokens in one go
  T token = top_tokens[src_idx];
  out_token_ids[dst_idx] = token;
  out_acc_logprob[dst_idx] = top_logprobs[src_idx];
  out_token_index[tid] = static_cast<int32_t>(beam_idx);

  // out_sequence_group[:, :, 0] = tokens
  const uint32_t seq_idx =
      batch_idx * beam_size * total_rounds + beam_idx * total_rounds + 0;
  out_sequence_group[seq_idx] = token;
}

// initialize the beam search tensors for first step
void beam_search_init(torch::Tensor top_tokens,
                      torch::Tensor top_logprobs,
                      torch::Tensor out_token_ids,
                      torch::Tensor out_acc_logprob,
                      torch::Tensor out_token_index,
                      torch::Tensor out_sequence_group,
                      uint32_t batch_size,
                      uint32_t beam_size,
                      uint32_t top_k,
                      uint32_t total_rounds) {
  constexpr uint32_t kBlockSize = 256;
  const uint32_t total_elements = batch_size * beam_size;
  const uint32_t num_blocks = (total_elements + kBlockSize - 1) / kBlockSize;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto dtype = top_logprobs.dtype();

  if (dtype == torch::kFloat32) {
    beam_search_init_kernel<float><<<num_blocks, kBlockSize, 0, stream>>>(
        top_tokens.data_ptr<int32_t>(),
        top_logprobs.data_ptr<float>(),
        out_token_ids.data_ptr<int32_t>(),
        out_acc_logprob.data_ptr<float>(),
        out_token_index.data_ptr<int32_t>(),
        out_sequence_group.data_ptr<int32_t>(),
        batch_size,
        beam_size,
        top_k,
        total_rounds);
    //   } else if (dtype == torch::kFloat16) {
    //     beam_search_init_kernel<half><<<num_blocks, kBlockSize, 0, stream>>>(
    //         top_tokens.data_ptr<int32_t>(),
    //         top_logprobs.data_ptr<half>(),
    //         out_token_ids.data_ptr<int32_t>(),
    //         out_acc_logprob.data_ptr<half>(),
    //         out_token_index.data_ptr<int32_t>(),
    //         out_sequence_group.data_ptr<int32_t>(),
    //         batch_size,
    //         beam_size,
    //         top_k,
    //         total_rounds);
  } else if (dtype == torch::kBFloat16) {
#ifdef ENABLE_BF16
    beam_search_init_kernel<__nv_bfloat16>
        <<<num_blocks, kBlockSize, 0, stream>>>(
            top_tokens.data_ptr<int32_t>(),
            top_logprobs.data_ptr<__nv_bfloat16>(),
            out_token_ids.data_ptr<int32_t>(),
            out_acc_logprob.data_ptr<__nv_bfloat16>(),
            out_token_index.data_ptr<int32_t>(),
            out_sequence_group.data_ptr<int32_t>(),
            batch_size,
            beam_size,
            top_k,
            total_rounds);
#else
    beam_search_init_kernel<float><<<num_blocks, kBlockSize, 0, stream>>>(
        top_tokens.data_ptr<int32_t>(),
        top_logprobs.data_ptr<float>(),
        out_token_ids.data_ptr<int32_t>(),
        out_acc_logprob.data_ptr<float>(),
        out_token_index.data_ptr<int32_t>(),
        out_sequence_group.data_ptr<int32_t>(),
        batch_size,
        beam_size,
        top_k,
        total_rounds);
#endif
  } else {
    beam_search_init_kernel<float><<<num_blocks, kBlockSize, 0, stream>>>(
        top_tokens.data_ptr<int32_t>(),
        top_logprobs.data_ptr<float>(),
        out_token_ids.data_ptr<int32_t>(),
        out_acc_logprob.data_ptr<float>(),
        out_token_index.data_ptr<int32_t>(),
        out_sequence_group.data_ptr<int32_t>(),
        batch_size,
        beam_size,
        top_k,
        total_rounds);
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void beam_search(torch::Tensor acc_logprob,
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

  //   CHECK_EQ(beam_size, top_k) << "beam_size must be equal with top_k.";

  if (current_step == 0) {
    // NvtxRange range("==== beam_search step 0 ====");
    beam_search_init(top_tokens,
                     top_logprobs,
                     out_token_ids,
                     out_acc_logprob,
                     out_token_index,
                     out_sequence_group,
                     batch_size,
                     beam_size,
                     top_k,
                     total_rounds);

  } else {
    // NvtxRange range("==== beam_search step " + std::to_string(current_step) +
    // " ====");
    auto combined_probs =
        (acc_logprob + top_logprobs).view({batch_size, beam_size * top_k});

    // auto [new_probs, new_indices] = compute_topk_for_beam_search(
    //     combined_probs, batch_size, beam_size, top_k, device);

    auto topk_result = torch::topk(combined_probs,
                                   beam_size,
                                   -1,
                                   /*largest=*/true,
                                   /*sorted=*/FLAGS_enable_topk_sorted);
    auto new_probs = std::get<0>(topk_result);    // [batch_size, beam_size]
    auto new_indices = std::get<1>(topk_result);  // [batch_size, beam_size]

    // Reorder new_probs (and corresponding new_indices) to keep alignment
    // only when sorted output is requested.
    if (FLAGS_enable_topk_sorted && current_step < total_rounds - 1) {
      auto ordered_indices =
          new_indices.argsort(static_cast<int64_t>(1), false);
      new_probs = new_probs.gather(1, ordered_indices);
      new_indices = new_indices.gather(1, ordered_indices);
    }

    auto parent_beam = (new_indices / top_k).to(torch::kLong);
    auto token_in_beam = (new_indices % top_k).to(torch::kLong);

    auto top_tokens_reshaped = top_tokens.view({batch_size, beam_size, top_k});

    auto batch_idx =
        torch::arange(batch_size,
                      torch::TensorOptions().dtype(torch::kLong).device(device))
            .unsqueeze(1)
            .expand_as(parent_beam);

    using torch::indexing::TensorIndex;
    auto new_tokens = top_tokens_reshaped.index({TensorIndex(batch_idx),
                                                 TensorIndex(parent_beam),
                                                 TensorIndex(token_in_beam)});

    out_acc_logprob.view({batch_size, beam_size}).copy_(new_probs);
    out_token_index.view({batch_size, beam_size})
        .copy_(new_indices.to(torch::kInt32));
    out_token_ids.view({batch_size, beam_size}).copy_(new_tokens);

    auto batch_range =
        torch::arange(
            batch_size,
            torch::TensorOptions().dtype(torch::kInt32).device(device))
            .unsqueeze(1)
            .expand({-1, beam_size});
    auto beam_range =
        torch::arange(
            beam_size,
            torch::TensorOptions().dtype(torch::kInt32).device(device))
            .unsqueeze(0)
            .expand({batch_size, -1});

    using torch::indexing::Slice;
    using torch::indexing::TensorIndex;
    out_sequence_group.slice(2, 0, current_step) =
        in_sequence_group.index({TensorIndex(batch_range),
                                 TensorIndex(parent_beam.to(torch::kInt32)),
                                 Slice(0, current_step)});

    out_sequence_group.slice(2, current_step, current_step + 1) =
        new_tokens.unsqueeze(2);
  }
}

}  // namespace xllm::kernel::cuda
