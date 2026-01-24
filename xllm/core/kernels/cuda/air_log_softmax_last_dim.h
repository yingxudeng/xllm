/* Copyright 2025 The xLLM Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://github.com/jd-opensource/xllm/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================*/

#pragma once

#include <torch/torch.h>

namespace xllm::kernel::cuda {

// LogSoftmax on the last dimension for 2D input [B, K].
//
// - input must be CUDA and 2D.
// - temperatures is optional. When defined, it must be CUDA, 1D [B], and
// float32.
//   A temperature value of 0 is treated as 1.
// - output is float32 CUDA tensor [B, K].
torch::Tensor air_log_softmax_last_dim(const torch::Tensor& input,
                                       const torch::Tensor& temperatures);

}  // namespace xllm::kernel::cuda
