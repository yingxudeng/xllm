/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "core/kernels/npu/pytorch_npu_helper.h"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {

at::Tensor quantize_per_tensor(
    const at::Tensor& self,
    const at::Tensor& scales,
    const at::Tensor& zero_points,
    at::ScalarType dtype,
    int64_t axis) {
  auto output_dtype = at::kInt;
  if (dtype == at::ScalarType::QInt8) {
    output_dtype = at::kChar;
  } else if (dtype == at::ScalarType::QUInt8) {
    output_dtype = at::kByte;
  } else if (dtype == at::ScalarType::QInt32) {
    output_dtype = at::kInt;
  }

  at::Tensor result =
      at::empty(self.sizes().vec(), self.options().dtype(output_dtype));
  EXEC_NPU_CMD(aclnnQuantize,
               self,
               scales,
               zero_points,
               output_dtype,
               axis,
               result);
  return result;
}

}  // namespace xllm::kernel::npu
