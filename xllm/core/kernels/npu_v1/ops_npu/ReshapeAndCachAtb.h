// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// #include "op_plugin/OpApiInterface.h"
#include <acl/acl.h>

#include "../custom_functions_npu/AtbCommon.h"

using namespace std;
namespace atb {
using ReshapeAndCacheParam = atb::infer::ReshapeAndCacheParam;
void _npu_reshape_and_cache(const at::Tensor& key,
                            const at::Tensor& value,
                            at::Tensor& key_cache,
                            at::Tensor& value_cache,
                            const at::Tensor& slot_indices);

}  // namespace atb
