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

#pragma once

#include <cstdint>
#include <string>

#ifdef USE_CUDA
#include <nvToolsExt.h>
#include <cuda_runtime.h>
#endif

namespace xllm {

#ifdef USE_CUDA
class NvtxRange {
 public:
  explicit NvtxRange(const char* name);
  explicit NvtxRange(const char* name, uint32_t color);

  ~NvtxRange();

 private:
  std::string name_;
  bool active_;
};

#define CONCATENATE(x, y) x##y

#define LLM_NVTX_RANGE(name) \
  NvtxRange CONCATENATE(llm_nvtx_range_, __LINE__) { name }

#define LLM_NVTX_RANGE_COLOR(name, color) \
  NvtxRange CONCATENATE(llm_nvtx_range_, __LINE__) { name, color }

#define LLM_NVTX_RANGE_FUNC() \
  NvtxRange CONCATENATE(llm_nvtx_range_, __LINE__) { __PRETTY_FUNCTION__ }
#else
// Dummy implementation when CUDA is not available
class NvtxRange {
 public:
  explicit NvtxRange(const char* name) {}
  explicit NvtxRange(const char* name, uint32_t color) {}
  ~NvtxRange() {}
};

#define LLM_NVTX_RANGE(name) \
  do {                       \
  } while (0)

#define LLM_NVTX_RANGE_COLOR(name, color) \
  do {                                    \
  } while (0)

#define LLM_NVTX_RANGE_FUNC() \
  do {                        \
  } while (0)
#endif

}  // namespace xllm

