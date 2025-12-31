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

#include "common/nvtx_helper.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace xllm {

#ifdef USE_CUDA
NvtxRange::NvtxRange(const char* name) : name_(name), active_(true) {
  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = 0xFF00FF00;  // Green color
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii = name_.c_str();
  nvtxRangePushEx(&eventAttrib);
}

NvtxRange::NvtxRange(const char* name, uint32_t color)
    : name_(name), active_(true) {
  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = color;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii = name_.c_str();
  nvtxRangePushEx(&eventAttrib);
}

NvtxRange::~NvtxRange() {
  if (active_) {
    nvtxRangePop();
  }
}
#endif

}  // namespace xllm

