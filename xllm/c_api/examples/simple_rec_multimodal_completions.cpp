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

#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "rec.h"

std::string devices = "cuda:1";
std::string model_name = "Qwen3-8B";
std::string model_path = "/export/home/models/Qwen3-8B";

class XLLM_MM_Data_Wrapper {
 public:
  XLLM_MM_Data_Wrapper() = default;

  ~XLLM_MM_Data_Wrapper() { reset(); }

  XLLM_MM_Data_Wrapper(const XLLM_MM_Data_Wrapper&) = delete;
  XLLM_MM_Data_Wrapper& operator=(const XLLM_MM_Data_Wrapper&) = delete;

  bool build(
      const std::vector<std::pair<uint32_t, uint32_t>>& token_positions) {
    if (is_built_ || token_positions.empty()) {
      fprintf(stderr,
              "build() failed: already built or empty token positions\n");
      return false;
    }

    mm_data_.type_mask = static_cast<uint32_t>(XLLM_MM_TYPE_EMBEDDING);
    mm_data_.is_dict = false;

    for (size_t i = 0; i < token_positions.size(); ++i) {
      const auto& [offset, length] = token_positions[i];

      if (length == 0) {
        fprintf(stderr, "build() skipped item %zu: length cannot be 0\n", i);
        continue;
      }

      items_.emplace_back(create_embedding_item(offset, length));
    }

    if (items_.empty()) {
      fprintf(stderr, "build() failed: no valid embedding items created\n");
      return false;
    }

    mm_data_.data.items.entries_size = items_.size();
    mm_data_.data.items.entries = items_.data();

    is_built_ = true;
    return true;
  }

  void reset() {
    memset(&mm_data_, 0, sizeof(mm_data_));

    items_.clear();
    tensor_buffers_.clear();
    is_built_ = false;
  }

  const XLLM_MM_Data* get_data() const {
    return is_built_ ? &mm_data_ : nullptr;
  }

  void validate() const {
    if (!is_built_) {
      fprintf(stderr,
              "validate() failed: no data available (call build() first)\n");
      return;
    }

    const size_t item_count = mm_data_.data.items.entries_size;
    printf("=== Validating %zu Embedding Items ===\n\n", item_count);

    for (size_t i = 0; i < item_count; ++i) {
      const auto& item = mm_data_.data.items.entries[i];
      printf("=== Embedding Item %zu ===\n", i + 1);
      printf("Token Position: offset=%u, length=%u\n",
             item.state.token_pos.offset,
             item.state.token_pos.length);
      printf("Data Type: FLOAT32 (XLLM_DTYPE_FLOAT32=%d)\n",
             item.data.data.tensor.dtype);
      printf("Tensor Shape: rank=%d, dim=[%d, %d]\n\n",
             item.data.data.tensor.dims.rank,
             item.data.data.tensor.dims.dim[0],
             item.data.data.tensor.dims.dim[1]);
    }
  }

  bool is_built() const { return is_built_; }

  size_t get_item_count() const {
    return is_built_ ? mm_data_.data.items.entries_size : 0;
  }

 private:
  XLLM_MM_Data mm_data_{};
  std::vector<XLLM_MM_Item> items_;
  std::vector<std::unique_ptr<float[]>> tensor_buffers_;
  bool is_built_ = false;

  XLLM_MM_Item create_embedding_item(uint32_t offset, uint32_t length) {
    XLLM_MM_Item item{};

    item.type = XLLM_MM_TYPE_EMBEDDING;
    item.state.token_pos.offset = offset;
    item.state.token_pos.length = length;

    item.data.is_single_tensor = true;
    item.data.data.tensor.dtype = XLLM_DTYPE_FLOAT32;
    item.data.data.tensor.dims.rank = 2;
    memset(item.data.data.tensor.dims.dim,
           0,
           sizeof(item.data.data.tensor.dims.dim));
    item.data.data.tensor.dims.dim[0] = static_cast<int>(length);
    item.data.data.tensor.dims.dim[1] = 4096;

    const size_t buffer_size = length * 4096;
    auto buffer = std::make_unique<float[]>(buffer_size);
    for (size_t i = 0; i < length; ++i) {
      for (size_t j = 0; j < 4096; ++j) {
        buffer[i * 4096 + j] =
            static_cast<float>(i * 4096 + j) / static_cast<float>(buffer_size);
      }
    }

    item.data.data.tensor.data = buffer.get();
    tensor_buffers_.push_back(std::move(buffer));

    return item;
  }
};

XLLM_REC_Handler* service_startup_hook() {
  XLLM_REC_Handler* rec_handler = xllm_rec_create();

  // If there is no separate setting, init_options can be passed as nullptr, and
  // the default value(XLLM_INIT_REC_OPTIONS_DEFAULT) will be used
  XLLM_InitOptions init_options;
  xllm_rec_init_options_default(&init_options);
  snprintf(
      init_options.log_dir, sizeof(init_options.log_dir), "/export/xllm/log");

  bool ret = xllm_rec_initialize(
      rec_handler, model_path.c_str(), devices.c_str(), &init_options);
  if (!ret) {
    std::cout << "REC init failed" << std::endl;
    xllm_rec_destroy(rec_handler);
    return nullptr;
  }

  std::cout << "REC init successfully" << std::endl;

  return rec_handler;
}

void service_stop_hook(XLLM_REC_Handler* rec_handler) {
  xllm_rec_destroy(rec_handler);
  std::cout << "REC stop" << std::endl;
}

int generate_random_int(int min, int max) {
  if (min > max) {
    throw std::invalid_argument("min cannot be greater than max");
  }

  static std::random_device rd;
  static std::mt19937 gen(rd());

  std::uniform_int_distribution<int> dist(min, max);

  return dist(gen);
}

int main(int argc, char** argv) {
  XLLM_REC_Handler* rec_handler = service_startup_hook();
  if (nullptr == rec_handler) {
    return -1;
  }

  // If there is no separate setting, request_params can be passed as nullptr,
  // and the default value(XLLM_REQUEST_PARAMS_DEFAULT) will be used
  XLLM_RequestParams request_params;
  xllm_rec_request_params_default(&request_params);
  request_params.beam_width = 128;

  size_t token_size = 512;
  int token_ids[512] = {0};
  for (int i = 0; i < token_size; i++) {
    token_ids[i] = generate_random_int(100, 150000);
  }

  XLLM_MM_Data_Wrapper multimodal_data_wrapper;
  std::vector<std::pair<uint32_t, uint32_t>> positions = {{100, 32}, {300, 64}};
  multimodal_data_wrapper.build(positions);
  multimodal_data_wrapper.validate();

  XLLM_Response* resp =
      xllm_rec_multimodal_completions(rec_handler,
                                      model_name.c_str(),
                                      token_ids,
                                      token_size,
                                      multimodal_data_wrapper.get_data(),
                                      10000,
                                      &request_params);
  if (nullptr == resp) {
    std::cout << "REC completions failed, response is nullptr" << std::endl;
    service_stop_hook(rec_handler);
    return -1;
  }

  if (resp->status_code != XLLM_StatusCode::kSuccess) {
    std::cout << "REC completions failed, status code:" << resp->status_code
              << ", error info:" << resp->error_info << std::endl;
  } else {
    std::cout << "REC completions successfully" << std::endl;

    if (nullptr != resp->choices.entries) {
      for (int i = 0; i < resp->choices.entries_size; ++i) {
        XLLM_Choice& choice = resp->choices.entries[i];
        for (int j = 0; j < choice.logprobs.entries_size; j++) {
          XLLM_LogProb& logprob = choice.logprobs.entries[j];
          std::cout << "xllm answer[" << choice.index
                    << "]: token id=" << logprob.token_id
                    << ", token logprob=" << logprob.logprob << std::endl;
        }
      }
    }
  }

  xllm_rec_free_response(resp);

  service_stop_hook(rec_handler);

  return 0;
}