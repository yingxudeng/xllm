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

#include "graph/types.h"
#include "layers/npu/npu_lm_head_impl.h"
#include "layers/npu/npu_word_embedding_impl.h"
#include "mm_embedding_vlm.h"

namespace xllm {

// NPU-specific MMEmbeddingVLM interface
class NPUMMEmbeddingVLM : public MMEmbeddingVLM {
 public:
  ~NPUMMEmbeddingVLM() override = default;

  // NPU-specific interfaces (may not be used by all MM embedding models)
  virtual layer::NpuLmHead get_npu_lm_head() = 0;
  virtual void set_npu_lm_head(layer::NpuLmHead& head) = 0;
  virtual layer::NpuWordEmbedding get_npu_word_embedding() = 0;
  virtual void set_npu_word_embedding(layer::NpuWordEmbedding& embedding) = 0;
};

// NPU-specific MMEmbeddingVLM implementation
template <typename Model>
class NPUMMEmbeddingVLMImpl : public NPUMMEmbeddingVLM {
 public:
  NPUMMEmbeddingVLMImpl(Model model, const torch::TensorOptions& options)
      : model_(std::move(model)), options_(options) {}

  // Implement MMEmbeddingVLM interface
  std::vector<torch::Tensor> encode(
      const ModelInputParams& input_params) override {
    return model_->encode(input_params);
  }

  // Implement base CausalLM interfaces
  torch::Tensor forward(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const ModelInputParams& input_params) override {
    return torch::Tensor{};  // MM Embedding models typically don't use forward
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) override {
    return torch::Tensor();  // MM Embedding models don't produce logits
  }

  void load_model(std::unique_ptr<ModelLoader> loader) override {
    model_->load_model(std::move(loader));
  }

  void prepare_expert_weight(int32_t layer_id,
                             const std::vector<int32_t>& expert_ids) override {
    return;  // Most MM embedding models don't have MoE
  }

  void update_expert_weight(int32_t layer_id) override {
    return;  // Most MM embedding models don't have MoE
  }

  torch::Device device() const override { return options_.device(); }

  const torch::TensorOptions& options() const override { return options_; }

  // Implement NPU-specific interfaces with default empty implementations
  // Most MM embedding models don't actually use lm_head or word_embedding
  layer::NpuLmHead get_npu_lm_head() override {
    return nullptr;  // MM embedding models typically don't have lm_head
  }

  void set_npu_lm_head(layer::NpuLmHead& head) override {
    // Empty implementation - MM embedding models don't need lm_head
  }

  layer::NpuWordEmbedding get_npu_word_embedding() override {
    return nullptr;  // MM embedding models typically don't have word_embedding
  }

  void set_npu_word_embedding(layer::NpuWordEmbedding& embedding) override {
    // Empty implementation - MM embedding models don't need word_embedding
  }

  // Override base class methods with FATAL logs
  layer::LmHead get_lm_head() override {
    LOG(FATAL)
        << "Method 'get_lm_head' is not supported by NPU MM embedding VLM "
           "models. Use 'get_npu_lm_head' instead.";
    return nullptr;
  }

  void set_lm_head(layer::LmHead& head) override {
    LOG(FATAL)
        << "Method 'set_lm_head' is not supported by NPU MM embedding VLM "
           "models. Use 'set_npu_lm_head' instead.";
  }

  layer::WordEmbedding get_word_embedding() override {
    LOG(FATAL) << "Method 'get_word_embedding' is not supported by NPU MM "
                  "embedding VLM models. Use 'get_npu_word_embedding' instead.";
    return nullptr;
  }

  void set_word_embedding(layer::WordEmbedding& embedding) override {
    LOG(FATAL) << "Method 'set_word_embedding' is not supported by NPU MM "
                  "embedding VLM models. Use 'set_npu_word_embedding' instead.";
  }

 private:
  Model model_;
  torch::TensorOptions options_;
};

}  // namespace xllm
