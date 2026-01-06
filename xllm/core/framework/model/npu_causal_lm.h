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

#include "causal_lm.h"
#include "graph/types.h"
#include "layers/npu/npu_lm_head_impl.h"
#include "layers/npu/npu_word_embedding_impl.h"

namespace xllm {

// NPU-specific CausalLM interface that extends the base CausalLM with
// NPU-specific operations
class NPUCausalLM : public CausalLM {
 public:
  ~NPUCausalLM() override = default;

  // NPU-specific interfaces for getting/setting lm_head and word_embedding
  virtual layer::NpuLmHead get_npu_lm_head() = 0;
  virtual void set_npu_lm_head(layer::NpuLmHead& head) = 0;
  virtual layer::NpuWordEmbedding get_npu_word_embedding() = 0;
  virtual void set_npu_word_embedding(layer::NpuWordEmbedding& embedding) = 0;
};

// NPU-specific implementation template
template <typename Model>
class NPUCausalLMImpl : public NPUCausalLM {
 public:
  NPUCausalLMImpl(Model model, const torch::TensorOptions& options)
      : model_(std::move(model)), options_(options) {}

  // Implement base CausalLM interfaces
  torch::Tensor forward(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const ModelInputParams& parameters) override {
    return model_->forward(tokens, positions, kv_caches, parameters);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) override {
    return model_->logits(hidden_states, seleted_idxes);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) override {
    model_->load_model(std::move(loader));
  }

  void prepare_expert_weight(int32_t layer_id,
                             const std::vector<int32_t>& expert_ids) override {
    return model_->prepare_expert_weight(layer_id, expert_ids);
  }

  void update_expert_weight(int32_t layer_id) override {
    return model_->update_expert_weight(layer_id);
  }

  torch::Device device() const override { return options_.device(); }

  const torch::TensorOptions& options() const override { return options_; }

  // Implement NPU-specific interfaces
  layer::NpuLmHead get_npu_lm_head() override {
    return model_->get_npu_lm_head();
  }

  void set_npu_lm_head(layer::NpuLmHead& head) override {
    model_->set_npu_lm_head(head);
  }

  layer::NpuWordEmbedding get_npu_word_embedding() override {
    return model_->get_npu_word_embedding();
  }

  void set_npu_word_embedding(layer::NpuWordEmbedding& embedding) override {
    model_->set_npu_word_embedding(embedding);
  }

  // Optional: Provide default implementations for the base class's get/set
  // methods These can be empty or delegate to NPU-specific versions if needed
  layer::LmHead get_lm_head() override {
    LOG(FATAL) << "Method 'get_lm_head' is not supported by NPU models. Use "
                  "'get_npu_lm_head' instead.";
    return nullptr;  // Unreachable, but needed for compilation
  }

  void set_lm_head(layer::LmHead& head) override {
    LOG(FATAL) << "Method 'set_lm_head' is not supported by NPU models. Use "
                  "'set_npu_lm_head' instead.";
  }

  layer::WordEmbedding get_word_embedding() override {
    LOG(FATAL) << "Method 'get_word_embedding' is not supported by NPU "
                  "models. Use 'get_npu_word_embedding' instead.";
    return nullptr;  // Unreachable, but needed for compilation
  }

  void set_word_embedding(layer::WordEmbedding& embedding) override {
    LOG(FATAL) << "Method 'set_word_embedding' is not supported by NPU "
                  "models. Use 'set_npu_word_embedding' instead.";
  }

 private:
  Model model_;
  torch::TensorOptions options_;
};

}  // namespace xllm
