#pragma once

#include <memory>

#include "core/framework/model/embedding_lm.h"
#include "embedding_model_base.h"
#include "qwen3.h"

namespace xllm {

class QWen3ForEmbeddingImpl : public LlmForEmbeddingImplBase<QWen3Model> {
 public:
  QWen3ForEmbeddingImpl(const ModelContext& context)
      : LlmForEmbeddingImplBase<QWen3Model>(context),
        options_(context.get_tensor_options()) {}

  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    auto h = hidden_states;
    if (seleted_idxes.defined()) {
      h = h.index_select(/*dim=*/0, seleted_idxes);
    }
    auto pooler_output = torch::nn::functional::normalize(
        h, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
    return pooler_output;
  }

  torch::Device device() const { return options_.device(); }
  const torch::TensorOptions& options() const { return options_; }

 private:
  torch::TensorOptions options_;
};
TORCH_MODULE(QWen3ForEmbedding);

// Use NPU-specific registration for this model
REGISTER_NPU_EMBEDDING_MODEL_WITH_VARNAME(qwen3_embedding,
                                          qwen3,
                                          QWen3ForEmbedding);
}  // namespace xllm