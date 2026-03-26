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
#include <optional>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "core/framework/model/model_output.h"
#include "core/layers/common/lm_head.h"
#include "core/layers/npu_torch/qwen3_5_decoder_layer_impl.h"
#include "core/layers/qwen3_vision_layer.h"
#include "models/model_registry.h"
#if defined(USE_NPU)
#include "models/vlm/npu/qwen3_vl.h"
#else
#include "models/vlm/qwen3_vl.h"
#endif
#include "processors/input_processor.h"
#include "processors/qwen2_vl_image_processor.h"
#include "qwen3_next.h"

namespace xllm {

#if defined(USE_NPU)
using Qwen3_5VisionTransformer = npu::model::Qwen3_VisionTransformer;
#else
using Qwen3_5VisionTransformer = Qwen3_VisionTransformer;
#endif

class Qwen3_5ModelImpl : public Qwen3NextModelImpl {
 public:
  explicit Qwen3_5ModelImpl(const ModelContext& context)
      : Qwen3NextModelImpl(context, /*init_decoder_layers=*/false) {
    const int32_t n_layers = context.get_model_args().n_layers();
    for (int32_t layer_id = 0; layer_id < n_layers; ++layer_id) {
      add_decoder_layer(
          std::make_shared<layer::Qwen3_5DecoderLayerImpl>(context, layer_id));
    }
  }
};
TORCH_MODULE(Qwen3_5Model);

class Qwen3_5ForCausalLMImpl : public Qwen3NextForCausalLMImpl {
 public:
  explicit Qwen3_5ForCausalLMImpl(const ModelContext& context)
      : Qwen3NextForCausalLMImpl(context, /*init_model=*/false) {
    set_model_module(std::make_shared<Qwen3_5ModelImpl>(context));
  }
};
TORCH_MODULE(Qwen3_5ForCausalLM);

class Qwen3_5VLInputProcessor : public InputProcessor {
  enum class TokenType {
    INVALID,
    IMAGE,
    VIDEO,
  };

 public:
  explicit Qwen3_5VLInputProcessor(const ModelArgs& args) {
    merge_size_ = args.mm_image_merge_size();
    vision_start_token_id_ = args.vision_start_token_id();
    vision_end_token_id_ = args.vision_end_token_id();
    image_token_id_ = args.image_token_id();
    video_token_id_ = args.video_token_id();
  }

  void process(std::string& prompt, const MMData& mm_data) override {
    torch::Tensor image_grid_thw;
    if (auto res = mm_data.get<torch::Tensor>("image_grid_thw")) {
      image_grid_thw = res.value();
    }

    torch::Tensor video_grid_thw;
    if (auto res = mm_data.get<torch::Tensor>("video_grid_thw")) {
      video_grid_thw = res.value();
    }

    if (!image_grid_thw.defined() && !video_grid_thw.defined()) {
      return;
    }

    const auto merge_length = merge_size_ * merge_size_;
    int total_image_token = 0;
    if (image_grid_thw.defined()) {
      const auto count = image_grid_thw.sizes()[0];
      for (int idx = 0; idx < count; ++idx) {
        total_image_token +=
            image_grid_thw[idx].prod().item<int>() / merge_length;
      }
    }

    int total_video_token = 0;
    if (video_grid_thw.defined()) {
      const auto count = video_grid_thw.sizes()[0];
      for (int idx = 0; idx < count; ++idx) {
        total_video_token +=
            video_grid_thw[idx].prod().item<int>() / merge_length;
      }
    }

    const size_t total_token_len = total_image_token * image_token_.size() +
                                   total_video_token * video_token_.size();
    std::string data;
    data.reserve(prompt.size() + total_token_len);

    int image_index = 0;
    int video_index = 0;

    const torch::Tensor* grid_thw = nullptr;
    const std::string* token = nullptr;
    int* index = nullptr;

    size_t begin = 0;
    auto [token_type, start_pos, end_pos] = find_vision_token(prompt, begin);

    while (start_pos != std::string::npos) {
      data.append(prompt, begin, start_pos - begin);
      if (token_type == TokenType::IMAGE) {
        grid_thw = &image_grid_thw;
        token = &image_token_;
        index = &image_index;
      } else if (token_type == TokenType::VIDEO) {
        grid_thw = &video_grid_thw;
        token = &video_token_;
        index = &video_index;
      } else {
        LOG(FATAL) << "The token between vision_start_token_id and "
                   << "vision_end_token_id should be image_token_id or "
                   << "video_token_id, but got: "
                   << prompt.substr(start_pos, end_pos - start_pos);
      }

      CHECK(grid_thw != nullptr);
      CHECK(index != nullptr);
      CHECK(token != nullptr);
      CHECK(*index < grid_thw->sizes()[0]);

      const int repeat = grid_thw->index({*index}).prod().item<int>() /
                         merge_size_ / merge_size_;
      for (int i = 0; i < repeat; ++i) {
        data.append(*token);
      }
      ++(*index);
      begin = end_pos;
      std::tie(token_type, start_pos, end_pos) =
          find_vision_token(prompt, begin);
    }

    data.append(prompt, begin, prompt.size() - begin);
    prompt = std::move(data);
  }

  void find_mm_spans(const std::vector<int>& prompt, MMData& mm_data) override {
    auto start = prompt.begin();
    uint32_t global_mm_index = 0;
    auto& mm_items = mm_data.items<MMItemVec>();

    while (true) {
      auto vision_start_it =
          std::find(start, prompt.end(), vision_start_token_id_);
      auto vision_end_it = std::find(start, prompt.end(), vision_end_token_id_);
      if (vision_start_it == prompt.end() || vision_end_it == prompt.end()) {
        break;
      }

      const uint32_t offset = std::distance(prompt.begin(), vision_start_it);
      const uint32_t length = std::distance(vision_start_it + 1, vision_end_it);

      CHECK(global_mm_index < mm_items.size());
      auto& item = mm_items[global_mm_index];
      if (*(vision_start_it + 1) == image_token_id_ ||
          *(vision_start_it + 1) == video_token_id_) {
        item.mutable_state().mutable_token_pos() = {offset + 1, length};
      }

      ++global_mm_index;
      start = std::next(vision_end_it);
    }
  }

 private:
  std::tuple<TokenType, size_t, size_t> find_vision_token(
      const std::string& prompt,
      size_t begin) {
    auto vision_start_pos = prompt.find(vision_start_token_, begin);
    if (vision_start_pos == std::string::npos) {
      return {TokenType::INVALID, std::string::npos, std::string::npos};
    }

    auto vision_end_pos = prompt.find(
        vision_end_token_, vision_start_pos + vision_start_token_.size());
    if (vision_end_pos == std::string::npos) {
      return {TokenType::INVALID, std::string::npos, std::string::npos};
    }

    auto image_pos = prompt.find(image_token_,
                                 vision_start_pos + vision_start_token_.size());
    auto video_pos = prompt.find(video_token_,
                                 vision_start_pos + vision_start_token_.size());

    TokenType token_type = TokenType::INVALID;
    size_t start_pos = std::string::npos;
    size_t end_pos = std::string::npos;
    if (image_pos != std::string::npos && image_pos < vision_end_pos) {
      token_type = TokenType::IMAGE;
      start_pos = image_pos;
      end_pos = image_pos + image_token_.size();
    } else if (video_pos != std::string::npos && video_pos < vision_end_pos) {
      token_type = TokenType::VIDEO;
      start_pos = video_pos;
      end_pos = video_pos + video_token_.size();
    }

    return {token_type, start_pos, end_pos};
  }

  int merge_size_ = 2;
  int32_t vision_start_token_id_;
  int32_t vision_end_token_id_;
  int32_t image_token_id_;
  int32_t video_token_id_;
  std::string vision_start_token_ = "<|vision_start|>";
  std::string vision_end_token_ = "<|vision_end|>";
  std::string image_token_ = "<|image_pad|>";
  std::string video_token_ = "<|video_pad|>";
};

struct Qwen3_5VLImageInputs {
  torch::Tensor pixel_values;
  torch::Tensor image_grid_thw;
};

struct Qwen3_5VLVideoInputs {
  torch::Tensor pixel_values_videos;
  torch::Tensor video_grid_thw;
  torch::Tensor second_per_grid_ts;
};

class Qwen3_5ForConditionalGenerationImpl : public torch::nn::Module {
 public:
  explicit Qwen3_5ForConditionalGenerationImpl(const ModelContext& context)
      : model_args_(context.get_model_args()),
        options_(context.get_tensor_options()) {
    visual_ = register_module("visual", Qwen3_5VisionTransformer(context));

    language_model_ =
        register_module("language_model", Qwen3_5ForCausalLM(context));
  }

  void prepare_encoder_input(
      const ModelInputParams& input_params,
      std::optional<Qwen3_5VLImageInputs>& image_inputs,
      std::optional<Qwen3_5VLVideoInputs>& video_inputs) {
    const auto& mm_data = input_params.mm_data;
    torch::Tensor pixel_values;
    if (const auto& res = mm_data.get<torch::Tensor>("pixel_values")) {
      pixel_values = res.value();
    }

    torch::Tensor image_grid_thw;
    if (const auto& res = mm_data.get<torch::Tensor>("image_grid_thw")) {
      image_grid_thw = res.value();
    }

    torch::Tensor pixel_values_videos;
    if (const auto& res = mm_data.get<torch::Tensor>("pixel_values_videos")) {
      pixel_values_videos = res.value();
    }

    torch::Tensor video_grid_thw;
    if (const auto& res = mm_data.get<torch::Tensor>("video_grid_thw")) {
      video_grid_thw = res.value();
    }

    if (pixel_values.defined() && image_grid_thw.defined()) {
      image_inputs = Qwen3_5VLImageInputs{pixel_values, image_grid_thw};
    }

    if (pixel_values_videos.defined() && video_grid_thw.defined()) {
      video_inputs = Qwen3_5VLVideoInputs{pixel_values_videos, video_grid_thw};
    }
  }

  MMDict get_multimodal_embeddings(const ModelInputParams& input_params) {
    std::optional<Qwen3_5VLImageInputs> image_input;
    std::optional<Qwen3_5VLVideoInputs> video_input;
    prepare_encoder_input(input_params, image_input, video_input);

    MMDict multimodal_embeds;
    const auto merge_size = model_args_.mm_image_merge_size();
    if (image_input) {
      auto [image_embeds, deep_stacks] =
          visual_(image_input->pixel_values.to(options_),
                  image_input->image_grid_thw.to(options_.device()),
                  input_params);

      auto image_tokens =
          (image_input->image_grid_thw.prod(-1) / merge_size / merge_size)
              .cpu()
              .contiguous()
              .to(torch::kLong);

      std::vector<int64_t> image_tokens_vec(
          image_tokens.data_ptr<int64_t>(),
          image_tokens.data_ptr<int64_t>() + image_tokens.numel());
      multimodal_embeds["image|embedding"] =
          image_embeds.split(image_tokens_vec, 0 /*dim*/);

      for (size_t i = 0; i < deep_stacks.size(); ++i) {
        multimodal_embeds[std::string("image|embedding|deepstack_") +
                          std::to_string(i)] =
            deep_stacks[i].split(image_tokens_vec, 0 /*dim*/);
      }
    }

    if (video_input) {
      auto [video_embeds, deep_stacks] =
          visual_(video_input->pixel_values_videos.to(options_),
                  video_input->video_grid_thw.to(options_.device()),
                  input_params);

      auto video_tokens =
          (video_input->video_grid_thw.prod(-1) / merge_size / merge_size)
              .cpu()
              .contiguous()
              .to(torch::kLong);

      std::vector<int64_t> video_tokens_vec(
          video_tokens.data_ptr<int64_t>(),
          video_tokens.data_ptr<int64_t>() + video_tokens.numel());
      multimodal_embeds["video|embedding"] =
          video_embeds.split(video_tokens_vec, 0 /*dim*/);

      for (size_t i = 0; i < deep_stacks.size(); ++i) {
        multimodal_embeds[std::string("video|embedding|deepstack_") +
                          std::to_string(i)] =
            deep_stacks[i].split(video_tokens_vec, 0 /*dim*/);
      }
    }

    return multimodal_embeds;
  }

  torch::Tensor generate_multimodal_mask(torch::Tensor input_ids) {
    auto special_token_ids = torch::tensor(
        {model_args_.image_token_id(), model_args_.video_token_id()},
        input_ids.options().dtype(torch::kInt64));
    return torch::isin(input_ids, special_token_ids);
  }

  std::vector<torch::Tensor> get_deep_stacks(
      const ModelInputParams& input_params) {
    const auto& mm_data = input_params.mm_data;
    if (!mm_data.has("embedding|deepstack_0")) {
      return {};
    }

    return {
        mm_data.get<torch::Tensor>("embedding|deepstack_0").value(),
        mm_data.get<torch::Tensor>("embedding|deepstack_1").value(),
        mm_data.get<torch::Tensor>("embedding|deepstack_2").value(),
    };
  }

  torch::Tensor merge_multimodal_embeddings(
      torch::Tensor inputs_embeds,
      const torch::Tensor& multimodal_embeds,
      const torch::Tensor& is_multimodal) {
    inputs_embeds.index_put_({is_multimodal}, multimodal_embeds);
    return inputs_embeds;
  }

  torch::Tensor get_input_embeddings(const torch::Tensor input_ids,
                                     const ModelInputParams& input_params) {
    const auto& mm_data = input_params.mm_data;
    torch::Tensor multimodal_embeds;
    if (const auto& emb = mm_data.get<torch::Tensor>("embedding")) {
      multimodal_embeds = emb.value();
    }

    auto inputs_embeds = language_model_->get_input_embeddings(input_ids);
    if (!multimodal_embeds.defined()) {
      return inputs_embeds;
    }

    auto is_multimodal = generate_multimodal_mask(input_ids);
    input_params.visual_pos_masks = is_multimodal;
    return merge_multimodal_embeddings(
        inputs_embeds, multimodal_embeds, is_multimodal);
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    input_params.deep_stacks = std::move(get_deep_stacks(input_params));
    return language_model_(tokens, positions, kv_caches, input_params);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    return language_model_->logits(hidden_states, seleted_idxes);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      visual_->load_state_dict(
          state_dict->get_dict_with_prefix("model.visual."));
    }
#ifdef USE_NPU
    visual_->verify_loaded_weights("model.visual.");
    visual_->merge_loaded_weights();
#endif

    if (!model_args_.image_embedding_mode()) {
      language_model_->load_model(std::move(loader));
    }
  }

  layer::LmHead get_lm_head() { return language_model_->get_lm_head(); }

  void set_lm_head(layer::LmHead& head) { language_model_->set_lm_head(head); }

  layer::WordEmbedding get_word_embedding() {
    return language_model_->get_word_embedding();
  }

  void set_word_embedding(layer::WordEmbedding& word_embedding) {
    language_model_->set_word_embedding(word_embedding);
  }

 private:
  ModelArgs model_args_;
  torch::TensorOptions options_;
  Qwen3_5VisionTransformer visual_{nullptr};
  Qwen3_5ForCausalLM language_model_{nullptr};
};
TORCH_MODULE(Qwen3_5ForConditionalGeneration);

#define LOAD_ARG_TEXT_OR_ROOT(arg_name, json_key, default_value) \
  LOAD_ARG_OR(arg_name, "text_config." json_key, default_value); \
  LOAD_ARG_OR(arg_name, json_key, args->arg_name())

#define LOAD_ARG_TEXT_OR_ROOT_CHAIN(arg_name, json_key, default_value) \
  LOAD_ARG_TEXT_OR_ROOT(arg_name, json_key, default_value)

#define LOAD_QWEN3_5_ROPE_ARG(arg_name, default_value)                       \
  LOAD_ARG_OR(arg_name, "text_config." #arg_name, default_value);            \
  LOAD_ARG_OR(arg_name, #arg_name, args->arg_name());                        \
  LOAD_ARG_OR(                                                               \
      arg_name, "text_config.rope_scaling." #arg_name, args->arg_name());    \
  LOAD_ARG_OR(arg_name, "rope_scaling." #arg_name, args->arg_name());        \
  LOAD_ARG_OR(                                                               \
      arg_name, "text_config.rope_parameters." #arg_name, args->arg_name()); \
  LOAD_ARG_OR(arg_name, "rope_parameters." #arg_name, args->arg_name())

#define LOAD_QWEN3_5_NEXT_COMPAT_ARGS(default_moe_intermediate_size,           \
                                      default_num_experts,                     \
                                      default_num_experts_per_tok,             \
                                      default_shared_expert_intermediate_size) \
  LOAD_ARG_TEXT_OR_ROOT(attention_bias, "attention_bias", false);              \
  LOAD_ARG_TEXT_OR_ROOT(attention_dropout, "attention_dropout", 0.0f);         \
  LOAD_ARG_TEXT_OR_ROOT(bos_token_id, "bos_token_id", 151643);                 \
  LOAD_ARG_TEXT_OR_ROOT(decoder_sparse_step, "decoder_sparse_step", 1);        \
  LOAD_ARG_TEXT_OR_ROOT(eos_token_id, "eos_token_id", 151645);                 \
  LOAD_ARG_TEXT_OR_ROOT(head_dim, "head_dim", 256);                            \
  LOAD_ARG_TEXT_OR_ROOT(hidden_act, "hidden_act", "silu");                     \
  LOAD_ARG_TEXT_OR_ROOT(hidden_size, "hidden_size", 2048);                     \
  LOAD_ARG_TEXT_OR_ROOT(initializer_range, "initializer_range", 0.02f);        \
  LOAD_ARG_TEXT_OR_ROOT(intermediate_size, "intermediate_size", 5120);         \
  LOAD_ARG_TEXT_OR_ROOT(                                                       \
      max_position_embeddings, "max_position_embeddings", 262144);             \
  LOAD_ARG_TEXT_OR_ROOT(max_window_layers, "max_window_layers", 28);           \
  LOAD_ARG_TEXT_OR_ROOT(moe_intermediate_size,                                 \
                        "moe_intermediate_size",                               \
                        default_moe_intermediate_size);                        \
  LOAD_ARG_TEXT_OR_ROOT(norm_topk_prob, "norm_topk_prob", true);               \
  LOAD_ARG_TEXT_OR_ROOT(n_heads, "num_attention_heads", 16);                   \
  LOAD_ARG_TEXT_OR_ROOT(num_experts, "num_experts", default_num_experts);      \
  LOAD_ARG_TEXT_OR_ROOT(num_experts_per_tok,                                   \
                        "num_experts_per_tok",                                 \
                        default_num_experts_per_tok);                          \
  LOAD_ARG_TEXT_OR_ROOT(n_layers, "num_hidden_layers", 48);                    \
  LOAD_ARG_OR(n_kv_heads, "text_config.num_key_value_heads", 2);               \
  LOAD_ARG_OR(                                                                 \
      n_kv_heads, "num_key_value_heads", args->n_kv_heads().value_or(2));      \
  LOAD_ARG_TEXT_OR_ROOT(output_router_logits, "output_router_logits", false);  \
  LOAD_ARG_TEXT_OR_ROOT(rms_norm_eps, "rms_norm_eps", 1e-6);                   \
  LOAD_QWEN3_5_ROPE_ARG(rope_theta, 10000000.0f);                              \
  LOAD_ARG(rope_scaling_mrope_section,                                         \
           "text_config.rope_scaling.mrope_section");                          \
  LOAD_ARG_OR(rope_scaling_mrope_section,                                      \
              "rope_scaling.mrope_section",                                    \
              args->rope_scaling_mrope_section());                             \
  LOAD_ARG_OR(rope_scaling_mrope_section,                                      \
              "text_config.rope_parameters.mrope_section",                     \
              args->rope_scaling_mrope_section());                             \
  LOAD_ARG_OR(rope_scaling_mrope_section,                                      \
              "rope_parameters.mrope_section",                                 \
              args->rope_scaling_mrope_section());                             \
  LOAD_ARG_TEXT_OR_ROOT(router_aux_loss_coef, "router_aux_loss_coef", 0.001f); \
  LOAD_ARG_TEXT_OR_ROOT(use_sliding_window, "use_sliding_window", false);      \
  LOAD_ARG_TEXT_OR_ROOT(sliding_window, "sliding_window", 4096);               \
  LOAD_ARG_TEXT_OR_ROOT(tie_word_embeddings, "tie_word_embeddings", false);    \
  LOAD_ARG_TEXT_OR_ROOT(vocab_size, "vocab_size", 151936);                     \
  LOAD_ARG_TEXT_OR_ROOT(                                                       \
      mlp_only_layers, "mlp_only_layers", std::vector<int32_t>());             \
  LOAD_ARG_TEXT_OR_ROOT(attn_output_gate, "attn_output_gate", true);           \
  LOAD_ARG_TEXT_OR_ROOT(                                                       \
      full_attention_interval, "full_attention_interval", 4);                  \
  LOAD_ARG_TEXT_OR_ROOT(linear_conv_kernel_dim, "linear_conv_kernel_dim", 4);  \
  LOAD_ARG_TEXT_OR_ROOT(linear_key_head_dim, "linear_key_head_dim", 128);      \
  LOAD_ARG_TEXT_OR_ROOT(linear_num_key_heads, "linear_num_key_heads", 16);     \
  LOAD_ARG_TEXT_OR_ROOT(linear_num_value_heads, "linear_num_value_heads", 32); \
  LOAD_ARG_TEXT_OR_ROOT(linear_value_head_dim, "linear_value_head_dim", 128);  \
  LOAD_QWEN3_5_ROPE_ARG(partial_rotary_factor, 0.25f);                         \
  LOAD_ARG_TEXT_OR_ROOT(shared_expert_intermediate_size,                       \
                        "shared_expert_intermediate_size",                     \
                        default_shared_expert_intermediate_size);              \
  LOAD_ARG_OR(                                                                 \
      num_nextn_predict_layers, "text_config.mtp_num_hidden_layers", 0);       \
  LOAD_ARG_OR(num_nextn_predict_layers,                                        \
              "mtp_num_hidden_layers",                                         \
              args->num_nextn_predict_layers());                               \
  LOAD_ARG_OR(num_nextn_predict_layers,                                        \
              "text_config.num_nextn_predict_layers",                          \
              args->num_nextn_predict_layers());                               \
  LOAD_ARG_OR(num_nextn_predict_layers,                                        \
              "num_nextn_predict_layers",                                      \
              args->num_nextn_predict_layers());                               \
  LOAD_ARG_OR(                                                                 \
      layer_types, "text_config.layer_types", std::vector<std::string>());     \
  LOAD_ARG_OR(layer_types, "layer_types", args->layer_types());                \
  LOAD_ARG_OR(                                                                 \
      layer_types, "text_config.layers_block_type", args->layer_types());      \
  LOAD_ARG_OR(layer_types, "layers_block_type", args->layer_types());          \
  LOAD_ARG_OR(                                                                 \
      n_routed_experts, "text_config.n_routed_experts", args->num_experts());  \
  LOAD_ARG_OR(n_routed_experts, "n_routed_experts", args->num_experts());      \
  SET_ARG(n_shared_experts,                                                    \
          args->shared_expert_intermediate_size() > 0 ? 1 : 0);                \
  SET_ARG(scoring_func, "softmax");                                            \
  SET_ARG(topk_method, "");                                                    \
  SET_ARG(n_group, -1);                                                        \
  SET_ARG(topk_group, 0);                                                      \
  SET_ARG(routed_scaling_factor, 1.0f);                                        \
  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}))

#define LOAD_QWEN3_5_TYPE_AND_DTYPE(default_model_type)         \
  LOAD_ARG_OR(model_type, "model_type", default_model_type);    \
  LOAD_ARG_OR(dtype, "text_config.dtype", "bfloat16");          \
  LOAD_ARG_OR(dtype, "dtype", args->dtype());                   \
  LOAD_ARG_OR(dtype, "text_config.torch_dtype", args->dtype()); \
  LOAD_ARG_OR(dtype, "torch_dtype", args->dtype())

#define LOAD_QWEN3_5_VISION_ARGS(default_mm_projection_dim)                    \
  LOAD_ARG_TEXT_OR_ROOT(                                                       \
      vision_start_token_id, "vision_start_token_id", 248053);                 \
  LOAD_ARG_TEXT_OR_ROOT(vision_end_token_id, "vision_end_token_id", 248054);   \
  LOAD_ARG_TEXT_OR_ROOT(vision_token_id, "vision_token_id", 248055);           \
  LOAD_ARG_TEXT_OR_ROOT(image_token_id, "image_token_id", 248056);             \
  LOAD_ARG_TEXT_OR_ROOT(video_token_id, "video_token_id", 248057);             \
  LOAD_ARG_OR(mm_num_hidden_layers, "vision_config.depth", 27);                \
  LOAD_ARG_OR(mm_hidden_act, "vision_config.hidden_act", "gelu_pytorch_tanh"); \
  LOAD_ARG_OR(mm_hidden_size, "vision_config.hidden_size", 1152);              \
  LOAD_ARG_OR(mm_intermediate_size, "vision_config.intermediate_size", 4304);  \
  LOAD_ARG_OR(mm_num_attention_heads, "vision_config.num_heads", 16);          \
  LOAD_ARG_OR(mm_num_channels, "vision_config.in_channels", 3);                \
  LOAD_ARG_OR(mm_projection_dim,                                               \
              "vision_config.out_hidden_size",                                 \
              default_mm_projection_dim);                                      \
  LOAD_ARG_OR(mm_patch_size, "vision_config.patch_size", 16);                  \
  LOAD_ARG_OR(mm_num_position_embeddings,                                      \
              "vision_config.num_position_embeddings",                         \
              2304);                                                           \
  LOAD_ARG_OR(mm_spatial_merge_size, "vision_config.spatial_merge_size", 2);   \
  LOAD_ARG(mm_deepstack_visual_indexes,                                        \
           "vision_config.deepstack_visual_indexes");                          \
  LOAD_ARG_OR(mm_temporal_patch_size, "vision_config.temporal_patch_size", 2); \
  LOAD_ARG_OR_FUNC(mm_head_dim, "vision_config.head_dim", [&] {                \
    return args->mm_hidden_size() / args->mm_num_attention_heads();            \
  });                                                                          \
  LOAD_ARG_OR(                                                                 \
      rope_scaling_rope_type, "vision_config.rope_scaling.type", "mrope")

REGISTER_CAUSAL_MODEL(qwen3_5, Qwen3_5ForCausalLM);
REGISTER_INPUT_PROCESSOR(qwen3_5, Qwen3_5VLInputProcessor);
REGISTER_CAUSAL_VLM_MODEL_WITH_VARNAME(qwen3_5_vlm,
                                       qwen3_5,
                                       Qwen3_5ForConditionalGeneration);
REGISTER_IMAGE_PROCESSOR(qwen3_5, Qwen2VLImageProcessor);
REGISTER_MODEL_ARGS(qwen3_5, [&] {
  LOAD_QWEN3_5_TYPE_AND_DTYPE("qwen3_5");
  LOAD_QWEN3_5_NEXT_COMPAT_ARGS(/*moe_intermediate_size=*/0,
                                /*num_experts=*/0,
                                /*num_experts_per_tok=*/0,
                                /*shared_expert_intermediate_size=*/0);
  LOAD_QWEN3_5_VISION_ARGS(/*default_mm_projection_dim=*/5120);
});

REGISTER_CAUSAL_MODEL(qwen3_5_text, Qwen3_5ForCausalLM);
REGISTER_MODEL_ARGS(qwen3_5_text, [&] {
  LOAD_QWEN3_5_TYPE_AND_DTYPE("qwen3_5_text");
  LOAD_QWEN3_5_NEXT_COMPAT_ARGS(/*moe_intermediate_size=*/0,
                                /*num_experts=*/0,
                                /*num_experts_per_tok=*/0,
                                /*shared_expert_intermediate_size=*/0);
});

REGISTER_CAUSAL_MODEL(qwen3_5_moe, Qwen3_5ForCausalLM);
REGISTER_INPUT_PROCESSOR(qwen3_5_moe, Qwen3_5VLInputProcessor);
REGISTER_CAUSAL_VLM_MODEL_WITH_VARNAME(qwen3_5_moe_vlm,
                                       qwen3_5_moe,
                                       Qwen3_5ForConditionalGeneration);
REGISTER_IMAGE_PROCESSOR(qwen3_5_moe, Qwen2VLImageProcessor);
REGISTER_MODEL_ARGS(qwen3_5_moe, [&] {
  LOAD_QWEN3_5_TYPE_AND_DTYPE("qwen3_5_moe");
  LOAD_QWEN3_5_NEXT_COMPAT_ARGS(/*moe_intermediate_size=*/512,
                                /*num_experts=*/512,
                                /*num_experts_per_tok=*/10,
                                /*shared_expert_intermediate_size=*/512);
  LOAD_QWEN3_5_VISION_ARGS(/*default_mm_projection_dim=*/2048);
});

REGISTER_CAUSAL_MODEL(qwen3_5_moe_text, Qwen3_5ForCausalLM);
REGISTER_MODEL_ARGS(qwen3_5_moe_text, [&] {
  LOAD_QWEN3_5_TYPE_AND_DTYPE("qwen3_5_moe_text");
  LOAD_QWEN3_5_NEXT_COMPAT_ARGS(/*moe_intermediate_size=*/512,
                                /*num_experts=*/512,
                                /*num_experts_per_tok=*/10,
                                /*shared_expert_intermediate_size=*/512);
});

#undef LOAD_QWEN3_5_TYPE_AND_DTYPE
#undef LOAD_QWEN3_5_VISION_ARGS
#undef LOAD_QWEN3_5_NEXT_COMPAT_ARGS
#undef LOAD_QWEN3_5_ROPE_ARG
#undef LOAD_ARG_TEXT_OR_ROOT_CHAIN
#undef LOAD_ARG_TEXT_OR_ROOT

}  // namespace xllm
