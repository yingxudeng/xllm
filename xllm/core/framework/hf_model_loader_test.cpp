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

#include "hf_model_loader.h"

#include <gtest/gtest.h>

#include "core/platform/device.h"
#if defined(USE_NPU)
#include "models/model_registry.h"
#endif

namespace xllm {

TEST(HFModelLoaderTest, LoadCompressedTensorsFp8StaticConfig) {
  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "quantization_config": {
        "config_groups": {
          "group_0": {
            "input_activations": {
              "dynamic": false,
              "num_bits": 8,
              "type": "float"
            },
            "weights": {
              "num_bits": 8,
              "type": "float"
            }
          }
        },
        "quant_method": "compressed-tensors"
      }
    }
  )json"));

  QuantArgs quant_args;
  if (Device::type_str() == "cuda") {
    ASSERT_TRUE(load_quant_cfg(reader, quant_args));
    EXPECT_EQ(quant_args.quant_method(), kQuantMethodFp8);
    EXPECT_EQ(quant_args.bits(), 8);
    EXPECT_EQ(quant_args.moe_weight_bits(), 8);
    EXPECT_FALSE(quant_args.activation_dynamic());
  }
}

TEST(HFModelLoaderTest, KeepLegacyFp8ConfigUnchanged) {
  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "quantization_config": {
        "activation_scheme": "static",
        "quant_method": "fp8"
      }
    }
  )json"));

  QuantArgs quant_args;
  ASSERT_TRUE(load_quant_cfg(reader, quant_args));
  EXPECT_EQ(quant_args.quant_method(), kQuantMethodFp8);
  EXPECT_FALSE(quant_args.activation_dynamic());
}

#if defined(USE_NPU)
TEST(HFModelLoaderTest, Qwen35MtpModelArgsFromDenseConfig) {
  auto loader = ModelRegistry::get_model_args_loader("qwen3_5_mtp");
  ASSERT_TRUE(loader != nullptr);

  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "model_type": "qwen3_5",
      "text_config": {
        "mtp_num_hidden_layers": 1,
        "layer_types": ["linear_attention"]
      }
    }
  )json"));

  ModelArgs args;
  ASSERT_TRUE(loader(reader, &args));
  EXPECT_EQ(args.model_type(), "qwen3_5_mtp");
  EXPECT_EQ(args.num_nextn_predict_layers(), 1);
  EXPECT_EQ(args.n_layers(), 1);
  ASSERT_EQ(args.layer_types().size(), 1);
  EXPECT_EQ(args.layer_types()[0], "full_attention");
}

TEST(HFModelLoaderTest, Qwen35MtpModelArgsFromMoeConfig) {
  auto loader = ModelRegistry::get_model_args_loader("qwen3_5_moe_mtp");
  ASSERT_TRUE(loader != nullptr);

  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "model_type": "qwen3_5_moe",
      "text_config": {
        "mtp_num_hidden_layers": 2,
        "layer_types": ["linear_attention", "linear_attention"]
      }
    }
  )json"));

  ModelArgs args;
  ASSERT_TRUE(loader(reader, &args));
  EXPECT_EQ(args.model_type(), "qwen3_5_moe_mtp");
  EXPECT_EQ(args.num_nextn_predict_layers(), 2);
  EXPECT_EQ(args.n_layers(), 2);
  ASSERT_EQ(args.layer_types().size(), 2);
  EXPECT_EQ(args.layer_types()[0], "full_attention");
  EXPECT_EQ(args.layer_types()[1], "full_attention");
}

TEST(HFModelLoaderTest, Qwen35DenseMultimodalModelArgsAndBackend) {
  auto loader = ModelRegistry::get_model_args_loader("qwen3_5");
  ASSERT_TRUE(loader != nullptr);

  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "model_type": "qwen3_5",
      "vision_start_token_id": 248053,
      "vision_end_token_id": 248054,
      "image_token_id": 248056,
      "video_token_id": 248057,
      "text_config": {
        "model_type": "qwen3_5_text",
        "hidden_size": 5120,
        "num_attention_heads": 24,
        "num_hidden_layers": 64,
        "layer_types": ["linear_attention", "full_attention"]
      },
      "vision_config": {
        "depth": 27,
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "num_heads": 16,
        "in_channels": 3,
        "out_hidden_size": 5120,
        "patch_size": 16,
        "num_position_embeddings": 2304,
        "spatial_merge_size": 2,
        "deepstack_visual_indexes": [],
        "temporal_patch_size": 2
      }
    }
  )json"));

  ModelArgs args;
  ASSERT_TRUE(loader(reader, &args));
  EXPECT_EQ(args.model_type(), "qwen3_5");
  EXPECT_EQ(ModelRegistry::get_model_backend("qwen3_5"), "vlm");
  EXPECT_EQ(args.vision_start_token_id(), 248053);
  EXPECT_EQ(args.vision_end_token_id(), 248054);
  EXPECT_EQ(args.image_token_id(), 248056);
  EXPECT_EQ(args.video_token_id(), 248057);
  EXPECT_EQ(args.mm_num_hidden_layers(), 27);
  EXPECT_EQ(args.mm_hidden_size(), 1152);
  EXPECT_EQ(args.mm_projection_dim(), 5120);
  EXPECT_EQ(args.mm_num_attention_heads(), 16);
}

TEST(HFModelLoaderTest, Qwen35MoeMultimodalModelArgsAndBackend) {
  auto loader = ModelRegistry::get_model_args_loader("qwen3_5_moe");
  ASSERT_TRUE(loader != nullptr);

  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "model_type": "qwen3_5_moe",
      "vision_start_token_id": 248053,
      "vision_end_token_id": 248054,
      "image_token_id": 248056,
      "video_token_id": 248057,
      "text_config": {
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "num_hidden_layers": 40,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "shared_expert_intermediate_size": 512,
        "layer_types": ["linear_attention", "full_attention"]
      },
      "vision_config": {
        "depth": 27,
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "num_heads": 16,
        "in_channels": 3,
        "out_hidden_size": 2048,
        "patch_size": 16,
        "num_position_embeddings": 2304,
        "spatial_merge_size": 2,
        "deepstack_visual_indexes": [],
        "temporal_patch_size": 2
      }
    }
  )json"));

  ModelArgs args;
  ASSERT_TRUE(loader(reader, &args));
  EXPECT_EQ(args.model_type(), "qwen3_5_moe");
  EXPECT_EQ(ModelRegistry::get_model_backend("qwen3_5_moe"), "vlm");
  EXPECT_EQ(args.image_token_id(), 248056);
  EXPECT_EQ(args.video_token_id(), 248057);
  EXPECT_EQ(args.mm_hidden_size(), 1152);
  EXPECT_EQ(args.mm_projection_dim(), 2048);
  EXPECT_EQ(args.num_experts(), 256);
  EXPECT_EQ(args.num_experts_per_tok(), 8);
}

#endif

}  // namespace xllm
