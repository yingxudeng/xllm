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

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/torch_npu.h>

#include <chrono>
#include <string>
#include <utility>
#include <vector>

#include "acl/acl.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

namespace xllm::kernel::npu::tilelang {
namespace {

class TileLangFusedGdnGatingWrapperTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { torch_npu::init_npu("npu:0"); }

  static void TearDownTestSuite() { torch_npu::finalize_npu(); }
};

struct FusedGdnGatingTestCase {
  std::string name;
  int64_t num_batches;
  int64_t num_heads;
  int64_t seed;
  float beta = 1.0F;
  float threshold = 20.0F;
};

std::pair<torch::Tensor, torch::Tensor> torch_fused_gdn_gating(
    const torch::Tensor& A_log,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& dt_bias,
    float beta = 1.0F,
    float threshold = 20.0F) {
  namespace F = torch::nn::functional;

  auto softplus_out =
      F::softplus(a.to(torch::kFloat32) + dt_bias,
                  F::SoftplusFuncOptions().beta(beta).threshold(threshold));
  auto g = -A_log.exp() * softplus_out;
  auto beta_output = torch::sigmoid(b.to(torch::kFloat32)).to(torch::kBFloat16);
  return {g.unsqueeze(0), beta_output.unsqueeze(0)};
}

void run_fused_gdn_gating_case(const FusedGdnGatingTestCase& test_case) {
  ASSERT_GT(test_case.num_batches, 0);

  const auto device = torch::Device("npu:0");
  torch::manual_seed(test_case.seed);

  auto fp32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto bf16_opts =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);

  auto A_log = torch::randn({test_case.num_heads}, fp32_opts);
  auto a =
      torch::randn({test_case.num_batches, test_case.num_heads}, bf16_opts);
  auto b =
      torch::randn({test_case.num_batches, test_case.num_heads}, bf16_opts);
  auto dt_bias = torch::randn({test_case.num_heads}, fp32_opts);

  auto [g_ref, beta_ref] = torch_fused_gdn_gating(
      A_log, a, b, dt_bias, test_case.beta, test_case.threshold);
  auto [g_out, beta_out] = fused_gdn_gating(
      A_log, a, b, dt_bias, test_case.beta, test_case.threshold);

  auto g_max_diff = (g_out - g_ref).abs().max().item<float>();
  auto beta_max_diff =
      (beta_out.to(torch::kFloat32) - beta_ref.to(torch::kFloat32))
          .abs()
          .max()
          .item<float>();

  EXPECT_TRUE(torch::allclose(g_out, g_ref, 1e-3, 1e-3))
      << "g mismatch, max_diff=" << g_max_diff;
  EXPECT_TRUE(torch::allclose(beta_out, beta_ref, 1e-2, 1e-2))
      << "beta mismatch, max_diff=" << beta_max_diff;
}

TEST_F(TileLangFusedGdnGatingWrapperTest, MatchesTorchReference) {
  const std::vector<FusedGdnGatingTestCase> cases = {
      {
          .name = "tiny_b1_h8",
          .num_batches = 1,
          .num_heads = 8,
          .seed = 101,
      },
      {
          .name = "tiny_b17_h8",
          .num_batches = 17,
          .num_heads = 8,
          .seed = 101,
      },
      {
          .name = "tiny_b1_h16",
          .num_batches = 1,
          .num_heads = 16,
          .seed = 101,
      },
      {
          .name = "small_b17_h32",
          .num_batches = 17,
          .num_heads = 32,
          .seed = 102,
      },
      {
          .name = "medium_b29_h48",
          .num_batches = 29,
          .num_heads = 48,
          .seed = 103,
      },
      {
          .name = "medium_b131_h64",
          .num_batches = 131,
          .num_heads = 64,
          .seed = 104,
      },
      {
          .name = "medium_b257_h128",
          .num_batches = 257,
          .num_heads = 128,
          .seed = 105,
      },
      {
          .name = "large_b4096_h32",
          .num_batches = 4096,
          .num_heads = 32,
          .seed = 106,
      },
      {
          .name = "chunked_b8192_h32",
          .num_batches = 8192,
          .num_heads = 32,
          .seed = 108,
      },
      {
          .name = "custom_beta2_threshold0p5_b33_h64",
          .num_batches = 33,
          .num_heads = 64,
          .seed = 107,
          .beta = 2.0F,
          .threshold = 0.5F,
      },
      {
          .name = "large_b65536_h32",
          .num_batches = 65536,
          .num_heads = 32,
          .seed = 109,
      },
      {
          .name = "large_b262144_h32",
          .num_batches = 262144,
          .num_heads = 32,
          .seed = 110,
      },
  };

  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    run_fused_gdn_gating_case(test_case);
  }
}

struct BenchmarkShape {
  int64_t num_batches;
  int64_t num_heads;
};

TEST_F(TileLangFusedGdnGatingWrapperTest, Benchmark) {
  const std::vector<BenchmarkShape> shapes = {
      {16, 32},
      {48, 32},
      {1024, 32},
      {4096, 32},
      {16384, 32},
      {65536, 32},
      {262144, 32},
      {4096, 128},
      {65536, 128},
      {262144, 128},
  };
  constexpr int kWarmup = 10;
  constexpr int kIters = 100;

  for (const auto& shape : shapes) {
    const auto device = torch::Device("npu:0");
    auto fp32_opts =
        torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto bf16_opts =
        torch::TensorOptions().dtype(torch::kBFloat16).device(device);

    auto A_log = torch::randn({shape.num_heads}, fp32_opts);
    auto a = torch::randn({shape.num_batches, shape.num_heads}, bf16_opts);
    auto b = torch::randn({shape.num_batches, shape.num_heads}, bf16_opts);
    auto dt_bias = torch::randn({shape.num_heads}, fp32_opts);

    for (int i = 0; i < kWarmup; ++i) {
      fused_gdn_gating(A_log, a, b, dt_bias, 1.0F, 20.0F);
    }
    aclrtSynchronizeStream(c10_npu::getCurrentNPUStream(0).stream());

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kIters; ++i) {
      fused_gdn_gating(A_log, a, b, dt_bias, 1.0F, 20.0F);
    }
    aclrtSynchronizeStream(c10_npu::getCurrentNPUStream(0).stream());
    auto t1 = std::chrono::high_resolution_clock::now();

    double fused_us =
        std::chrono::duration<double, std::micro>(t1 - t0).count() / kIters;

    // Naive PyTorch baseline.
    auto naive_fn = [&]() {
      auto x = a.to(torch::kFloat32) + dt_bias;
      auto g = -A_log.exp() * at::softplus(x, /*beta=*/1.0, /*threshold=*/20.0);
      auto beta_val =
          torch::sigmoid(b.to(torch::kFloat32)).to(torch::kBFloat16);
      return std::make_pair(g, beta_val);
    };

    for (int i = 0; i < kWarmup; ++i) {
      naive_fn();
    }
    aclrtSynchronizeStream(c10_npu::getCurrentNPUStream(0).stream());

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kIters; ++i) {
      naive_fn();
    }
    aclrtSynchronizeStream(c10_npu::getCurrentNPUStream(0).stream());
    t1 = std::chrono::high_resolution_clock::now();

    double naive_us =
        std::chrono::duration<double, std::micro>(t1 - t0).count() / kIters;

    double speedup = naive_us / fused_us;

    printf("B=%7ld H=%3ld: fused=%7.1fus naive=%7.1fus speedup=%.2fx\n",
           shape.num_batches,
           shape.num_heads,
           fused_us,
           naive_us,
           speedup);
  }
}

}  // namespace
}  // namespace xllm::kernel::npu::tilelang
