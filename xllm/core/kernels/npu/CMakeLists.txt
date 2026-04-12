include(cc_library)
include(cc_test)

add_subdirectory(tilelang)

add_subdirectory(xllm_ops)

cc_library(
  NAME
    npu_kernels
  HDRS
    npu_ops_api.h
  SRCS
    active.cpp
    attention.cpp
    fused_layernorm.cpp
    gated_delta_net.cpp
    matmul.cpp
    npu_grouped_matmul.cpp
    npu_moe_gating_topk_softmax.cpp
    npu_moe_init_routing_v2.cpp
    npu_moe_token_unpermute.cpp
    rope.cpp
  DEPS
    :triton_adapter
    :torch_npu_kernels
    :tilelang_kernels
)

cc_test(
  NAME
    npu_kernels_test
  SRCS
    causal_conv1d_test.cpp
    chunk_gated_delta_rule_test.cpp
  DEPS
    :npu_kernels
    :xllm_ops
    nnopbase
    GTest::gtest_main
)
