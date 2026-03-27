include(cc_library)
include(cc_test)

cc_library(
  NAME
    npu_torch_layers
  HDRS
    fused_moe.h
    attention.h
    qwen3_gated_delta_net_base.h
    qwen3_next_attention.h
    qwen3_next_gated_delta_net.h
    qwen3_5_gated_delta_net.h
    qwen3_next_hybrid_decoder_layer_base.h
    qwen3_next_decoder_layer_impl.h
    qwen3_5_decoder_layer_impl.h
  SRCS
    fused_moe.cpp
    attention.cpp
    qwen3_gated_delta_net_base.cpp
    qwen3_next_attention.cpp
    qwen3_next_gated_delta_net.cpp
    qwen3_next_hybrid_decoder_layer_base.cpp
    qwen3_5_gated_delta_net.cpp
    qwen3_next_decoder_layer_impl.cpp
    qwen3_5_decoder_layer_impl.cpp
  DEPS
    :common_layers
)

cc_test(
  NAME
    qwen3_gated_delta_net_base_test
  SRCS
    qwen3_gated_delta_net_base_test.cpp
  DEPS
    :npu_torch_layers
    GTest::gtest_main
)

target_link_libraries(qwen3_gated_delta_net_base_test PRIVATE atb)
