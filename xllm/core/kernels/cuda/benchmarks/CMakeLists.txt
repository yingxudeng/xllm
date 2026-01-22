include(cc_binary)

include_directories(${CMAKE_CURRENT_LIST_DIR})

# Common link options for brpc static library dependency resolution.
# :util (dep of :cuda_kernels) depends on brpc, which depends on OpenSSL and leveldb.
# When using static libraries, link order matters. Use LINK_GROUP with RESCAN
# to let the linker rescan the group until all symbols are resolved.
set(BRPC_LINK_GROUP "$<LINK_GROUP:RESCAN,brpc,leveldb::leveldb,OpenSSL::SSL,OpenSSL::Crypto>")

# Common dependencies for cuda kernel benchmarks
set(CUDA_KERNEL_BENCHMARK_DEPS
    :cuda_kernels
    torch
    glog::glog
)

# cutlass_scaled_mm_sm120_benchmark - SM120 (Blackwell) FP8 GEMM grid search benchmark
# This benchmark is used to find optimal tile configurations for different M, N, K shapes.
# Usage: ./cutlass_scaled_mm_sm120_benchmark [--warmup N] [--iters N] [--quick]
cc_binary(
  NAME
    cutlass_scaled_mm_sm120_benchmark
  SRCS
    cutlass_scaled_mm_sm120_benchmark.cpp
  DEPS
    ${CUDA_KERNEL_BENCHMARK_DEPS}
  LINKOPTS
    ${BRPC_LINK_GROUP}
)
add_dependencies(cutlass_scaled_mm_sm120_benchmark brpc-static)
# Add benchmark to all_tests target so it gets rebuilt when building all_tests
add_dependencies(all_tests cutlass_scaled_mm_sm120_benchmark)

# cutlass_scaled_mm_sm120_config_analysis - SM120 FP8 GEMM config performance analysis
# This tool analyzes the performance of different CUTLASS configurations for SM120.
# Uses gtest framework for organizing tests, but NOT registered with ctest.
# Usage: ./cutlass_scaled_mm_sm120_config_analysis [gtest options]
cc_binary(
  NAME
    cutlass_scaled_mm_sm120_config_analysis
  SRCS
    cutlass_scaled_mm_sm120_config_analysis.cpp
  DEPS
    ${CUDA_KERNEL_BENCHMARK_DEPS}
    GTest::gtest_main
  LINKOPTS
    ${BRPC_LINK_GROUP}
)
add_dependencies(cutlass_scaled_mm_sm120_config_analysis brpc-static)
# Add benchmark to all_tests target so it gets rebuilt when building all_tests
add_dependencies(all_tests cutlass_scaled_mm_sm120_config_analysis)
