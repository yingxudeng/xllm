include(cc_library)
include(cc_test)

include(FetchContent)
set(CUTLASS_REVISION "v4.2.1")
FetchContent_Declare(
    cutlass
    GIT_REPOSITORY https://github.com/nvidia/cutlass.git
    GIT_TAG ${CUTLASS_REVISION}
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(cutlass)

include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CUTLASS_INCLUDE_DIR})
include_directories(${CUTLASS_TOOLS_UTIL_INCLUDE_DIR})

# Definitions from vllm
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0)
    add_definitions(-DENABLE_SCALED_MM_SM90=1)
    add_definitions(-DENABLE_CUTLASS_MOE_SM90=1)
endif()

# SM80/SM89 support
add_definitions(-DENABLE_SCALED_MM_C2X=1)

# FP8 quantization support (requires SM 8.9+ for hardware FP8)
# Enable for H100/H800 and newer GPUs
add_definitions(-DENABLE_FP8=1)

file(GLOB_RECURSE CUDA_HEADER_FILES
  "${CMAKE_CURRENT_LIST_DIR}/*.h"
  "${CMAKE_CURRENT_LIST_DIR}/*.cuh"
)

file(GLOB_RECURSE CUDA_SOURCE_FILES
  "${CMAKE_CURRENT_LIST_DIR}/*.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/*.cu"
)

add_subdirectory(triton)
# add_subdirectory(tests)
cc_test(
  NAME
    decoder_reshape_and_cache_test
  SRCS
    decoder_reshape_and_cache_test.cpp
  DEPS
    :cuda_kernels
    torch
    GTest::gtest_main
    glog::glog
)
target_link_libraries(decoder_reshape_and_cache_test PRIVATE brpc)

# beam_search_test
cc_test(
  NAME
    beam_search_test
  SRCS
    beam_search_test.cpp
  DEPS
    :cuda_kernels
    torch
    GTest::gtest_main
    glog::glog
)
target_link_libraries(beam_search_test PRIVATE brpc)

# cutlass_scaled_mm_test
cc_test(
  NAME
    cutlass_scaled_mm_test
  SRCS
    cutlass_scaled_mm_test.cpp
  DEPS
    :cuda_kernels
    torch
    GTest::gtest_main
    glog::glog
)
target_link_libraries(cutlass_scaled_mm_test PRIVATE brpc)

# static_scaled_fp8_quant_test
cc_test(
  NAME
    static_scaled_fp8_quant_test
  SRCS
    static_scaled_fp8_quant_test.cpp
  DEPS
    :cuda_kernels
    torch
    GTest::gtest_main
    glog::glog
)
target_link_libraries(static_scaled_fp8_quant_test PRIVATE brpc)

# FP8 quantization example (optional, enable with -DBUILD_FP8_EXAMPLES=ON)
if(BUILD_FP8_EXAMPLES)
  add_executable(fp8_quant_example
    fp8_quant_example.cpp
    fp8_quant.cu
  )
  target_compile_definitions(fp8_quant_example PRIVATE 
    FP8_QUANT_STANDALONE_TEST
    ENABLE_FP8
  )
  target_link_libraries(fp8_quant_example
    torch
    c10
    c10_cuda
    glog::glog
  )
  message(STATUS "FP8 quantization example will be built")
endif()

cc_library(
  NAME
    cuda_kernels
  HDRS
    ${CUDA_HEADER_FILES}
  SRCS
    ${CUDA_SOURCE_FILES}
  DEPS
    tvm_ffi
    torch
    :util
    :platform
    :rec_triton
)
