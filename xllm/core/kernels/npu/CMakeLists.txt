include(cc_library)

add_subdirectory(xllm_ops)

file(GLOB_RECURSE NPU_HEADER_FILES
  "${CMAKE_CURRENT_LIST_DIR}/*.h"
)

file(GLOB_RECURSE NPU_SOURCE_FILES
  "${CMAKE_CURRENT_LIST_DIR}/*.cpp"
)

cc_library(
  NAME
    npu_kernels
  HDRS
    ${NPU_HEADER_FILES}
  SRCS
    ${NPU_SOURCE_FILES}
  DEPS
    :torch_npu_kernels
)
