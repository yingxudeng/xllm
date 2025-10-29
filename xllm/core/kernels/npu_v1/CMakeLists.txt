include(cc_library)
include(cc_test)


file(GLOB_RECURSE OPPLUGIN_UTILS_HEADER
  "${CMAKE_CURRENT_LIST_DIR}/custom_functions_npu/*.h"
  "${CMAKE_CURRENT_LIST_DIR}/ops_npu/*.h"
)

file(GLOB_RECURSE OPPLUGIN_UTILS_SRCS
  "${CMAKE_CURRENT_LIST_DIR}/custom_functions_npu/*.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/ops_npu/*.cpp"
)

cc_library(
  NAME
    xllm_npu_ops
  HDRS
    ${OPPLUGIN_UTILS_HEADER}
  SRCS
    ${OPPLUGIN_UTILS_SRCS}
  DEPS
    # :npu_layers
    :model_context
    # :state_dict
    glog::glog
    torch
    torch_npu
)
