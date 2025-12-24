include(cc_library)
include(cc_test)

# find all *.h/*.c files in folder aot recursively
# file(GLOB_RECURSE AOT_HEADERS ptx_kernels/*.h)
# file(GLOB_RECURSE AOT_SOURCES aot/*.c)

# cc_library(
#   NAME 
#     rec_triton_kernel
#   HDRS
#     ${AOT_HEADERS}
#   SRCS 
#   DEPS
#     glog::glog
# )

cc_library(
  NAME
    rec_triton
  HDRS
    rec/rec.h
    # rec/rec_triton.h
    rec/rec_torch.h
  SRCS
    # rec/rec_triton.cpp
    rec/rec_torch.cpp
  DEPS
    # rec_triton_kernel
    torch
    cuda
)
