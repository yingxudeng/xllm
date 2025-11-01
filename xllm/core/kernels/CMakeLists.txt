include(cc_library)

if(USE_NPU)
  include_directories(
    ${CMAKE_SOURCE_DIR}/third_party/spdlog/include
  )
  add_subdirectory(npu)
  add_subdirectory(npu_v1)
endif()

if(USE_MLU)
  add_subdirectory(mlu)
endif()
