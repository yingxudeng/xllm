#pragma once

#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <fstream>
#include <iostream>

namespace xllm::kernel::cuda::triton {

#define PROCESS_KERNEL(ARCH,                                                   \
                       DTYPE,                                                  \
                       CONFIG_TYPE,                                            \
                       CONFIG_QUEUE,                                           \
                       DIM_ARRAY,                                              \
                       KERNEL_NAME,                                            \
                       BLOCK_SPEC)                                             \
  do {                                                                         \
    std::size_t dim_range_size =                                               \
        KERNEL_NAME##_##ARCH##_##DTYPE##_INPUT_DIM_ARRAY.size();               \
                                                                               \
    if (dim_range_size != KERNEL_NAME##_##ARCH##_##DTYPE##_PTX_ARRAY.size()) { \
      LOG(FATAL) << "dim_range_size is wrong ";                                \
    }                                                                          \
    if (dim_range_size !=                                                      \
        KERNEL_NAME##_##ARCH##_##DTYPE##_SHARED_MEM_BYTES.size()) {            \
      LOG(FATAL) << "dim_range_size is wrong ";                                \
    }                                                                          \
    for (size_t i = 0;                                                         \
         i < KERNEL_NAME##_##ARCH##_##DTYPE##_INPUT_DIM_ARRAY.size();          \
         ++i) {                                                                \
      CONFIG_TYPE config;                                                      \
      LOG(INFO) << "KERNEL_NAME##_##ARCH##_##DTYPE: " << KERNEL_NAME##_##ARCH##_##DTYPE; \
      config.loader = std::make_unique<PtxKernelLoader>(                       \
          KERNEL_NAME##_##ARCH##_##DTYPE##_PTX_ARRAY.at(i),                    \
          KERNEL_NAME##_##ARCH##_##DTYPE);                                     \
      config.kernel = config.loader->get_kernel();                             \
      BLOCK_SPEC                                                               \
      config.shared_mem_bytes =                                                \
          KERNEL_NAME##_##ARCH##_##DTYPE##_SHARED_MEM_BYTES.at(i);             \
      if (config.shared_mem_bytes > 49152) {                                   \
        CUDA_CHECK(cuFuncSetAttribute(                                         \
            config.kernel,                                                     \
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,                   \
            config.shared_mem_bytes));                                         \
      }                                                                        \
      CONFIG_QUEUE.push(std::move(config));                                    \
      DIM_ARRAY.emplace_back(                                                  \
          KERNEL_NAME##_##ARCH##_##DTYPE##_INPUT_DIM_ARRAY.at(i).begin(),      \
          KERNEL_NAME##_##ARCH##_##DTYPE##_INPUT_DIM_ARRAY.at(i).end());       \
    }                                                                          \
  } while (0);

#define BLOCK_SINGLE(VAR) \
  config.block_size.x = VAR##_##BLOCK_SIZE.at(i);

#define BLOCK_TRIPLE(VAR)                                       \
  config.block_size.x = VAR##_##BLOCK_M_ARRAY.at(i); \
  config.block_size.y = VAR##_##BLOCK_N_ARRAY.at(i); \
  config.block_size.z = VAR##_##BLOCK_K_ARRAY.at(i);

static uint32_t get_default_thread_nums() {
  constexpr uint32_t nums_warp = 4;
  int warp_size;
  cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, 0);
  uint32_t thread_nums = warp_size * nums_warp;
  return thread_nums;
}

template <typename T>
static int cuda_check_impl(T result) {
  if constexpr (std::is_same_v<T, CUresult>) {
    // handle CUresult type
    if (result == CUDA_SUCCESS) {
      return 0;
    }
    cudaError_t err = cudaGetLastError();
// LOG(INFO) << "Kernel launch failed: " << cudaGetErrorString(err);
    const char* errorStr = nullptr;
    if (cuGetErrorName(result, &errorStr) == CUDA_SUCCESS) {
      LOG(FATAL) << "CUDA error (CUresult): " << errorStr << " (code " << result
                 << ")";
    } else {
      LOG(FATAL) << "CUDA error (CUresult): Unknown error (code " << result
                 << ")";
    }
  } else if constexpr (std::is_same_v<T, cudaError_t>) {
    // handle cudaError_t type
// LOG(INFO) << "cudaSuccess: " << cudaSuccess;
// LOG(INFO) << "result: " << result;
    // if (result != cudaSuccess) {
    //   LOG(FATAL) << "CUDA error (cudaError_t): " << result << " (code "
    //              << result << ")";
    // }
  }
}

#define CUDA_CHECK(result)   \
  do {                       \
    cuda_check_impl(result); \
  } while (0)

template <typename FuncT>
void print_kernel_performance(const FuncT& kernel_func,
                              const std::size_t execute_num) {
  // auto start = std::chrono::high_resolution_clock::now();
  for (std::size_t i = 0; i < execute_num; ++i) {
    kernel_func();
  }
  // auto end = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double> duration = end - start;
}

class PtxKernelLoader {
 public:
  PtxKernelLoader(const char* ptx_content, const char* kernel_name) {
    char temp_filename[] = "temp_ptx_XXXXXX";

    int fd = mkstemp(temp_filename);
    if (fd == -1) {
      LOG(FATAL) << "failed to create temp ptx file.";
    }
    std::ofstream temp_file(temp_filename);
    if (!temp_file.is_open()) {
      close(fd);
      LOG(FATAL) << "failed to open temp ptx file: " << temp_filename;
    }
    temp_file << ptx_content;
    temp_file.close();
    close(fd);
// LOG(INFO) << "before cuInit(0).";
    CUDA_CHECK(cuInit(0));
// LOG(INFO) << "before cuModuleLoad(0).";
    CUDA_CHECK(cuModuleLoad(&module, temp_filename));

    // delete temp file
    std::remove(temp_filename);
// LOG(INFO) << "kernel_name_: " << kernel_name_;
    kernel_name_ = kernel_name;
  }

  CUfunction get_kernel() {
    CUfunction kernel;
// LOG(INFO) << "before cuModuleGetFunction.";
    CUDA_CHECK(cuModuleGetFunction(&kernel, module, kernel_name_));
// LOG(INFO) << "after cuModuleGetFunction.";
    return kernel;
  }

  ~PtxKernelLoader() { CUDA_CHECK(cuModuleUnload(module)); }

 private:
  CUmodule module;
  const char* kernel_name_;
};

enum class ARCH { UN_SUPPORTED, SM_86, SM_89, SM_90 };

static ARCH get_cuda_arch() {
  cudaDeviceProp props;
  (cudaGetDeviceProperties(&props, 0));
  if (props.major == 8 && props.minor == 6) {
    return ARCH::SM_86;
  } else if (props.major == 8 && props.minor == 9) {
    return ARCH::SM_89;
  } else if (props.major == 9 && props.minor == 0) {
    return ARCH::SM_90;
  }
  return ARCH::UN_SUPPORTED;
}

template <typename KernelConfigT>
struct GenericKernelConfigs {
 public:
  void push(KernelConfigT&& kernel_config) {
    kernel_configs_.emplace_back(std::forward<KernelConfigT>(kernel_config));
  }

  template <size_t M, size_t N, typename T>
  const KernelConfigT& find_closest_index(
      const std::array<std::array<T, N>, M>& input_dim_array,
      const std::array<T, N>& input_dim) {
    static_assert(M > 0, "Configuration array must not be empty");
    size_t closest_index = 0;
    T min_distance = std::numeric_limits<T>::max();
    for (size_t i = 0; i < input_dim_array.size(); ++i) {
      const T current_distance =
          squared_distance(input_dim_array[i], input_dim);
      if (current_distance < min_distance) {
        min_distance = current_distance;
        closest_index = i;
      }
    }
// LOG(INFO) << "closest_index: " << closest_index;
    return kernel_configs(closest_index);
  }

  template <size_t N, typename T>
  const KernelConfigT& find_closest_index(
      const std::vector<std::vector<T>>& input_dim_array,
      const std::array<T, N>& input_dim) {
    size_t closest_index = 0;
    T min_distance = std::numeric_limits<T>::max();
    for (size_t i = 0; i < input_dim_array.size(); ++i) {
      const T current_distance =
          squared_distance(input_dim_array[i], input_dim);
      if (current_distance < min_distance) {
        min_distance = current_distance;
        closest_index = i;
      }
    }
// LOG(INFO) << "closest_index: " << closest_index;
    return kernel_configs(closest_index);
  }

 private:
  template <typename T, size_t N>
  constexpr T squared_distance(const std::array<T, N>& a,
                               const std::array<T, N>& b) {
    T sum = 0;
    for (size_t i = 0; i < N; ++i) {
      sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sum;
  }

  template <typename T, size_t N>
  constexpr T squared_distance(const std::vector<T>& a,
                               const std::array<T, N>& b) {
    T sum = 0;
    for (size_t i = 0; i < N; ++i) {
      sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sum;
  }

  const KernelConfigT& kernel_configs(const std::size_t index) const {
    return kernel_configs_.at(index);
  }

  std::vector<KernelConfigT> kernel_configs_;
};

struct BaseKernelConfig {
  std::unique_ptr<PtxKernelLoader> loader{nullptr};
  CUfunction kernel{nullptr};
  uint32_t shared_mem_bytes{0};
};

template <typename BlockSizeT>
struct GenericKernelConfig : public BaseKernelConfig {
  BlockSizeT block_size;
};

struct ThreeDimBlockSize {
  uint32_t x{0}, y{0}, z{0};
};

struct OneDimBlockSize {
  uint32_t x{0};
};

} // namespace xllm::kernel::cuda::triton