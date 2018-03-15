// Include implementations that depend on macros
//   - clone functions
#include <memory>

#include <cuda.h>

#include "Cuda.hpp"
#include "BatchMatrix.hpp"

// CUDA to CPU
#define CBS_CLONE_CUDA_TO_CPU(T)					\
  template<>  BatchMatrix<T, CPU>*					\
  BatchMatrix<T, CUDA>::clone_cpu() const {				\
    auto ret = new BatchMatrix<T, CPU>(dim_[0], dim_[1], dim_[2]);	\
    CUDA_CHECK(cudaMemcpy(ret->data(), data_, len_*sizeof(T), cudaMemcpyDeviceToHost)); \
    return ret;								\
  }

// CPU to CUDA
#define CBS_CLONE_CPU_TO_CUDA(T)					\
  template<>  BatchMatrix<T, CUDA>*					\
  BatchMatrix<T, CPU>::clone_cuda() const {				\
    auto ret = new BatchMatrix<T, CUDA>(dim_[0], dim_[1], dim_[2]);	\
    CUDA_CHECK(cudaMemcpy(ret->data(), data_, len_*sizeof(T), cudaMemcpyHostToDevice)); \
    return ret;								\
  }

// CPU to CPU
#define CBS_CLONE_CPU_TO_CPU(T)						\
  template<>  BatchMatrix<T, CPU>*					\
  BatchMatrix<T, CPU>::clone_cpu() const {				\
    auto ret = new BatchMatrix<T, CPU>(dim_[0], dim_[1], dim_[2]);	\
    for (int i = 0; i < len_; i++) ret->data()[i] = data_[i];		\
    return ret;								\
  }

// CUDA to CUDA
#define CBS_CLONE_CUDA_TO_CUDA(T)					\
  template<>  BatchMatrix<T, CUDA>*					\
  BatchMatrix<T, CUDA>::clone_cuda() const {				\
    auto ret = new BatchMatrix<T, CUDA>(dim_[0], dim_[1], dim_[2]);	\
    CUDA_CHECK(cudaMemcpy(ret->data(), data_, len_*sizeof(T), cudaMemcpyDeviceToDevice)); \
    return ret;								\
  }



namespace cbm {

  CBS_CLONE_CUDA_TO_CPU(float);
  CBS_CLONE_CUDA_TO_CPU(double);
  CBS_CLONE_CUDA_TO_CPU(int);

  CBS_CLONE_CPU_TO_CUDA(float);
  CBS_CLONE_CPU_TO_CUDA(double);
  CBS_CLONE_CPU_TO_CUDA(int);

  CBS_CLONE_CUDA_TO_CUDA(float);
  CBS_CLONE_CUDA_TO_CUDA(double);
  CBS_CLONE_CUDA_TO_CUDA(int);

  CBS_CLONE_CPU_TO_CPU(float);
  CBS_CLONE_CPU_TO_CPU(double);
  CBS_CLONE_CPU_TO_CPU(int);
  
}  // namespace cbs

	
	
