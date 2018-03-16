// Include CUDA implementations of BatchMatrix

#include <cstdint>
#include <memory>

#include <cuda.h>

#include "Log.hpp"
#include "Cuda.hpp"
#include "Util_Cuda.hpp"
#include "BatchMatrix.hpp"

namespace cbm {
  // Constructors
  template<typename ScaType, Type MemType>
  BatchMatrix<ScaType, MemType>::BatchMatrix(const BatchMatrix<ScaType, MemType>& t)
    : ptr_(nullptr) {
    static_assert(MemType==CUDA, "This function is only for constructing CUDA from CUDA");
    CUDA_CHECK(cudaMalloc(&data_, t.len()*sizeof(ScaType)));
    CUDA_CHECK(cudaMemcpy( data_, t.data(), t.len()*sizeof(ScaType), cudaMemcpyDeviceToDevice));

    for (int i=0; i<3; i++) {
      dim_[i]    = t.dim()[i];
      stride_[i] = t.stride()[i];
    }

    update_ptr();
  }

  // Destructor
  template<typename ScaType, Type MemType>
  BatchMatrix<ScaType, MemType>::~BatchMatrix() {
    static_assert(MemType==CUDA, "This function is only for CUDA");

    if (data_!=nullptr) CUDA_CHECK(cudaFree(data_));
  }

  template<typename ScaType, Type MemType>
  BatchMatrix<ScaType,MemType>::BatchMatrix(int a, int b, int c) : ptr_(nullptr) {
    static_assert(MemType==CUDA, "This function is only for CUDA");

    // copy
    dim_[0] = a; dim_[1] = b; dim_[2] = c;
    
    // default: column major for each matrix
    stride_[1] = 1;
    stride_[2] = dim_[1];
    stride_[0] = dim_[1]*dim_[2];
    len_       = stride_[0]*dim_[0];
    
    CUDA_CHECK(cudaMalloc(&data_, len_*sizeof(ScaType)));

    update_ptr();
  }


  template<typename ScaType, Type MemType>
  BatchMatrix<ScaType,MemType>::BatchMatrix(const std::vector<int>& d)
    : BatchMatrix(d[0], d[1], d[2]) {}

  template<typename ScaType, Type MemType>
  void BatchMatrix<ScaType,MemType>::update_ptr() {
    if (ptr_!=nullptr) CUDA_CHECK(cudaFree(ptr_));
    
    CUDA_CHECK(cudaMalloc(&ptr_, dim_[0]*sizeof(ScaType*)));

    std::unique_ptr<ScaType*[]> tmp(new ScaType*[dim_[0]]);
    for (int i = 0; i < dim_[0]; i++) tmp[i] = data_ + i*stride_[0];
    CUDA_CHECK(cudaMemcpy(ptr_, tmp.get(), dim_[0]*sizeof(ScaType*), cudaMemcpyHostToDevice));

    // //!The following code also works, but calling a kernel instead!
    //    inplace_set_inc<<<std::floor(dim_[0]/TPB)+1, TPB>>>(reinterpret_cast<std::intptr_t*>(ptr_),
    //    							reinterpret_cast<std::intptr_t>(data_),
    //    							static_cast<std::intptr_t>(stride_[0]*sizeof(std::intptr_t)),
    //       							1, dim_[0]);
         
  }

  template<typename ScaType, Type MemType>
  BatchMatrix<ScaType, MemType>*
    BatchMatrix<ScaType, MemType>::ones(const std::vector<int>& d) {
    static_assert(MemType==CUDA, "This function is only for CUDA");
    auto ret = new BatchMatrix<ScaType, MemType>(d);
    set_to_const<<<std::floor(ret->len()/TPB)+1, TPB>>>(ret->data(), static_cast<ScaType>(1), ret->len());

    CUDA_CHECK(cudaDeviceSynchronize());
    
    return ret;
  }

  template<typename ScaType, Type MemType>
  BatchMatrix<ScaType, MemType>*
    BatchMatrix<ScaType, MemType>::zeros(const std::vector<int>& d) {
    static_assert(MemType==CUDA, "This function is only for CUDA");
    auto ret = new BatchMatrix<ScaType, MemType>(d);
    set_to_const<<<std::floor(ret->len()/TPB)+1, TPB>>>(ret->data(), static_cast<ScaType>(0), ret->len());
    return ret;
  }

  template<typename ScaType, Type MemType>
  void BatchMatrix<ScaType, MemType>::add(ScaType v) {
    inplace_add<<<std::floor(len_/TPB)+1, TPB>>>(data_, v, len_);
  }

  template class BatchMatrix<float, CUDA>;
  template class BatchMatrix<int, CUDA>;
  template class BatchMatrix<double, CUDA>;
    
    
}  // namespace cbm
