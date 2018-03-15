// Include CUDA implementations of BatchMatrix

#include <cuda.h>

#include "Cuda.hpp"
#include "Util_Cuda.hpp"
#include "BatchMatrix.hpp"

namespace cbm {
  // Constructors
  template<typename ScaType, Type MemType>
  BatchMatrix<ScaType, MemType>::BatchMatrix(const BatchMatrix<ScaType, MemType>& t) {
    static_assert(MemType==CUDA, "This function is only for constructing CUDA from CUDA");
    CUDA_CHECK(cudaMalloc(&data_, t.len()*sizeof(ScaType)));
    CUDA_CHECK(cudaMemcpy( data_, t.data(), t.len()*sizeof(ScaType), cudaMemcpyDeviceToDevice));

    for (int i=0; i<3; i++) {
      dim_[i]    = t.dim()[i];
      stride_[i] = t.stride()[i];
    }
  }

  // Destructor
  template<typename ScaType, Type MemType>
  BatchMatrix<ScaType, MemType>::~BatchMatrix() {
    static_assert(MemType==CUDA, "This function is only for CUDA");

    if (data_!=nullptr) CUDA_CHECK(cudaFree(data_));
  }


  template<typename ScaType, Type MemType>
  BatchMatrix<ScaType,MemType>::BatchMatrix(int a, int b, int c) {
    static_assert(MemType==CUDA, "This function is only for CUDA");

    // copy
    dim_[0] = a; dim_[1] = b; dim_[2] = c;
    
    // default: column major for each matrix
    stride_[1] = 1;
    stride_[2] = dim_[1];
    stride_[0] = dim_[1]*dim_[2];
    len_       = stride_[0]*dim_[0];

    CUDA_CHECK(cudaMalloc(&data_, len_*sizeof(ScaType)));
  }


  template<typename ScaType, Type MemType>
  BatchMatrix<ScaType,MemType>::BatchMatrix(const std::vector<int>& d)
    : BatchMatrix(d[0], d[1], d[2]) {}


  template<typename ScaType, Type MemType>
  BatchMatrix<ScaType, MemType>*
    BatchMatrix<ScaType, MemType>::ones(const std::vector<int>& d) {
    static_assert(MemType==CUDA, "This function is only for CUDA");
    auto ret = new BatchMatrix<ScaType, MemType>(d);
    set_to_const<<<std::floor(ret->len()/TPB)+1, TPB>>>(ret->data(), static_cast<ScaType>(1), ret->len());
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

  // Clone (CUDA to CUDA)
  template<typename ScaType, Type MemType>
  BatchMatrix<ScaType, MemType>* BatchMatrix<ScaType, MemType>::clone_cuda() const {
    static_assert(MemType==CUDA, "This function is only for CUDA to CUDA clone");
    auto ret = new BatchMatrix<ScaType, MemType>(dim_[0], dim_[1], dim_[2]);
    CUDA_CHECK(cudaMemcpy(ret->data(), data_, len_*sizeof(MemType),cudaMemcpyDeviceToDevice));
    return ret;
  };


  
  template<typename ScaType, Type MemType>
  void BatchMatrix<ScaType, MemType>::add(ScaType v) {
    inplace_add<<<std::floor(len_/TPB)+1, TPB>>>(data_, v, len_);
  }

  template class BatchMatrix<float, CUDA>;
  template class BatchMatrix<int, CUDA>;
  template class BatchMatrix<double, CUDA>;
    
    
}  // namespace cbm
