#include <memory>
#include <iostream>
#include <cstring>
#include <type_traits>

#include "BatchMatrix.hpp"
#include "Log.hpp"

namespace cbm {
  // Constructors 
  template<typename ScaType, Type MemType>
  BatchMatrix<ScaType, MemType>::BatchMatrix(const BatchMatrix<ScaType, MemType>& t)
    : ptr_(nullptr) {
    static_assert(MemType==CPU, "This function is only for constructing CPU from CPU");
    
    data_ = new ScaType[t.len()];
    std::memcpy(data_, t.data(), len_*sizeof(ScaType));

    for (int i=0; i<3; i++) {
      dim_[i]    = t.dim()[i];
      stride_[i] = t.stride()[i];
    }

    update_ptr();
  }

  // Destructor
  template<typename ScaType, Type MemType>
  BatchMatrix<ScaType, MemType>::~BatchMatrix() {
    static_assert(MemType==CPU, "This function is only for CPU");

    if (data_!=nullptr) delete[] data_;
    if (ptr_!=nullptr) delete[] ptr_;
  }


  template<typename ScaType, Type MemType>
  BatchMatrix<ScaType,MemType>::BatchMatrix(int a, int b, int c) : ptr_(nullptr) {
    static_assert(MemType==CPU, "This function is only for CPU");

    // copy
    dim_[0] = a; dim_[1] = b; dim_[2] = c;
    
    // default: column major for each matrix
    stride_[1] = 1;
    stride_[2] = dim_[1];
    stride_[0] = dim_[1]*dim_[2];
    len_       = stride_[0]*dim_[0];
    
    data_ = new ScaType[len_];

    update_ptr();
  }

  template<typename ScaType, Type MemType>
  void BatchMatrix<ScaType,MemType>::update_ptr() {
    if (ptr_!=nullptr) delete[] ptr_;

    ptr_ = new ScaType*[dim_[0]];
    for (int i = 0; i < dim_[0]; i++) {
      ptr_[i] = data_ + i*stride_[0];
    }
  }
  
  template<typename ScaType, Type MemType>
  BatchMatrix<ScaType,MemType>::BatchMatrix(const std::vector<int>& d)
    : BatchMatrix(d[0], d[1], d[2]) {}
  
  template<typename ScaType, Type MemType>
  BatchMatrix<ScaType, MemType>*
    BatchMatrix<ScaType, MemType>::ones(const std::vector<int>& d) {
    static_assert(MemType==CPU, "This function is only for CPU");
    ASSERT(d.size()==3);

    auto ret = new BatchMatrix<ScaType, MemType>(d);
    
    for(int i = 0; i < ret->len(); i++) {
      ret->data()[i] = 1;
    }
    return ret;
  }

  template<typename ScaType, Type MemType>
  BatchMatrix<ScaType, MemType>*
    BatchMatrix<ScaType,MemType>::zeros(const std::vector<int>& d) {
    static_assert(MemType==CPU, "This function is only for CPU");
    auto ret = new BatchMatrix<ScaType,MemType>(d);
    for(int i = 0; i < ret->len(); i++) {
      ret->data()[i] = 0;
    }
    return ret;
  }

  template<typename ScaType, Type MemType>
  void BatchMatrix<ScaType, MemType>::add(ScaType v) {
    for (int i = 0; i < len_; i++) data_[i] += v;
  }

  template<typename ScaType, Type MemType>
  void BatchMatrix<ScaType, MemType>::minus(ScaType v) {
    for (int i = 0; i < len_; i++) data_[i] -= v;
  }

  template<typename ScaType, Type MemType>
  void BatchMatrix<ScaType, MemType>::divide(ScaType v) {
    for (int i = 0; i < len_; i++) data_[i] /= v;
  }

  template<typename ScaType, Type MemType>
  void BatchMatrix<ScaType, MemType>::multiply(ScaType v) {
    for (int i = 0; i < len_; i++) data_[i] *= v;
  }
  
  template class BatchMatrix<float, CPU>;
  template class BatchMatrix<int, CPU>;
  template class BatchMatrix<double, CPU>;
    
}  // namespace cbs
