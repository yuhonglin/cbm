// Include CPU specific implementations
#include <memory>

#include <cstring>
#include <type_traits>

#include "BatchMatrix.hpp"
#include "Log.hpp"

namespace cbm {

  template<typename ScaType, Type MemType>
  ScaType* BatchMatrix<ScaType, MemType>::data() const {
    return data_;
  }

  template<typename ScaType, Type MemType>
  const int* BatchMatrix<ScaType, MemType>::dim() const {
    return dim_;
  }

  template<typename ScaType, Type MemType>
  int BatchMatrix<ScaType, MemType>::len() const {
    return len_;
  }

  template<typename ScaType, Type MemType>
  const int* BatchMatrix<ScaType, MemType>::stride() const {
    return stride_;
  }

  template<typename ScaType, Type MemType>
  Type BatchMatrix<ScaType, MemType>::type() const {
    return MemType;
  }

  
  template class BatchMatrix<float, CPU>;
  template class BatchMatrix<int, CPU>;
  template class BatchMatrix<double, CPU>;

  template class BatchMatrix<float, CUDA>;
  template class BatchMatrix<int, CUDA>;
  template class BatchMatrix<double, CUDA>;
  
}  // namespace cbs