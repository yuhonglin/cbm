#ifndef MATH_HPP
#define MATH_HPP

#include "BatchMatrix.hpp"

namespace cbm {

  namespace math {

    // matrix multiplication
    template<typename ScaType, Type MemType>
    BatchMatrix<ScaType, MemType>*
    matmul(BatchMatrix<ScaType, MemType>* a, BatchMatrix<ScaType, MemType>* b);
    
  }  // math

}  // cbm

#endif /* MATH_HPP */
