#include "Cublas.hpp"

#include "Log.hpp"
#include "Math.hpp"

namespace cbm {

  namespace math {

    template<typename ScaType, Type MemType>
    BatchMatrix<ScaType, MemType>*
    matmul(BatchMatrix<ScaType, MemType>* a, BatchMatrix<ScaType, MemType>* b) {
      ASSERT(a->dim()[0]==b->dim()[0]);
      ASSERT(a->dim()[2]==b->dim()[1]);

      auto c = new BatchMatrix<ScaType, MemType>(a->dim()[0], a->dim()[1], b->dim()[2]);

      cublas::cublasGemmBatched<ScaType>(cublas::default_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
					 a->dim()[1], b->dim()[2], a->dim()[2], 1.0,
					 a->ptr(), a->stride()[2],
					 b->ptr(), b->stride()[2], 0.0,
					 c->ptr(), c->stride()[2], a->dim()[0]);
      return c;
    }

    template
    BatchMatrix<double, CUDA>*
    matmul<double, CUDA>(BatchMatrix<double, CUDA>* a, BatchMatrix<double, CUDA>* b);

    template
    BatchMatrix<float, CUDA>*
    matmul<float, CUDA>(BatchMatrix<float, CUDA>* a, BatchMatrix<float, CUDA>* b);
    
  }

}
