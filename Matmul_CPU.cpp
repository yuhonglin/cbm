#include "Blas.hpp"
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
      
      for (int i = 0; i < a->dim()[0]; i++) {
	blas::gemm<ScaType>('N', 'N', a->dim()[1], b->dim()[2], a->dim()[2], 1.0,
			    a->data()+i*a->stride()[0], a->stride()[2],
			    b->data()+i*b->stride()[0], b->stride()[2], 0.0,
			    c->data()+i*c->stride()[0], c->stride()[2]);
      }
      
      return c;
    }

    template
    BatchMatrix<double, CPU>*
    matmul<double, CPU>(BatchMatrix<double, CPU>* a, BatchMatrix<double, CPU>* b);

    template
    BatchMatrix<float, CPU>*
    matmul<float, CPU>(BatchMatrix<float, CPU>* a, BatchMatrix<float, CPU>* b);
    
  }

}
