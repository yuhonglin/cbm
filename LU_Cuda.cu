#include "Cuda.hpp"

#include "Log.hpp"
#include "Decomp.hpp"
#include "Cublas.hpp"

namespace cbm {

  namespace decomp {

    template<class ScaType, Type MemType>
    void lu(BatchMatrix<ScaType, MemType>* t,
	    BatchMatrix<int, MemType>* p,
	    BatchMatrix<int, MemType>* info) {
      
      ASSERT(t->dim()[1]==t->dim()[2]); // can only handle square matrix
      
      int n    = t->dim()[1];
      int lda  = n;

      cublas::cublasGetrfBatched<ScaType>(cublas::default_cublas_handle,
					  n, t->ptr(), lda,
					  p->data(), info->data(), t->dim()[0]);

      CUDA_CHECK(cudaDeviceSynchronize());
    }
    

    template void lu(BatchMatrix<double, CUDA>* t,
		     BatchMatrix<int, CUDA>* p,
		     BatchMatrix<int, CUDA>* info);

    template void lu(BatchMatrix<float, CUDA>* t,
		     BatchMatrix<int, CUDA>* p,
		     BatchMatrix<int, CUDA>* info);
  }
}