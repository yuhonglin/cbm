#include "Log.hpp"
#include "Lapack.hpp"
#include "Decomp.hpp"


namespace cbm {

  namespace decomp {
    
    template<class ScaType, Type MemType>
    void lu(BatchMatrix<ScaType, MemType>* t,
	    BatchMatrix<int, MemType>* p,
	    BatchMatrix<int, MemType>* info) {
      
      int m    = t->dim()[1];
      int n    = t->dim()[2];
      int lda  = m;

      for (int i = 0; i < t->dim()[0]; i++)
	lapack::getrf<ScaType>(&m, &n, t->data()+i*t->stride()[0], &lda, p->data(), info->data());
    }

    template void lu(BatchMatrix<double, CPU>* t,
		     BatchMatrix<int, CPU>* p,
		     BatchMatrix<int, CPU>* info);

    template void lu(BatchMatrix<float, CPU>* t,
		     BatchMatrix<int, CPU>* p,
		     BatchMatrix<int, CPU>* info);

      
  }  // decomp

}  // cbm
