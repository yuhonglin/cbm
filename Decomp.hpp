#include <utility>

#include "BatchMatrix.hpp"

namespace cbm {
  namespace decomp {

    template<class ScaType, Type MemType>
    void lu(BatchMatrix<ScaType, MemType>* t,
	    BatchMatrix<int, MemType>* p,
	    BatchMatrix<int, MemType>* info);
    
  }  // decomp

}  // cbm
