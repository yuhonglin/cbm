#include "Fortran.hpp"

extern "C" {
  void FORTRAN(dgetrf)(int* M, int* N, double* A, int* lda, int* p, int *info);
  void FORTRAN(sgetrf)(int* M, int* N, float*  A, int* lda, int* p, int *info);
}

namespace cbm {
  namespace lapack {

    template<class T> void getrf(int M, int N, T* A, int lda, int* p, int *info);
    
    template<>
    void getrf<float>(int M, int N, float* A, int lda, int* p, int *info) {
      FORTRAN(sgetrf)(&M, &N, A, &lda, p, info);
    };

    template<>
    void getrf<double>(int M, int N, double* A, int lda, int* p, int *info) {
      FORTRAN(dgetrf)(&M, &N, A, &lda, p, info);
    };

  }  // lapack
}  // cbm
