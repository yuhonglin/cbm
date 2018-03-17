#ifndef BLAS_HPP
#define BLAS_HPP

#include "Fortran.hpp"

extern "C" {
  void FORTRAN(dgemm)(char* TRANSA, char* TRANSB, int* M, int* N, int* K, double* ALPHA,
		      double* A, int* LDA, double* B, int* LDB, double* BETA,
		      double* C, int* LDC);
  void FORTRAN(sgemm)(char* TRANSA, char* TRANSB, int* M, int* N, int* K, float* ALPHA,
		      float* A, int* LDA, float* B, int* LDB, float* BETA,
		      float* C, int* LDC);
}

namespace cbm {
  namespace blas {
    template<class T> void gemm(char TRANSA, char TRANSB, int M, int N, int K, T ALPHA,
				T* A, int LDA, T* B, int LDB, T BETA, T* C, int LDC);

    template<>
    void gemm<float>(char TRANSA, char TRANSB, int M, int N, int K, float ALPHA,
		     float* A, int LDA, float* B, int LDB, float BETA, float* C, int LDC) {
      FORTRAN(sgemm)(&TRANSA, &TRANSB, &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC);
    };

    template<>
    void gemm<double>(char TRANSA, char TRANSB, int M, int N, int K, double ALPHA,
		      double* A, int LDA, double* B, int LDB, double BETA, double* C, int LDC) {
      FORTRAN(dgemm)(&TRANSA, &TRANSB, &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC);
    };

  }
}


#endif /* BLAS_H */
