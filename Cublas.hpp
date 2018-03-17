#ifndef CUBLAS_H
#define CUBLAS_H

#include <cublas_v2.h>

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(call)						\
  {									\
    cublasStatus_t stat = call;						\
    if (stat != CUBLAS_STATUS_SUCCESS)					\
      {									\
	printf("Error: %s:%d, ", __FILE__, __LINE__);			\
	printf("stat:%d, reason: %s\n", stat, cbm::cublas::cublasGetErrorString(stat)); \
	exit(1);							\
      }									\
  }
#endif


namespace cbm {
  namespace cublas {
    
    extern cublasHandle_t default_cublas_handle;

    // cuBlas check
    static const char *cublasGetErrorString(cublasStatus_t error);

    template<class T> void cublasGetrfBatched(cublasHandle_t& handle, int n, T *Array[],
					      int lda, int *PivotArray, int*infoArray,
					      int batchSize);
    
    template<class T> void cublasGemmBatched(cublasHandle_t& handle, cublasOperation_t transa,
					     cublasOperation_t transb, int m, int n, int k,
					     T alpha, T *Aarray[], int lda, T *Barray[], int ldb,
					     T beta, T *Carray[], int ldc, int batchCount);

    void init();
    void clear();
  } // namespace cublas
}   // namespace cbm


#endif /* CUBLAS_H */
