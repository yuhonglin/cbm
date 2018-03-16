#include <iostream>
#include <mutex>

#include "Cublas.hpp"

namespace cbm {
  namespace cublas {

    // the default cublas handle
    cublasHandle_t default_cublas_handle;

    static const char *cublasGetErrorString(cublasStatus_t error)
    {
      switch (error)
	{
	case CUBLAS_STATUS_SUCCESS:
	  return "CUBLAS_STATUS_SUCCESS";

	case CUBLAS_STATUS_NOT_INITIALIZED:
	  return "CUBLAS_STATUS_NOT_INITIALIZED";

	case CUBLAS_STATUS_ALLOC_FAILED:
	  return "CUBLAS_STATUS_ALLOC_FAILED";

	case CUBLAS_STATUS_INVALID_VALUE:
	  return "CUBLAS_STATUS_INVALID_VALUE";

	case CUBLAS_STATUS_ARCH_MISMATCH:
	  return "CUBLAS_STATUS_ARCH_MISMATCH";

	case CUBLAS_STATUS_MAPPING_ERROR:
	  return "CUBLAS_STATUS_MAPPING_ERROR";

	case CUBLAS_STATUS_EXECUTION_FAILED:
	  return "CUBLAS_STATUS_EXECUTION_FAILED";

	case CUBLAS_STATUS_INTERNAL_ERROR:
	  return "CUBLAS_STATUS_INTERNAL_ERROR";
	}

      return "<unknown>";
    }
    
    template<>
    void cublasGetrfBatched<float>(cublasHandle_t& handle, int n, float *Array[],
				   int lda, int *PivotArray, int*infoArray,
				   int batchSize) {
      CUBLAS_CHECK(cublasSgetrfBatched(handle, n, Array, lda, PivotArray, infoArray, batchSize));
    };


    template<>
    void cublasGetrfBatched<double>(cublasHandle_t& handle, int n, double *Array[],
				    int lda, int *PivotArray, int*infoArray,
				    int batchSize) {
      CUBLAS_CHECK(cublasDgetrfBatched(handle, n, Array, lda, PivotArray, infoArray, batchSize));
    };


    void init() {
      static bool first = true;
      if (first) {
	CUBLAS_CHECK(cublasCreate(&default_cublas_handle));
	first = false;
      }
    }

    void clear() {
      cublasDestroy_v2(default_cublas_handle);
    }
  }
}
