// Some common stuff for using cuda 
#ifndef CUDA_H
#define CUDA_H

#include <cuda.h>
#include <cublas.h>

namespace cbm {

#define LINEAR_IDX (blockDim.x * blockIdx.x + threadIdx.x)
  
  // dimension of block
  const dim3 DB(256,1,1);
  // thread per block
  const int TPB = DB.x*DB.y*DB.z;


  // Cuda check

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)						\
  {									\
    const cudaError_t error = call;					\
    if (error != cudaSuccess)						\
      {									\
	printf("Error: %s:%d, ", __FILE__, __LINE__);			\
	printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
	exit(1);							\
      }									\
  }
#endif


  // cuBlas check

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



#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(call)						\
  {									\
    cublasStatus_t stat = call;						\
    if (stat != CUBLAS_STATUS_SUCCESS)					\
      {									\
	printf("Error: %s:%d, ", __FILE__, __LINE__);			\
	printf("stat:%d, reason: %s\n", stat, cbs::cublasGetErrorString(stat)); \
	exit(1);							\
      }									\
  }
#endif

}  // cbs
  
#endif /* CUDA_H */
