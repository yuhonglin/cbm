// Some common stuff for using cuda 
#ifndef CUDA_HPP
#define CUDA_HPP

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
	printf("Cuda Error: %s:%d, ", __FILE__, __LINE__);		\
	printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
	exit(1);							\
      }									\
  }
#endif
  
}  // cbs
  
#endif /* CUDA_H */
