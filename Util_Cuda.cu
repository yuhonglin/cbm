#include <cstdint>

#include "Cuda.hpp"
#include "Util_Cuda.hpp"

namespace cbm {

  template<typename ScaType>
  __global__ void set_to_zero(ScaType* p, int len) {
    int idx = LINEAR_IDX;
    if (idx < len) p[idx] = 0;
  }

  template<typename ScaType>
  __global__ void set_to_one(ScaType* p, int len) {
    int idx = LINEAR_IDX;
    if (idx < len) p[idx] = 1;
  }

  template<typename ScaType>
  __global__ void set_to_const(ScaType* p, ScaType v, int len) {
    int idx = LINEAR_IDX;
    if (idx < len) p[idx] = v;
  }

  template<typename ScaType>
  __global__ void inplace_multiply(ScaType* p, ScaType v, int len) {
    int idx = LINEAR_IDX;
    if (idx < len) p[idx] *= v;
  }

  template<typename ScaType>
  __global__ void inplace_divide(ScaType* p, ScaType v, int len) {
    int idx = LINEAR_IDX;
    if (idx < len) p[idx] /= v;
  }

  template<typename ScaType>
  __global__ void inplace_add(ScaType* p, ScaType v, int len) {
    int idx = LINEAR_IDX;
    if (idx < len) p[idx] += v;
  }

  template<typename ScaType>
  __global__ void inplace_minus(ScaType* p, ScaType v, int len) {
    int idx = LINEAR_IDX;
    if (idx < len) p[idx] -= v;
  }

  // set increasing values to an array
  // set an array to v0, v0+inc, v0+2*inc,..., v0+(len-1)*inc
  template<typename ScaType>
  __global__ void inplace_set_inc(ScaType* p, ScaType v0, ScaType inc, int stride, int len) {
    int idx = LINEAR_IDX;
    if (idx < len) {
      int i = idx*stride;
      p[i] = v0 + idx*inc;
    }
  }
  
  template __global__ void set_to_zero<float>(float* p, int len);
  template __global__ void set_to_zero<double>(double* p, int len);
  template __global__ void set_to_zero<int>(int* p, int len);

  template __global__ void set_to_one<float>(float* p, int len);
  template __global__ void set_to_one<double>(double* p, int len);
  template __global__ void set_to_one<int>(int* p, int len);

  template __global__ void set_to_const<float>(float* p, float v, int len);
  template __global__ void set_to_const<double>(double* p, double v, int len);
  template __global__ void set_to_const<int>(int* p, int v, int len);


  template __global__ void inplace_multiply<float>(float* p, float v, int len);
  template __global__ void inplace_multiply<double>(double* p, double v, int len);
  template __global__ void inplace_multiply<int>(int* p, int v, int len);

  template __global__ void inplace_divide<float>(float* p, float v, int len);
  template __global__ void inplace_divide<double>(double* p, double v, int len);
  template __global__ void inplace_divide<int>(int* p, int v, int len);

  template __global__ void inplace_add<float>(float* p, float v, int len);
  template __global__ void inplace_add<double>(double* p, double v, int len);
  template __global__ void inplace_add<int>(int* p, int v, int len);

  template __global__ void inplace_minus<float>(float* p, float v, int len);
  template __global__ void inplace_minus<double>(double* p, double v, int len);
  template __global__ void inplace_minus<int>(int* p, int v, int len);

  template __global__ void inplace_set_inc<float>(float* p, float v0, float inc, int stride, int len);
  template __global__ void inplace_set_inc<double>(double* p, double v0, double inc, int stride, int len);
  template __global__ void inplace_set_inc<int>(int* p, int v0, int inc, int stride, int len);
  template __global__ void inplace_set_inc<std::intptr_t>(std::intptr_t* p, std::intptr_t v0, std::intptr_t inc, int stride, int len);
  
}  // namespace cbm