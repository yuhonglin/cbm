// some utility routine on CUDA
#ifndef UTIL_CUDA_H
#define UTIL_CUDA_H


namespace cbm {

  template<typename ScaType>
  __global__ void set_to_zero(ScaType* p, int len);

  template<typename ScaType>
  __global__ void set_to_one(ScaType* p, int len);

  template<typename ScaType>
  __global__ void set_to_const(ScaType* p, ScaType v, int len);

  template<typename ScaType>
  __global__ void inplace_add(ScaType* p, ScaType v, int len);

  template<typename ScaType>
  __global__ void inplace_minus(ScaType* p, ScaType v, int len);

  template<typename ScaType>
  __global__ void inplace_multiply(ScaType* p, ScaType v, int len);

  template<typename ScaType>
  __global__ void inplace_divide(ScaType* p, ScaType v, int len);

  template<typename ScaType>
  __global__ void inplace_set_inc(ScaType* p, ScaType v0, ScaType stp, int stride, int len);
  
}  // namespace cbm


#endif /* UTIL_CUDA_H */

