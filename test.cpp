#include <iostream>
#include <type_traits>
#include <vector>
#include <cmath>

#include "Cuda.hpp"
#include "Log.hpp"
#include "BatchMatrix.hpp"
#include "IO.hpp"
#include "Decomp.hpp"
#include "Cublas.hpp"


int main(int argc, char *argv[])
{
  cbm::cublas::init();
  
  const cbm::Type MemType = cbm::CUDA;

  int m = 12;
  
  auto t = cbm::BatchMatrix<float, MemType>::ones({4,m,m});
  auto p = cbm::BatchMatrix<int, MemType>::zeros({4,m,1});
  auto info = cbm::BatchMatrix<int, MemType>::zeros({4, 1, 1});
  cbm::IO::print(*t);
  cbm::decomp::lu(t, p, info);
  cbm::IO::print(*t);
  cbm::IO::print(*p);
  cbm::IO::print(*info);

  int num_one = 0;
  float sum = 0.;
  auto tmp = t->clone_cpu();
  for (int i = 0; i < t->len(); i++) {
    if (std::abs(tmp->data()[i]-1) < 1e-6) {
      num_one++;
    }
    sum += tmp->data()[i];
  }
  LOG_INFO(num_one);
  LOG_INFO(sum);

  cublasDestroy_v2(cbm::cublas::default_cublas_handle);

  return 0;
}
