#include <iostream>
#include <type_traits>
#include <vector>

#include "Log.hpp"
#include "BatchMatrix.hpp"
#include "IO.hpp"
#include "Decomp.hpp"
#include "Cublas.hpp"

int main(int argc, char *argv[])
{
  cbm::cublas::init();
  
  const cbm::Type MemType = cbm::CUDA;
  
  auto t = cbm::BatchMatrix<float, MemType>::ones({4,3,3});
  auto p = cbm::BatchMatrix<int, MemType>::zeros({4, 3, 1});
  auto info = cbm::BatchMatrix<int, MemType>::zeros({4, 1, 1});
  cbm::IO::print(*t);
  cbm::decomp::lu(t, p, info);
  cbm::IO::print(*t);
  cbm::IO::print(*p);
  cbm::IO::print(*info);
  
  return 0;
}
