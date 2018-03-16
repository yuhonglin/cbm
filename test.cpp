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
  int m = 4;
  auto original = cbm::BatchMatrix<float, cbm::CPU>::unif({4,m,m});
  
  {  
    const cbm::Type MemType = cbm::CUDA;

    
  
    auto t = original->clone_cuda();
    auto p = cbm::BatchMatrix<int, MemType>::zeros({4,m,1});
    auto info = cbm::BatchMatrix<int, MemType>::zeros({4, 1, 1});
    //  cbm::IO::print(*t);
    cbm::decomp::lu(t, p, info);
    cbm::IO::print(*t);
    //  cbm::IO::print(*p);
    //  cbm::IO::print(*info);
  }

  {  
    const cbm::Type MemType = cbm::CPU;

    auto t = original->clone_cpu();
    auto p = cbm::BatchMatrix<int, MemType>::zeros({4,m,1});
    auto info = cbm::BatchMatrix<int, MemType>::zeros({4, 1, 1});
    //  cbm::IO::print(*t);
    cbm::decomp::lu(t, p, info);
    cbm::IO::print(*t);
    //  cbm::IO::print(*p);
    //  cbm::IO::print(*info);
  }
  
  cbm::cublas::clear();

  return 0;
}
