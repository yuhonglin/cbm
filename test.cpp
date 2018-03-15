#include <iostream>
#include <type_traits>
#include <vector>

#include "Log.hpp"
#include "BatchMatrix.hpp"
#include "IO.hpp"
#include "Decomp.hpp"

int main(int argc, char *argv[])
{
  auto t = cbm::BatchMatrix<float, cbm::CPU>::ones({3,20,10});

  auto p = cbm::BatchMatrix<int, cbm::CPU>::zeros({3, 10, 1});

  auto info = cbm::BatchMatrix<int, cbm::CPU>::zeros({3, 1, 1});
  
  cbm::IO::print(*t);

  cbm::decomp::lu(t, p, info);

  cbm::IO::print(*t);
  cbm::IO::print(*p);
  cbm::IO::print(*info);
  
  return 0;
}
