#include <iostream>
#include <type_traits>
#include <vector>

#include "Log.hpp"
#include "BatchMatrix.hpp"
#include "IO.hpp"

int main(int argc, char *argv[])
{
  auto t = cbm::BatchMatrix<float, cbm::CUDA>::ones({3,20,10});

  cbm::IO::print(*t);

  return 0;
}
