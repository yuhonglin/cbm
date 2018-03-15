#include "IO.hpp"

namespace cbm {
  template<typename ScaType, Type MemType>
  void IO::print(const BatchMatrix<ScaType, MemType>& t) {
    auto bm = t.clone_cpu();
    for (int i = 0; i < bm->dim()[0]; i++) {
      std::printf("\n[%d, , ]\n", i);
      print_matrix(*bm, i);
    }
    delete bm;
    std::printf("\n[%dx%dx%d BatchMatrix on CUDA]\n\n", t.dim()[0], t.dim()[1], t.dim()[2]);
  }

  template void IO::print<float, CUDA>(const BatchMatrix<float, CUDA>& t);
  template void IO::print<double, CUDA>(const BatchMatrix<double, CUDA>& t);
  template void IO::print<int, CUDA>(const BatchMatrix<int, CUDA>& t);
  
}