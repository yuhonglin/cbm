#include <iostream>

#include "Log.hpp"
#include "IO.hpp"


namespace cbm {


  int IO::accuracy   = 4;
  int IO::num_left   = 3;
  int IO::num_right  = 3;
  int IO::num_top    = 4;
  int IO::num_bottom = 4;
  std::string IO::sep    = "\t";


  template<typename ScaType, Type MemType>
  void IO::print_row(const BatchMatrix<ScaType, MemType>& t, int mat_idx, int row_idx) {
    int j = 0;
    int first_idx = mat_idx*t.stride()[0] + row_idx*t.stride()[1];
    if (t.dim()[2] <= num_left + num_right) {
      for (; j < t.dim()[2]-1; j++) {
	std::printf("%.*f%s", accuracy,
		    static_cast<double>(t.data()[first_idx + j*t.stride()[2]]),
		    sep.c_str());
      }
      std::printf("%.*f", accuracy,
		  static_cast<double>(t.data()[first_idx + j*t.stride()[2]]));
    } else {
      for (; j < num_left; j++) {
	std::printf("%.*f%s", accuracy,
		    static_cast<double>(t.data()[first_idx + j*t.stride()[2]]),
		    sep.c_str());
      }
      std::printf("...%s", sep.c_str());
      j = t.dim()[2] - num_right;
      for (; j < t.dim()[2]-1; j++) {
	std::printf("%.*f%s", accuracy,
		    static_cast<double>(t.data()[first_idx + j*t.stride()[2]]),
		    sep.c_str());
      }
      std::printf("%.*f%s", accuracy,
		  static_cast<double>(t.data()[first_idx + j*t.stride()[2]]),
		  sep.c_str());
    }
  }

  
  template<typename ScaType, Type MemType>
  void IO::print_matrix(const BatchMatrix<ScaType, MemType>& t, int mat_idx) {
    if (t.dim()[1] > num_top+num_bottom) {
      int shift = mat_idx*t.dim()[0];
      for (int i = 0; i < num_top; i++) {
	print_row(t, mat_idx, i);
	std::printf("\n");
      }
      for (int i = 0; i < num_left+num_right+1; i++) {
	if (i == num_left) {
	  std::printf(" ⋱%s", sep.c_str());
	  continue;
	}
	std::printf("...%s", sep.c_str());
      }
      std::printf("\n");
      for (int i = t.dim()[1]-num_bottom; i < t.dim()[1]; i++) {
	print_row(t, mat_idx, i);
	std::printf("\n");
      }
    } else {
      for (int i = 0; i < t.dim()[1]; i++) {
	print_row(t, mat_idx, i);
	std::printf("\n");
      }
    }
  };
  

  template<typename ScaType, Type MemType>
  void IO::print(const BatchMatrix<ScaType, MemType>& t) {
    for (int i = 0; i < t.dim()[0]; i++) {
      std::printf("\n[%d, , ]\n", i);
      print_matrix(t, i);
    }
    std::printf("\n[%dx%dx%d BatchMatrix on CPU]\n\n", t.dim()[0], t.dim()[1], t.dim()[2]);
  }

  template void IO::print<float, CPU>(const BatchMatrix<float, CPU>& t);
  template void IO::print<double, CPU>(const BatchMatrix<double, CPU>& t);
  template void IO::print<int, CPU>(const BatchMatrix<int, CPU>& t);
  
}  // namespace cbm
