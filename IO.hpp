#ifndef IO_H
#define IO_H

#include <string>

#include "BatchMatrix.hpp"

namespace cbm {
  
    class IO {
    private:
      template<typename ScaType, Type MemType>      
      static void print_row(const BatchMatrix<ScaType, MemType>& t, int mat_idx, int row_idx);

      template<typename ScaType, Type MemType>
      static void print_matrix(const BatchMatrix<ScaType, MemType>& t, int mat_idx);
      
    public:
      static int accuracy;
      static int num_left;
      static int num_right;
      static int num_top;
      static int num_bottom;
      static std::string sep;

      template<typename ScaType, Type MemType>      
      static void print(const BatchMatrix<ScaType, MemType>& t);
      
    };
    
}  // namespace cbm

#endif /* IO_H */
