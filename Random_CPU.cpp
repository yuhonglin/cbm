#include "BatchMatrix.hpp"
#include "Random.hpp"

namespace cbm {

  namespace rand {

    std::default_random_engine generator;

    template<typename ScaType, Type MemType>
    void runif(ScaType* dist, ScaType lb, ScaType ub, int len) {
      std::uniform_real_distribution<double> distribution(lb,ub);
      for (int i = 0; i < len; i++) dist[i] = static_cast<ScaType>(distribution(generator));
    }

    template<typename ScaType, Type MemType>
    void runif(BatchMatrix<ScaType, MemType>* t, ScaType lb, ScaType ub) {
      std::uniform_real_distribution<double> distribution(lb,ub);
      for (int i = 0; i < t->len(); i++) t->data()[i] = static_cast<ScaType>(distribution(generator));
    }

    template
    void runif<double, CPU>(double* dist, double lb, double ub, int len);
    template
    void runif<float, CPU>(float* dist, float lb, float ub, int len);
    template
    void runif<int, CPU>(int* dist, int lb, int ub, int len);
    
    template
    void runif<double, CPU>(BatchMatrix<double, CPU>* t, double lb, double ub);
    template
    void runif<float, CPU>(BatchMatrix<float, CPU>* t, float lb, float ub);
    template
    void runif<int, CPU>(BatchMatrix<int, CPU>* t, int lb, int ub);
    
  }  // rand

}  // cbm
