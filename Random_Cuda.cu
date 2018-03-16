#include <memory>

#include "BatchMatrix.hpp"
#include "Random.hpp"
#include "Cuda.hpp"

namespace cbm {

  namespace rand {

    template<typename ScaType, Type MemType>
    void runif(ScaType* dist, ScaType lb, ScaType ub, int len) {
      std::uniform_real_distribution<double> distribution(lb,ub);
      std::unique_ptr<ScaType[]> tmp(new ScaType[len]);
      for (int i = 0; i < len; i++) tmp[i] = static_cast<ScaType>(distribution(generator));
      CUDA_CHECK(cudaMemcpy(dist, tmp.get(),
			    len*sizeof(ScaType), cudaMemcpyHostToDevice));
    }

    template<typename ScaType, Type MemType>
    void runif(BatchMatrix<ScaType, MemType>* t, ScaType lb, ScaType ub) {
      std::uniform_real_distribution<double> distribution(lb,ub);
      std::unique_ptr<ScaType[]> tmp(new ScaType[t->len()]);
      for (int i = 0; i < t->len(); i++) tmp[i] = static_cast<ScaType>(distribution(generator));
      CUDA_CHECK(cudaMemcpy(t->data(), tmp.get(),
			    t->len()*sizeof(ScaType), cudaMemcpyHostToDevice));
    }

    template
    void runif<double, CUDA>(double* dist, double lb, double ub, int len);
    template
    void runif<float, CUDA>(float* dist, float lb, float ub, int len);
    template
    void runif<int, CUDA>(int* dist, int lb, int ub, int len);
    
    template
    void runif<double, CUDA>(BatchMatrix<double, CUDA>* t, double lb, double ub);
    template
    void runif<float, CUDA>(BatchMatrix<float, CUDA>* t, float lb, float ub);
    template
    void runif<int, CUDA>(BatchMatrix<int, CUDA>* t, int lb, int ub);
    
  }  // rand

}  // cbm
