#ifndef RANDOM_HPP
#define RANDOM_HPP

#include <random>

#include "BatchMatrix.hpp"

namespace cbm {

  namespace rand {

    extern std::default_random_engine generator;

    template<typename ScaType, Type MemType>
    void runif(ScaType* dist, ScaType lb, ScaType ub, int len);

    template<typename ScaType, Type MemType>
    void runif(BatchMatrix<ScaType, MemType>* t, ScaType lb, ScaType ub);

  }  // rand

}  // cbm

#endif /* RANDOM_H */
