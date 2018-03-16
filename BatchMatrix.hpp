// BatchMatrix class has following responsibilities
// 1. Creation
//    BatchMatrix(m,n), BatchMatrix::ones(), BatchMatrix::zeros(), BatchMatrix::rand() ...
// 2. Copy / move
// 3. Clone and switch between CPU and GPU
// 4. Interfaces to inner parameters

#ifndef BATCHMATRIX_H
#define BATCHMATRIX_H

#include <string>
#include <memory>
#include <vector>

namespace cbm {

  enum Type { CPU, CUDA };
    
  template<typename ScaType, Type MemType>
  class BatchMatrix
  {
  public:
    BatchMatrix() = delete;
    BatchMatrix(const std::vector<int>& d);
    BatchMatrix(int a, int b, int c);
    //    BatchMatrix<ScaType, MemType>&
    //    operator=(const BatchMatrix<ScaType, MemType>& d) = delete;
    //    BatchMatrix(const BatchMatrix<ScaType, MemType>&& d) = delete;
    ~BatchMatrix();
	
    // Copy / move
    BatchMatrix(const BatchMatrix<ScaType, MemType>&  t);
    //    BatchMatrix(BatchMatrix<ScaType, MemType>&& t) = delete;
	
    // Creation
    static BatchMatrix<ScaType, MemType>*
      ones(const std::vector<int>& d);

    static BatchMatrix<ScaType, MemType>*
      zeros(const std::vector<int>& d);
	
    // Clone
    BatchMatrix<ScaType, CPU>* clone_cpu() const;
    BatchMatrix<ScaType, CUDA>* clone_cuda() const;

    // Getters
    Type type() const;
    ScaType* data() const; // use with care
    ScaType** ptr() const;
    int len() const;
    const int* dim() const;
    const int* stride() const;

    // Setter
    void set_data(ScaType* d);
    void set_dim(int a, int b, int c);
    void set_stride(int a, int b, int c);

    // arithmetic
    void add(ScaType v);
    void multiply(ScaType v);
    void divide(ScaType v);
    void minus(ScaType v);
    
  private:
    // meta data
    int dim_[3];
    int stride_[3];
    int len_;

    // The data pointer
    // It is convenient to be a raw pointer because it may point to device
    ScaType* data_;
    
    // The reason for this memeber is that cublas batch routines only accept pointer
    // array
    ScaType** ptr_;
  };

}  // namespace cbs

#endif /* BATCHMATRIX_H */
