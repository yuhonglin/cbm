LAPACK_LIB = /usr/lib/liblapack.a

all: test

BatchMatrix.o: BatchMatrix.cpp BatchMatrix.hpp Cuda.hpp
	g++ -std=c++11 -c BatchMatrix.cpp

BatchMatrix_CPU.o: BatchMatrix_CPU.cpp BatchMatrix.hpp Cuda.hpp
	g++ -std=c++11 -c BatchMatrix_CPU.cpp

Log.o: Log.cpp Log.hpp Cuda.hpp
	g++ -std=c++11 -c Log.cpp

IO_CPU.o: IO_CPU.cpp Cuda.hpp
	g++ -std=c++11 -c IO_CPU.cpp

IO_Cuda.o: IO_Cuda.cu Cuda.hpp
	nvcc -std=c++11 -c IO_Cuda.cu

Util_Cuda.o: Util_Cuda.cu Cuda.hpp
	nvcc -std=c++11 -c Util_Cuda.cu

BatchMatrix_Cuda.o: BatchMatrix_Cuda.cu Cuda.hpp
	nvcc -std=c++11 -c BatchMatrix_Cuda.cu

BatchMatrix_macro.o: BatchMatrix_macro.cu Cuda.hpp
	nvcc -std=c++11 -c BatchMatrix_macro.cu

LU_CPU.o: LU_CPU.cpp Decomp.hpp Lapack.hpp
	g++ -std=c++11 -c LU_CPU.cpp

LU_Cuda.o: LU_Cuda.cu Decomp.hpp Cublas.hpp
	nvcc -std=c++11 -c LU_Cuda.cu

Cublas.o: Cublas.cu Cublas.hpp
	nvcc -std=c++11 -c Cublas.cu

Random_CPU.o: Random_CPU.cpp Random.hpp
	g++ -std=c++11 -c Random_CPU.cpp

Random_Cuda.o: Random_Cuda.cu Random.hpp
	nvcc -std=c++11 -c Random_Cuda.cu

Matmul_CPU.o: Matmul_CPU.cpp Blas.hpp
	g++ -std=c++11 -c Matmul_CPU.cpp

Matmul_Cuda.o: Matmul_Cuda.cu Blas.hpp
	g++ -std=c++11 -c Matmul_Cuda.cu


test: BatchMatrix.o BatchMatrix_CPU.o Log.o IO_CPU.o test.cpp Util_Cuda.o BatchMatrix_Cuda.o BatchMatrix_macro.o IO_Cuda.o Cuda.hpp LU_CPU.o LU_Cuda.o Cublas.o Random_CPU.o Random_Cuda.o Matmul_CPU.o
	nvcc -std=c++11 BatchMatrix.o BatchMatrix_CPU.o IO_CPU.o Log.o test.cpp Util_Cuda.o BatchMatrix_Cuda.o BatchMatrix_macro.o IO_Cuda.o LU_CPU.o LU_Cuda.o Cublas.o Random_CPU.o Random_Cuda.o Matmul_CPU.o Matmul_Cuda.o -llapack -lcublas -lblas -o test

