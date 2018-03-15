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

test: BatchMatrix.o BatchMatrix_CPU.o Log.o IO_CPU.o test.cpp Util_Cuda.o BatchMatrix_Cuda.o BatchMatrix_macro.o IO_Cuda.o Cuda.hpp
	nvcc -std=c++11 BatchMatrix.o BatchMatrix_CPU.o IO_CPU.o Log.o test.cpp Util_Cuda.o BatchMatrix_Cuda.o BatchMatrix_macro.o IO_Cuda.o -o test
