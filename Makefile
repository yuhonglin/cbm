all: test

BatchMatrix.o: BatchMatrix.cpp BatchMatrix.hpp
	g++ -std=c++11 -c BatchMatrix.cpp

BatchMatrix_CPU.o: BatchMatrix_CPU.cpp BatchMatrix.hpp
	g++ -std=c++11 -c BatchMatrix_CPU.cpp

Log.o: Log.cpp Log.hpp
	g++ -std=c++11 -c Log.cpp

IO_CPU.o: IO_CPU.cpp
	g++ -std=c++11 -c IO_CPU.cpp

Util_Cuda.o: Util_Cuda.cu
	nvcc -std=c++11 -c Util_Cuda.cu

test: BatchMatrix.o BatchMatrix_CPU.o Log.o IO_CPU.o test.cpp Util_Cuda.o
	g++ -std=c++11 BatchMatrix.o BatchMatrix_CPU.o IO_CPU.o Log.o test.cpp -o test
