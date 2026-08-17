main: main.cpp
	g++ -std=c++17 main.cpp -O3
	./a.out > img.ppm

cuda: main_cuda.cu
	nvcc main_cuda.cu
	./a.out > img.ppm
