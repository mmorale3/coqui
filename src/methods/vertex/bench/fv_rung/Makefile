# Factorized-vertex rung miniapp (needs a C++17 compiler with OpenMP, HDF5 and MKL: set HDF5_ROOT and MKLROOT)
CXX ?= g++
CXXFLAGS = -O3 -march=native -fopenmp -std=c++17
INC = -I$(HDF5_ROOT)/include -I$(MKLROOT)/include
LIBS = -L$(HDF5_ROOT)/lib -lhdf5 -L$(MKLROOT)/lib -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
fv_rung: fv_rung.cpp fv_common.hpp
	$(CXX) $(CXXFLAGS) $(INC) $< -o $@ $(LIBS)
# GPU build (additionally needs CUDA and cuBLAS)
fv_rung_gpu: fv_rung_gpu.cu fv_common.hpp
	nvcc -O3 -std=c++17 -arch=$(ARCH) -Xcompiler -fopenmp $(INC) $< -o $@ $(LIBS) -lcublas
ARCH ?= sm_80
