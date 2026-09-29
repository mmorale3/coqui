# factorize_vertex rung miniapp (rusty: module load gcc/13.3.0 hdf5/1.14.5 intel-oneapi-mkl/2024.2.2)
CXX ?= g++
CXXFLAGS = -O3 -march=native -fopenmp -std=c++17
INC = -I$(HDF5_ROOT)/include -I$(MKLROOT)/include
LIBS = -L$(HDF5_ROOT)/lib -lhdf5 -L$(MKLROOT)/lib -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
fv_rung: fv_rung.cpp fv_common.hpp
	$(CXX) $(CXXFLAGS) $(INC) $< -o $@ $(LIBS)
# GPU (rusty: + module load cuda/12.5.1)
fv_rung_gpu: fv_rung_gpu.cu fv_common.hpp
	nvcc -O3 -std=c++17 -arch=$(ARCH) -Xcompiler -fopenmp $(INC) $< -o $@ $(LIBS) -lcublas
ARCH ?= sm_80
