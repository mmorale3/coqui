# factorize_vertex rung miniapp (rusty: module load gcc/13.3.0 hdf5/1.14.5 intel-oneapi-mkl/2024.2.2)
CXX ?= g++
CXXFLAGS = -O3 -march=native -fopenmp -std=c++17
INC = -I$(HDF5_ROOT)/include -I$(MKLROOT)/include
LIBS = -L$(HDF5_ROOT)/lib -lhdf5 -L$(MKLROOT)/lib -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
fv_rung: fv_rung.cpp
	$(CXX) $(CXXFLAGS) $(INC) $< -o $@ $(LIBS)
