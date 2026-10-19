# L0 miniapp -- standalone on purpose: it must build on a GPU node without the CoQuí tree.
#   make cpu           reference CPU kernel (OpenMP; add BLAS=1 for MKL zgemm)
#   make gpu           CUDA kernel (nvcc + cuBLAS)
CXX      ?= g++
NVCC     ?= nvcc
CXXFLAGS ?= -O3 -march=native -fopenmp -std=c++17
NVFLAGS  ?= -O3 -std=c++17 -Xcompiler -fopenmp -lineinfo
ARCH     ?= -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90
ifdef BLAS
CXXFLAGS += -DUSE_BLAS
LDLIBS   += $(BLAS_LIBS)
endif

cpu: l0_cpu
gpu: l0_gpu

l0_cpu: l0_cpu.cpp l0_common.hpp
	$(CXX) $(CXXFLAGS) -o $@ l0_cpu.cpp $(LDLIBS)

l0_gpu: l0_gpu.cu l0_common.hpp
	$(NVCC) $(NVFLAGS) $(ARCH) -o $@ l0_gpu.cu -lcublas

clean:
	rm -f l0_cpu l0_gpu
.PHONY: cpu gpu clean
