# g++ for the serial baseline in every job (a loaded nvhpc module would set CXX=nvc++);
# override on the command line if needed: make CXX=...
BIN       ?= bin
CXX        = g++
MPICXX    ?= mpicxx
NVCC      ?= nvcc
# A40 (Vanda) is compute capability 8.6; change for other GPUs
CUDA_ARCH ?= sm_86

# serial + CPU MPI programs (no GPU needed)
all: $(BIN)/jacobi_serial $(BIN)/jacobi_mpi

serial: $(BIN)/jacobi_serial
mpi: $(BIN)/jacobi_mpi
gpu: $(BIN)/jacobi_mpi_cuda

$(BIN)/jacobi_serial: src/jacobi_serial.cpp src/common.h
	mkdir -p $(BIN)
	$(CXX) -O3 -std=c++17 -o $@ $<

$(BIN)/jacobi_mpi: src/jacobi_mpi.cpp src/common.h
	mkdir -p $(BIN)
	$(MPICXX) -O3 -std=c++17 -o $@ $<

# MPI+CUDA: nvcc with the MPI compiler wrapper as host compiler
$(BIN)/jacobi_mpi_cuda: src/jacobi_mpi_cuda.cu src/common.h
	mkdir -p $(BIN)
	$(NVCC) -O3 -std=c++17 -arch=$(CUDA_ARCH) -ccbin $(MPICXX) -o $@ $<

# fallback if the wrapper cannot be used as host compiler (e.g. it wraps nvc++):
# plain g++ as host compiler, MPI flags taken from the wrapper
gpu-alt:
	mkdir -p $(BIN)
	$(NVCC) -O3 -std=c++17 -arch=$(CUDA_ARCH) -ccbin g++ \
	    $$($(MPICXX) --showme:compile) -o $(BIN)/jacobi_mpi_cuda src/jacobi_mpi_cuda.cu \
	    $$($(MPICXX) --showme:link | sed 's/-Wl,/-Xlinker /g')

clean:
	rm -r $(BIN)

.PHONY: all serial mpi gpu gpu-alt clean
