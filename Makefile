CXX = g++
NVCC = nvcc
MPICXX ?= mpicxx

CXXFLAGS = -O3 -Wall -std=c++17
NVCCFLAGS ?= -O3 -arch=sm_75 -std=c++17

INCLUDES = -I./include
SRC_DIR = src

COMMON_KERNELS = \
	$(SRC_DIR)/linear_layers.cu \
	$(SRC_DIR)/fused_layernorm.cu \
	$(SRC_DIR)/flash_attention.cu \
	$(SRC_DIR)/backward_kernels.cu

BENCHMARK_SRCS = \
	$(SRC_DIR)/benchmark.cu \
	$(SRC_DIR)/linear_layers.cu \
	$(SRC_DIR)/fused_layernorm.cu \
	$(SRC_DIR)/flash_attention.cu \
	$(SRC_DIR)/cpu_reference.cpp

.PHONY: all benchmark train clean

all: benchmark train comprehensive_benchmark

benchmark: gpt2_benchmark

train: train_step1 train_step2 train_step3 train_step4

gpt2_benchmark: $(BENCHMARK_SRCS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $^ -o $@

comprehensive_benchmark: $(SRC_DIR)/comprehensive_benchmark.cu \
    $(SRC_DIR)/linear_layers.cu \
    $(SRC_DIR)/fused_layernorm.cu \
    $(SRC_DIR)/flash_attention.cu \
    $(SRC_DIR)/backward_kernels.cu \
    $(SRC_DIR)/cpu_reference.cpp
	$(NVCC) $(NVCCFLAGS) -ccbin=$(MPICXX) $(INCLUDES) $^ -o $@

train_step1: $(SRC_DIR)/train_step1.cu $(SRC_DIR)/linear_layers.cu $(SRC_DIR)/backward_kernels.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $^ -o $@

train_step2: $(SRC_DIR)/train_step2.cu $(SRC_DIR)/linear_layers.cu $(SRC_DIR)/fused_layernorm.cu $(SRC_DIR)/backward_kernels.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $^ -o $@

train_step3: $(SRC_DIR)/train_step3.cu $(COMMON_KERNELS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $^ -o $@

train_step4: $(SRC_DIR)/train_step4.cu $(COMMON_KERNELS)
	$(NVCC) $(NVCCFLAGS) -ccbin=$(MPICXX) $(INCLUDES) $^ -o $@

clean:
	rm -f $(SRC_DIR)/*.o gpt2_benchmark comprehensive_benchmark train_step1 train_step2 train_step3 train_step4
