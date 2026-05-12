CXX = g++
NVCC = nvcc

CXXFLAGS = -O3 -Wall -std=c++17
NVCCFLAGS = -O3 -arch=sm_60 -std=c++17

INCLUDES = -I./include
SRC_DIR = src

# Object files
OBJS = $(SRC_DIR)/fused_layernorm.o \
       $(SRC_DIR)/flash_attention.o \
       $(SRC_DIR)/linear_layers.o \
       $(SRC_DIR)/cpu_reference.o \
       $(SRC_DIR)/main.o

TARGET = gpt2_benchmark

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(SRC_DIR)/*.o $(TARGET)
