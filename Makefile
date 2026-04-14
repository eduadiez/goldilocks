TARGET_EXEC := example

BUILD_DIR := ./build
BUILD_DIR_GPU := ./build_gpu
SRC_DIRS := ./src
TEST_DIRS := ./test

-include CudaArch.mk

### Detect OS and architecture
KERNEL := $(shell uname -s)
ARCH   := $(shell uname -m)

### Detect Homebrew prefix (macOS only) and locate libomp
ifeq ($(KERNEL),Darwin)
  HOMEBREW_PREFIX := $(shell brew --prefix 2>/dev/null)
  LIBOMP_PREFIX   := $(shell brew --prefix libomp 2>/dev/null)
  LIBOMP_INC      := $(LIBOMP_PREFIX)/include
  LIBOMP_LIB      := $(LIBOMP_PREFIX)/lib
  LIBOMP_FLAGS    := -Xpreprocessor -fopenmp -I$(LIBOMP_INC) -L$(LIBOMP_LIB) -lomp
else
  LIBOMP := $(shell find /usr/lib/llvm-* -name "libomp.so" 2>/dev/null | sed 's/libomp.so//')
  ifndef LIBOMP
    $(error LIBOMP is not set, you need to install libomp-dev)
  endif
  LIBOMP_FLAGS := -L$(LIBOMP) -fopenmp
endif

### GMP and GTest locations
ifeq ($(KERNEL),Darwin)
  GMP_PREFIX   := $(shell brew --prefix gmp 2>/dev/null)
  GTEST_PREFIX := $(shell brew --prefix googletest 2>/dev/null)
  BENCH_PREFIX := $(shell brew --prefix google-benchmark 2>/dev/null)
  GMP_FLAGS    := -I$(GMP_PREFIX)/include -L$(GMP_PREFIX)/lib
  GTEST_FLAGS  := -I$(GTEST_PREFIX)/include -L$(GTEST_PREFIX)/lib
  BENCH_FLAGS  := -I$(BENCH_PREFIX)/include -L$(BENCH_PREFIX)/lib
else
  GMP_FLAGS    :=
  GTEST_FLAGS  :=
  BENCH_FLAGS  :=
endif

### Architecture-specific SIMD flags
ifeq ($(ARCH),x86_64)
  SIMD_FLAGS := -mavx2
else
  SIMD_FLAGS :=
endif

### Compiler selection
ifeq ($(KERNEL),Darwin)
  CXX := clang++
  CC  := clang
else
  CXX := g++
  CC  := gcc
endif

CXXFLAGS := -std=c++17 -Wall -pthread $(LIBOMP_FLAGS)
LDFLAGS  := -lpthread $(GMP_FLAGS) -lgmp -lstdc++ -lgmpxx -lbenchmark
ASFLAGS := -felf64

NVCC := /usr/local/cuda/bin/nvcc

# Debug build flags
ifeq ($(dbg),1)
      CXXFLAGS += -g
else
      CXXFLAGS += -O3
endif

SRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.asm -or -name *.cu)
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)
ALLSRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.asm -or -name *.hpp -or -name *.cu -or -name *.cuh)

INC_DIRS := $(shell find $(SRC_DIRS) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CPPFLAGS ?= $(INC_FLAGS) -MMD -MP $(SIMD_FLAGS)

testcpu: tests/tests.cpp $(ALLSRCS)
	$(CXX) -std=c++17 tests/tests.cpp src/*.cpp $(GTEST_FLAGS) $(GMP_FLAGS) $(LIBOMP_FLAGS) -lgtest -lgmp -O3 -Wall -pthread $(SIMD_FLAGS) -o $@

$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS)
	$(CXX) $(OBJS) $(CXXFLAGS) -o $@ $(LDFLAGS)

# c++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -c $< -o $@

# c++ source with CUDA support
$(BUILD_DIR_GPU)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) -D__USE_CUDA__ $(SIMD_FLAGS) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -c $< -o $@

$(BUILD_DIR_GPU)/%.cu.o: %.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) -D__USE_CUDA__ -DGPU_TIMING -Iutils -Xcompiler -O3 -Xcompiler -fopenmp -Xcompiler -fPIC -Xcompiler $(SIMD_FLAGS) -arch=$(CUDA_ARCH) -dc $< --output-file $@

.PHONY: clean


testgpu: $(BUILD_DIR_GPU)/tests/tests.cpp.o $(BUILD_DIR)/src/goldilocks_base_field.cpp.o $(BUILD_DIR)/src/goldilocks_cubic_extension.cpp.o $(BUILD_DIR)/utils/timer_gl.cpp.o $(BUILD_DIR_GPU)/src/ntt_goldilocks.cpp.o $(BUILD_DIR)/src/poseidon_goldilocks.cpp.o $(BUILD_DIR_GPU)/src/ntt_goldilocks.cu.o $(BUILD_DIR_GPU)/src/poseidon_goldilocks.cu.o $(BUILD_DIR_GPU)/utils/cuda_utils.cu.o
	$(NVCC) -Xcompiler -O3 -Xcompiler -fopenmp -arch=$(CUDA_ARCH) -o $@ $^ -lgtest -lgmp

runtestcpu: testcpu
	./testcpu --gtest_filter=GOLDILOCKS_TEST.merkletree_seq

runtestgpu: testgpu
	./testgpu --gtest_filter=GOLDILOCKS_TEST.merkletree_cuda

full: $(BUILD_DIR_GPU)/tests/tests.cu.o $(BUILD_DIR_GPU)/src/goldilocks_base_field.cpp.o  $(BUILD_DIR_GPU)/utils/timer_gl.cpp.o $(BUILD_DIR_GPU)/utils/cuda_utils.cu.o  $(BUILD_DIR_GPU)/src/ntt_goldilocks.cpp.o $(BUILD_DIR_GPU)/src/poseidon_goldilocks.cpp.o $(BUILD_DIR_GPU)/src/ntt_goldilocks.cu.o $(BUILD_DIR_GPU)/src/poseidon_goldilocks.cu.o
	$(NVCC) -Xcompiler -O3 -Xcompiler -fopenmp -arch=$(CUDA_ARCH) -o $@ $^ -lgtest -lgmp

runfullgpu: full
	./full --gtest_filter=GOLDILOCKS_TEST.full_gpu

runfullcpu: full
	./full --gtest_filter=GOLDILOCKS_TEST.full_cpu

benchcpu: benchs/bench.cpp src/*.cpp
	$(CXX) $^ $(GMP_FLAGS) $(GTEST_FLAGS) $(BENCH_FLAGS) $(LIBOMP_FLAGS) -lbenchmark -lpthread -lgmp -std=c++17 -Wall -pthread $(SIMD_FLAGS) -O3 -o $@

benchgpu: $(BUILD_DIR_GPU)/benchs/bench.cpp.o $(BUILD_DIR)/src/goldilocks_base_field.cpp.o $(BUILD_DIR)/src/goldilocks_cubic_extension.cpp.o $(BUILD_DIR_GPU)/src/poseidon_goldilocks.cpp.o $(BUILD_DIR_GPU)/src/ntt_goldilocks.cu.o $(BUILD_DIR_GPU)/src/poseidon_goldilocks.cu.o
	$(NVCC) -Xcompiler -O3 -Xcompiler -fopenmp -arch=$(CUDA_ARCH) -o $@ $^ -lgtest -lgmp -lbenchmark

runbenchcpu: benchcpu
	./benchcpu --benchmark_filter=MERKLETREE_BENCH_AVX

runbenchgpu: benchgpu
	./benchgpu --benchmark_filter=MERKLETREE_BENCH_CUDA

clean:
	$(RM) -r $(BUILD_DIR)
	$(RM) -r $(BUILD_DIR_GPU)
	$(RM) test
	$(RM) bench
	$(RM) testcpu
	$(RM) testgpu
	$(RM) benchcpu
	$(RM) benchgpu


-include $(DEPS)

MKDIR_P ?= mkdir -p
