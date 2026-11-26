# metal-cpp on Asahi Linux: libmetal_asahi (Objective-C runtime subset,
# Foundation subset, Metal over the Asahi DRM uAPI, AIR -> AGX through the
# in-tree LLVM AGX backend) and its test suite.
#
#   make                 build build/libmetal_asahi.so
#   make check           run the test suite (METAL_ASAHI_MODE=verify: simulator first, then GPU, results compared)
#   make check-sim       run the test suite on the simulator only (no GPU)
#   make check-hw        run the test suite on the GPU only

LLVM_BUILD ?= /scratch/llvm-build
LLVM_SRC   ?= $(abspath ../../llvm-project)
AGX_RUN    ?= /scratch/agx-sim-target/release/agx_run
BUILD      ?= build

CXX        := g++
CLANGXX    := $(LLVM_BUILD)/bin/clang++

LLVM_CXXFLAGS := -I$(LLVM_BUILD)/include -I$(LLVM_SRC)/llvm/include
LLVM_LIBS     := -L$(LLVM_BUILD)/lib -lLLVM -Wl,-rpath,$(LLVM_BUILD)/lib

OPT      ?= -O2 -g
CXXFLAGS := $(OPT) -std=c++20 -fPIC -Wall -Wno-unused-parameter -Iinclude -fvisibility=hidden \
            -DMETAL_ASAHI_DEFAULT_AGX_RUN='"$(AGX_RUN)"'

SRCS := src/objc.cpp src/foundation.cpp src/drm.cpp src/compiler.cpp src/sim.cpp src/metal_device.cpp src/metal_exec.cpp
OBJS := $(SRCS:src/%.cpp=$(BUILD)/obj/%.o) $(BUILD)/obj/objc_msgsend.o

LIB := $(BUILD)/libmetal_asahi.so

all: $(LIB)

$(BUILD)/obj/%.o: src/%.cpp src/*.h
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(LLVM_CXXFLAGS) -c $< -o $@

# LLVM is built without RTTI.
$(BUILD)/obj/compiler.o: src/compiler.cpp src/*.h
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(LLVM_CXXFLAGS) -fno-rtti -c $< -o $@

$(BUILD)/obj/objc_msgsend.o: src/objc_msgsend.S
	@mkdir -p $(dir $@)
	$(CXX) -c $< -o $@

$(LIB): $(OBJS)
	$(CXX) -shared -o $@ $(OBJS) $(LLVM_LIBS) -Wl,--no-undefined -lpthread -Wl,-soname,libmetal_asahi.so

# metal-cpp programs: clang++ (blocks, constant NSStrings), our compat headers first.
APP_CXXFLAGS := -std=c++20 -O2 -g -fblocks -I$(abspath include) -I$(abspath ..) -I$(abspath tests) -Wno-deprecated-declarations
APP_LIBS     := -L$(abspath $(BUILD)) -lmetal_asahi -Wl,-rpath,$(abspath $(BUILD))

# Kernels: OpenCL C or LLVM IR -> AIR (.air), no MSL (tools/air-build.sh).
KERNEL_SRCS := $(wildcard kernels/*.cl) $(wildcard kernels/*.ll)
KERNELS     := $(patsubst kernels/%,$(BUILD)/kernels/%,$(addsuffix .air,$(basename $(KERNEL_SRCS))))

$(BUILD)/kernels/%.air: kernels/%.cl tools/air-build.sh
	@mkdir -p $(dir $@)
	LLVM_BUILD=$(LLVM_BUILD) sh tools/air-build.sh $< $@

$(BUILD)/kernels/%.air: kernels/%.ll tools/air-build.sh
	@mkdir -p $(dir $@)
	LLVM_BUILD=$(LLVM_BUILD) sh tools/air-build.sh $< $@

TEST_SRCS := $(wildcard tests/*.cpp)
TESTS     := $(TEST_SRCS:tests/%.cpp=$(BUILD)/tests/%)

$(BUILD)/tests/%: tests/%.cpp tests/harness.h $(LIB)
	@mkdir -p $(dir $@)
	$(CLANGXX) $(APP_CXXFLAGS) $< -o $@ $(APP_LIBS)

kernels: $(KERNELS)
tests: $(TESTS) $(KERNELS)

RUN_TESTS = KERNEL_DIR=$(abspath $(BUILD)/kernels) python3 tools/run-tests.py $(TESTS)

check: tests
	METAL_ASAHI_MODE=$${METAL_ASAHI_MODE:-verify} $(RUN_TESTS)

check-sim: tests
	METAL_ASAHI_MODE=sim $(RUN_TESTS)

check-hw: tests
	METAL_ASAHI_MODE=hw $(RUN_TESTS)

clean:
	rm -f $(BUILD)/obj/*.o $(LIB) $(TESTS) $(KERNELS)

.PHONY: all kernels tests check check-sim check-hw clean
.SECONDARY:
