# make: build/libmllm.a, build/mllm (CLI), build/tests/*, build/mllm.metallib.  make test: run every test.
CXX := clang++
B := build
WERROR ?=  # CI: make WERROR=-Werror
CXXFLAGS := -std=c++20 -O2 -Wall -Wextra $(WERROR) -MMD -MP -Isrc -isystem src/vendor
METALLIB := $(abspath $(B)/mllm.metallib)
OBJCFLAGS := -fobjc-arc -DMETALLIB='"$(METALLIB)"'
# MLX's own kernel flags (mlx/backend/metal/kernels/CMakeLists.txt): no fast math, or results drift from MLX's
METALFLAGS := -Wall -Wextra $(WERROR) -fno-fast-math -Wno-c++17-extensions -Wno-c++20-extensions -Isrc/vendor/mlx
# floor for clang and the Metal compiler: the kernels' lambdas need Metal on macOS 15
export MACOSX_DEPLOYMENT_TARGET ?= 15.0
LDLIBS := -framework Metal -framework Foundation -framework CoreML -framework CoreVideo -framework IOSurface

LIB_SRC := $(filter-out src/main.mm,$(wildcard src/*.cpp src/*.mm src/vendor/*/*.cpp))
LIB_OBJ := $(LIB_SRC:src/%=$(B)/obj/%.o)
TESTS := $(patsubst tests/%,$(B)/tests/%,$(basename $(wildcard tests/*.cpp tests/*.mm)))
AIR := $(patsubst src/kernels/%.metal,$(B)/air/%.air,$(wildcard src/kernels/*.metal))

# needs the Metal Toolchain (xcodebuild -downloadComponent MetalToolchain)
all: $(B)/libmllm.a $(B)/mllm $(TESTS) $(B)/mllm.metallib

$(B)/obj/%.cpp.o: src/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(B)/obj/%.mm.o: src/%.mm
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(OBJCFLAGS) -c $< -o $@

$(B)/libmllm.a: $(LIB_OBJ)
	@rm -f $@
	ar rcs $@ $^

# the kernels ride inside the binary (GpuOps loads the section), so build/mllm runs from anywhere; tests load the file
$(B)/mllm: $(B)/obj/main.mm.o $(B)/libmllm.a $(B)/mllm.metallib
	$(CXX) $(filter-out %.metallib,$^) $(LDLIBS) -Wl,-sectcreate,__TEXT,__metallib,$(B)/mllm.metallib -o $@

$(B)/tests/%: tests/%.cpp $(B)/libmllm.a
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -Itests $< $(B)/libmllm.a $(LDLIBS) -o $@

$(B)/tests/%: tests/%.mm $(B)/libmllm.a
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(OBJCFLAGS) -Itests $< $(B)/libmllm.a $(LDLIBS) -o $@

$(B)/air/%.air: src/kernels/%.metal
	@mkdir -p $(@D)
	xcrun -sdk macosx metal $(METALFLAGS) -MMD -MP -MF $(@:.air=.d) -c $< -o $@

$(B)/mllm.metallib: $(AIR)
	xcrun -sdk macosx metallib $^ -o $@

test: all
	@for t in $(TESTS); do echo "== $$t"; $$t || exit 1; done

# CI (GitHub-hosted macOS VMs): only tests that need no model, golden data, GPU or ANE
CI_TESTS := $(B)/tests/test_arena $(B)/tests/test_prefix_cache $(B)/tests/test_memory_fuzz
test-ci: all
	@for t in $(CI_TESTS); do echo "== $$t"; $$t || exit 1; done

clean:
	rm -rf $(B)

.PHONY: all test test-ci clean
-include $(wildcard $(B)/obj/*.d $(B)/obj/*/*/*.d $(B)/tests/*.d $(B)/air/*.d)
