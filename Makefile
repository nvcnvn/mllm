# make: build/libmllm.a, build/mllm (CLI), build/tests/*, build/mllm.metallib.  make test: run every test.
CXX := clang++
CXXFLAGS := -std=c++20 -O2 -Wall -Wextra -MMD -MP -Isrc -isystem src/vendor
OBJCFLAGS := -fobjc-arc
LDLIBS := -framework Metal -framework Foundation
B := build
export HF_HOME ?= /Volumes/T5/huggingface

LIB_SRC := $(filter-out src/main.cpp,$(wildcard src/*.cpp src/*.mm))
LIB_OBJ := $(LIB_SRC:src/%=$(B)/obj/%.o)
TESTS := $(patsubst tests/%,$(B)/tests/%,$(basename $(wildcard tests/*.cpp tests/*.mm)))
AIR := $(patsubst src/kernels/%.metal,$(B)/air/%.air,$(wildcard src/kernels/*.metal))

METALLIB := $(abspath $(B)/mllm.metallib)

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

$(B)/mllm: $(B)/obj/main.cpp.o $(B)/libmllm.a
	$(CXX) $^ $(LDLIBS) -o $@

$(B)/tests/%: tests/%.cpp $(B)/libmllm.a
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -Itests $< $(B)/libmllm.a $(LDLIBS) -o $@

$(B)/tests/%: tests/%.mm $(B)/libmllm.a
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(OBJCFLAGS) -Itests -DMETALLIB='"$(METALLIB)"' $< $(B)/libmllm.a $(LDLIBS) -o $@

$(B)/air/%.air: src/kernels/%.metal
	@mkdir -p $(@D)
	xcrun -sdk macosx metal -Wall -MMD -MP -MF $(@:.air=.d) -c $< -o $@

$(B)/mllm.metallib: $(AIR)
	xcrun -sdk macosx metallib $^ -o $@

test: all
	@for t in $(TESTS); do echo "== $$t"; $$t || exit 1; done

clean:
	rm -rf $(B)

.PHONY: all test clean
-include $(wildcard $(B)/obj/*.d $(B)/tests/*.d $(B)/air/*.d)
