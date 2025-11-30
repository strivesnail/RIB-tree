CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -march=native -Isrc -I.
LDFLAGS = -fopenmp -lpthread

DEBUG_FLAGS = -g -O0 -DDEBUG -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer
RELEASE_FLAGS = -O3 -DNDEBUG -march=native -mtune=native

BENCHMARK_SRC = test/benchmark.cpp
HEADERS = src/btreeolc.h src/rib_segment.h src/rib_utils.h config_generator/segmentation.h

DEBUG_BENCHMARK = build/debug/benchmark_btreeolc
RELEASE_BENCHMARK = build/release/benchmark_btreeolc
PARTITION_OPTIMIZE = config_generator/partition_optimize

.PHONY: all debug release partition_optimize clean help

all: release

debug: $(DEBUG_BENCHMARK)

release: $(RELEASE_BENCHMARK)

partition_optimize: $(PARTITION_OPTIMIZE)

$(DEBUG_BENCHMARK): $(BENCHMARK_SRC) $(HEADERS) | build/debug
	$(CXX) $(CXXFLAGS) $(DEBUG_FLAGS) -o $@ $(BENCHMARK_SRC) $(LDFLAGS)

$(RELEASE_BENCHMARK): $(BENCHMARK_SRC) $(HEADERS) | build/release
	$(CXX) $(CXXFLAGS) $(RELEASE_FLAGS) -o $@ $(BENCHMARK_SRC) $(LDFLAGS)

$(PARTITION_OPTIMIZE): config_generator/partition_optimize.cpp config_generator/segmentation.h src/rib_utils.h
	$(CXX) $(CXXFLAGS) $(RELEASE_FLAGS) -o $@ config_generator/partition_optimize.cpp

build/debug build/release:
	@mkdir -p $@

clean:
	@rm -rf build benchmark_btreeolc benchmark_btreeolc_debug $(PARTITION_OPTIMIZE)

help:
	@echo "Targets: debug, release (default), partition_optimize, clean, help"

