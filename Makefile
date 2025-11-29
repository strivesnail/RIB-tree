# Makefile for RIB-tree project
# Supports both debug and release builds

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -march=native
LDFLAGS = -fopenmp -lpthread

# Debug flags
DEBUG_FLAGS = -g -O0 -DDEBUG -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer

# Release flags
RELEASE_FLAGS = -O3 -DNDEBUG -march=native -mtune=native

# Directories
SRC_DIR = .
BUILD_DIR = build
DEBUG_DIR = $(BUILD_DIR)/debug
RELEASE_DIR = $(BUILD_DIR)/release

# Source files
BENCHMARK_SRC = benchmark.cpp

# Header files (dependencies)
HEADERS = btreeolc.h rib_segment.h lib_utils.h config_generator/segmentation.h

# Targets
DEBUG_BENCHMARK = $(DEBUG_DIR)/benchmark_btreeolc
RELEASE_BENCHMARK = $(RELEASE_DIR)/benchmark_btreeolc

# Default target
.PHONY: all
all: release

# Debug targets
.PHONY: debug
debug: $(DEBUG_BENCHMARK)

# Release targets
.PHONY: release
release: $(RELEASE_BENCHMARK)

# Create build directories
$(DEBUG_DIR):
	@mkdir -p $(DEBUG_DIR)

$(RELEASE_DIR):
	@mkdir -p $(RELEASE_DIR)

# Debug build rules
$(DEBUG_BENCHMARK): $(BENCHMARK_SRC) $(HEADERS) | $(DEBUG_DIR)
	@echo "Building DEBUG version: $@"
	$(CXX) $(CXXFLAGS) $(DEBUG_FLAGS) -o $@ $(BENCHMARK_SRC) $(LDFLAGS)
	@echo "Debug build complete: $@"

# Release build rules
$(RELEASE_BENCHMARK): $(BENCHMARK_SRC) $(HEADERS) | $(RELEASE_DIR)
	@echo "Building RELEASE version: $@"
	$(CXX) $(CXXFLAGS) $(RELEASE_FLAGS) -o $@ $(BENCHMARK_SRC) $(LDFLAGS)
	@echo "Release build complete: $@"

# Convenience targets (symlinks in root directory)
.PHONY: install-debug install-release install
install-debug: debug
	@echo "Creating symlink for debug build..."
	@ln -sf $(DEBUG_BENCHMARK) benchmark_btreeolc_debug
	@echo "Debug symlink created"

install-release: release
	@echo "Creating symlink for release build..."
	@ln -sf $(RELEASE_BENCHMARK) benchmark_btreeolc
	@echo "Release symlink created"

install: install-release

# Clean targets
.PHONY: clean clean-debug clean-release
clean: clean-debug clean-release
	@rm -rf $(BUILD_DIR)
	@rm -f benchmark_btreeolc benchmark_btreeolc_debug
	@echo "Cleaned all build files"

clean-debug:
	@rm -rf $(DEBUG_DIR)
	@rm -f benchmark_btreeolc_debug
	@echo "Cleaned debug build files"

clean-release:
	@rm -rf $(RELEASE_DIR)
	@rm -f benchmark_btreeolc
	@echo "Cleaned release build files"

# Help target
.PHONY: help
help:
	@echo "RIB-tree Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make debug          - Build debug version of benchmark_btreeolc (with -g -O0 -fsanitize)"
	@echo "  make release        - Build release version of benchmark_btreeolc (with -O3)"
	@echo "  make all            - Build release version (default)"
	@echo "  make install-debug  - Build debug and create symlink"
	@echo "  make install-release - Build release and create symlink"
	@echo "  make install        - Build release and create symlink (default)"
	@echo "  make clean          - Remove all build files"
	@echo "  make clean-debug    - Remove debug build files"
	@echo "  make clean-release  - Remove release build files"
	@echo "  make help           - Show this help message"
	@echo ""
	@echo "Debug build features:"
	@echo "  - Debug symbols (-g)"
	@echo "  - No optimization (-O0)"
	@echo "  - Address sanitizer (-fsanitize=address)"
	@echo "  - Undefined behavior sanitizer (-fsanitize=undefined)"
	@echo "  - Frame pointers (-fno-omit-frame-pointer)"
	@echo ""
	@echo "Release build features:"
	@echo "  - Full optimization (-O3)"
	@echo "  - Native architecture tuning"
	@echo "  - NDEBUG defined"

