# RIB-tree

A high-performance B+ tree implementation with optimistic locking concurrency control and segment-based data organization.

## Overview

RIB-tree is a concurrent B+ tree data structure designed for high-throughput key-value operations. It features:

- **Optimistic Locking**: Lock-free read operations with version-based concurrency control
- **Segment-based Storage**: Data organized into segments for efficient range queries
- **Dynamic Splitting**: Automatic node splitting when thresholds are reached
- **Bulk Loading**: Efficient bulk insertion with configurable segment boundaries
- **Promoted Segments**: Top segments cached at root level for faster access

## Project Structure

```
RIB-tree/
├── src/                    # Core source code
│   ├── btreeolc.h         # B+ tree implementation with optimistic locking
│   ├── rib_segment.h      # Segment implementation
│   └── rib_utils.h        # Utility functions for segments
├── test/                   # Test and benchmark code
│   └── benchmark.cpp      # Benchmark program
├── config_generator/      # Configuration generation tools
│   ├── partition_optimize.cpp  # Generate segment configuration files
│   └── segmentation.h     # Segmentation algorithms
└── Makefile              # Build configuration
```

## Building

### Requirements

- C++20 compatible compiler (GCC 9+ or Clang 10+)
- OpenMP support
- pthread library

### Build Commands

```bash
# Build release version (default)
make release

# Build debug version (with sanitizers)
make debug

# Clean build files
make clean

# Show help
make help
```

The build outputs are placed in `build/release/` and `build/debug/` directories.

## Configuration File Format

Segment configuration files are CSV format with the following structure:

```
lower,upper,box_range,keys_count
```

- `lower`: Lower bound of the segment key range
- `upper`: Upper bound of the segment key range (exclusive)
- `box_range`: Box range for the segment
- `keys_count`: Number of keys in this segment range

Example:
```
-1800000000,-1798828480,33472,1642
-1798828480,-1798378345,12861,1596
```

## Generating Configuration Files

Use the `partition_optimize` tool to generate segment configuration files from input data:

```bash
# Build the tool
make partition_optimize

# Run the tool
./config_generator/partition_optimize <input_file> <output_file>
```

Example:
```bash
make partition_optimize
./config_generator/partition_optimize w106.csv configs/segments_w106.csv
```

The tool processes input data and generates a CSV configuration file with segment boundaries, box ranges, and key counts.

## Running Benchmarks

The benchmark program supports various operations and configurations:

```bash
./build/release/benchmark_btreeolc \
  --keys_file=<path> \
  --keys_file_type=text \
  --config_file_path=<config_file> \
  --init_num_keys=<number> \
  --total_num_keys=<number> \
  --thread_num=<number> \
  --insert_frac=<fraction> \
  --batch_size=<size>
```

### Key Parameters

- `keys_file`: Path to input data file (optional, generates test data if empty)
- `keys_file_type`: Data format (`text` or `binary`)
- `config_file_path`: Segment configuration file path
- `init_num_keys`: Number of keys for initial bulk load
- `total_num_keys`: Total number of keys to process
- `thread_num`: Number of threads for parallel operations
- `insert_frac`: Fraction of operations that are inserts
- `batch_size`: Batch size for batch operations

### Example

```bash
./build/release/benchmark_btreeolc \
  --keys_file=/path/to/data.txt \
  --keys_file_type=text \
  --config_file_path=configs/segment_longitudes-200.csv \
  --init_num_keys=100000000 \
  --total_num_keys=200000000 \
  --thread_num=8 \
  --insert_frac=0.5
```

## Features

### Node Splitting

Nodes automatically split when they reach `SPLIT_THRESHOLD` (default: 32 entries). This applies to both leaf nodes and inner nodes.

### Bulk Loading

Bulk loading mode builds the tree structure first, then inserts data in parallel:
- 16 segments per leaf node
- 16 children per inner node
- No splits during bulk load phase

### Promoted Segments

Top segments (by key count) can be promoted to root level for faster access. The default maximum is 32 promoted segments.

## Configuration

### Split Threshold

The split threshold can be configured by defining `SPLIT_THRESHOLD` before including `btreeolc.h`:

```cpp
#define SPLIT_THRESHOLD 32
#include "src/btreeolc.h"
```

### Maximum Promoted Segments

The maximum number of promoted segments can be configured:

```cpp
#define MAX_PROMOTED_SEGMENTS 32
```

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

