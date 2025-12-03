GPU-Accelerated Deduplication + LZ4 Compressor

A high-performance archival compressor using GPU-accelerated deduplication, GPU LZ4 compression, frame-based chunking, and multi-volume output.
This project implements a full custom archival format with its own compressor, decompressor, metadata index, and hybrid CPU/GPU fallback pipeline.

Designed for large directory backups, fast distribution, and high compression throughput using commodity GPUs.

🚀 Key Features

    1. GPU-Accelerated Deduplication

File-level deduplication with a multi-stage filter pipeline:

Size grouping

First bytes check

Last bytes check

Middle bytes check

Full 64-bit GPU hash validation

GPU hashing using FNV-1a parallel chunked hashing

Detects true duplicates with zero false positives

Reduces I/O and final archive size dramatically before compression

    2. GPU LZ4 Compressor (Extended 16MB Window)

A custom LZ4 implementation running entirely on the GPU:

16 MB sliding window (larger than standard LZ4)

3-byte offset encoding (extended range)

Hash table with multiple candidates per entry

Lazy matching and skip heuristics

Per-frame GPU parallelism

Fallback to raw blocks when compression is not beneficial

    3. Frame-Based Streaming Compressor

Files are concatenated into a continuous byte stream

Stream is split into 16 MB frames

Each frame is compressed independently

Supports multi-GPU processing

Highly parallel batch compressor with worker pools

    4. Multi-Volume Output

Archive is split into multiple volumes (.001, .002, ...)

Default volume size: 98 MB

Enables:

easier distribution

FAT32 compatibility

resumable processing

    5. Embedded Compressed Index

The final volume stores:

File list with metadata

Deduplication map

Frame table (offsets, sizes, compression mode)

Compressor parameters

Archive integrity info

The index is zlib-compressed and appended with a footer:

[offset][size][IDX1_MAGIC]

    6. GPU/CPU Hybrid Decompressor

Tries GPU decompression first

Falls back to CPU on:

unsupported hardware

errors

corrupted GPU kernels

Reconstructs deduplicated files

First restores originals

Then creates duplicates via hardlink or file copy

📦 Project Structure

/compressor_lz4_dedup.py     → Main compressor

/decompressor_lz4.py         → Main decompressor (with GPU + CPU fallback)

/deduplicator.py             → Multi-stage GPU deduplicator

/gpu_lz4_compressor.py       → LZ4 GPU kernel + batch engine

/gpu_lz4_decompressor.py     → LZ4 GPU decompressor

/gpu_capabilities.py         → GPU hardware detection

/decompressor_dedup.py       → Duplicate file reconstruction

/iotools.py                  → File streaming utilities

⚙️ How It Works (Architecture Overview)
    1. Directory Scan

Collect file metadata, preserve relative paths, and detect empty files.

    2. Deduplication Pipeline

Fast filters (size, first/last/center bytes)

GPU hash for remaining candidates

Build deduplication table

Remove duplicates before compression

    3. Frame Generation

Sequential 16 MB frames created from non-duplicate file data.

    4. GPU LZ4 Compression

Each frame is compressed on GPU using extended-window LZ4.

    5. Multi-Volume Writing

Frames are inserted into volumes respecting the maximum size.

    6. Index Embedding

Final volume receives compressed index + footer.

    7. Decompression

Reverse process with GPU or CPU, then deduplication reconstruction.

📈 Example Output (Real Production Log)
Dedup Final: Found 129 duplicates (586.13 MB saved)
Remaining size: 1.01 GB

GPU LZ4 Compression:
LZ_EXT3_GPU=46 (75.41%) | RAW=15 (24.59%)
Final: 971 MB → 240 MB (75.3% reduction)

🧪 Performance Notes

GPU hashing extremely reduces CPU overhead for large folders

GPU LZ4 compression can reach 2–3+ GB/s depending on card

Dedup + LZ4 typically yields 60–85% total reduction

Fallback mechanisms guarantee correctness on older GPUs

🔧 Requirements

Python 3.9+

PyOpenCL

Numpy

LZ4

Zlib (built-in)

Any CUDA/OpenCL compatible GPU

NVIDIA GTX 1050 Ti or better recommended

📘 Usage
Compress a directory

                python compressor_lz4_dedup.py <source_folder> -o <output_name>

Extract

                python decompressor_lz4.py <archive.001> -o <destination_folder>

archive.NNN

Footer (last volume):
uint64  index_offset
uint32  index_size
char[8] GPU_IDX1_MAGIC

Index Content:

Version

File entries (original / duplicates)

Frame descriptors

Compression modes per frame

GPU parameters

Checksums

📍 Project Goals

Experiment with GPU-accelerated data compression

Create a flexible archival format for large backups

Explore large-window LZ4 variants on OpenCL

Achieve high throughput on consumer GPUs

🛣️ Roadmap

 Multi-GPU scaling

 Streaming compressor (stdin/stdout mode)

 Optional Zstd-GPU backend

 Adaptive window sizes

 Error correction and checksums

 Archive info viewer (gpuinfo)

 Repair tool for missing volumes

🤝 Contributing

Pull requests, issues, and suggestions are welcome!
This project is experimental, research-oriented, and constantly improving.
