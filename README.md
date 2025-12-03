# 🚀 GPU-Accelerated Deduplication + LZ4 Compressor

## The High-Performance Archival Compressor

This project is a **high-performance archival compression solution** that uses **GPU acceleration** to perform deduplication and LZ4 compression. Unlike traditional tools, it is designed to handle large volumes of data, such as *extensive directory backups*, offering a significantly higher compression throughput by leveraging the power of commodity Graphics Processing Units (GPUs).

The project implements a complete **custom archival format**, including its own compressor, decompressor, metadata index, and a hybrid (CPU/GPU) pipeline to ensure maximum performance and compatibility.

## ✨ Key Features

What makes this compressor unique and extremely efficient:

| Feature | Description | Primary Benefit |
| :--- | :--- | :--- |
| **GPU-Accelerated Deduplication** | Multi-stage filter pipeline (size, first/last/center bytes) culminating in 64-bit hash validation by GPU (FNV-1a). | **Drastic reduction in I/O and final size** before compression, detecting duplicates with zero false positives. |
| **GPU LZ4 Compressor (Extended Window)** | Custom LZ4 implementation running entirely on the GPU, with a **16 MB** sliding window (larger than standard) and 3-byte offset encoding. | **Compression speed of 2-3+ GB/s** (depending on the GPU), optimized for high throughput. |
| **Frame-Based Streaming Compressor** | Concatenates files into a continuous byte stream, split into 16 MB frames, processed **independently and in parallel** (supports multi-GPU). | **Highly parallel** and efficient processing for large files. |
| **Multi-Volume Output** | The final archive is split into smaller volumes (default: 98 MB), such as `.001`, `.002`, etc. | Facilitates distribution, compatibility with legacy file systems (e.g., FAT32), and allows **resumable** processing. |
| **Hybrid Decompressor (GPU/CPU)** | Attempts GPU decompression first and, in case of incompatible hardware or error, performs an **automatic fallback to CPU**, ensuring integrity and compatibility. | **Guarantees correctness** on older hardware or in case of GPU kernel corruption. |

## ⚙️ How It Works (Architecture Overview)

The compression process is divided into stages optimized for GPU parallelism:

1.  **Directory Scan:** Collects metadata, preserves relative paths, and detects empty files.
2.  **Deduplication Pipeline:** Fast filters (size, bytes) + GPU hash for remaining candidates. Duplicate files are removed before compression.
3.  **Frame Generation:** Creation of sequential 16 MB frames from non-duplicate file data.
4.  **GPU LZ4 Compression:** Each frame is compressed on the GPU using the extended-window LZ4.
5.  **Multi-Volume Writing:** Frames are inserted into volumes, respecting the maximum size.
6.  **Indexing:** The final volume receives a compressed index (zlib) containing metadata, deduplication map, and frame descriptors.
7.  **Decompression:** Reverse process with GPU or CPU, followed by the reconstruction of deduplicated files (restores originals and creates duplicates via hardlink/copy).

## 📊 Performance and Results

The use of the GPU provides notable performance gains:

*   **Deduplication:** GPU hashing drastically reduces CPU overhead for large folders.
*   **Compression Speed:** GPU LZ4 can reach **2–3+ GB/s**, depending on the card.
*   **Total Reduction:** The combination of Deduplication + LZ4 typically results in a **total reduction of 60–85%** in file size.

> **Example Output (Real Production Log):**
>
> ```
> Dedup Final: Found 129 duplicates (586.13 MB saved)
> Remaining size: 1.01 GB
> 
> GPU LZ4 Compression:
> LZ_EXT3_GPU=46 (75.41%) | RAW=15 (24.59%)
> Final: 971 MB → 240 MB (75.3% reduction)
> ```

## 🛠️ Requirements

To run the compressor, you will need:

*   **Python 3.9+**
*   **PyOpenCL**
*   **Numpy**
*   **LZ4**
*   **Zlib** (usually built-in with Python)
*   **Any CUDA/OpenCL compatible GPU**
    *   *Recommended:* NVIDIA GTX 1050 Ti or better.

## 🚀 Usage

### 1. Installation

```bash
# Clone the repository
git clone [YOUR_REPOSITORY_URL]
cd [REPOSITORY_NAME]

# Install dependencies
pip install pyopencl numpy lz4
```

### 2. Compress a Directory

Use the main script `compressor_lz4_dedup.py`:

```bash
python compressor_lz4_dedup.py <source_folder> -o <output_file_name>
```

*Example:*
```bash
python compressor_lz4_dedup.py /home/user/my_files -o backup_2025
# This will create files like backup_2025.001, backup_2025.002, etc.
```

### 3. Decompress an Archive

Use the `decompressor_lz4.py` script and point to the first volume (`.001`):

```bash
python decompressor_lz4.py <output_file.001> -o <destination_folder>
```

*Example:*
```bash
python decompressor_lz4.py backup_2025.001 -o /home/user/restoration
```

## 🗺️ Roadmap

The project is constantly evolving. Future plans include:

*   Multi-GPU scaling.
*   Streaming Compressor (`stdin`/`stdout` mode).
*   Optional Zstd-GPU backend.
*   Adaptive window sizes.
*   Repair tool for missing volumes.

## 🤝 Contributing

Contributions, pull requests, issue reports, and suggestions are highly welcome! This is an experimental, research-oriented project, and your help is essential for continuous improvement.

---

