# 🚀 GPU-Accelerated Deduplication + LZ4 Compressor

## The High-Performance Archival Compressor

This project is a **high-performance archival compression solution** that uses **GPU acceleration** to perform deduplication and LZ4 compression. Unlike traditional tools, it is designed to handle large volumes of data, such as *extensive directory backups*, offering a significantly higher compression throughput by leveraging the power of commodity Graphics Processing Units (GPUs).

High-performance data compression and archival system leveraging GPU acceleration via OpenCL. It features a custom, highly optimized compression kernel and a multi-stage deduplication pipeline to achieve superior speed and efficiency.

## ✨ Key Features

The system is engineered for maximum throughput and efficiency, focusing on several critical optimizations:

### 1. Custom LZ4 Kernel with Extended Window (The Custom Kernel)

The core compression logic is implemented in a custom OpenCL kernel (`lz4_compress_ext3.cl`). This kernel significantly enhances the standard LZ4 algorithm by utilizing **3-byte offsets**, which extends the maximum match distance (window size) to **16 MB** (compared to the standard LZ4's 64 KB limit). This modification drastically improves the compression ratio for large, repetitive data blocks, making it ideal for archival and backup scenarios.

### 2. Parallel GPU Hashing for Deduplication

File deduplication is accelerated by calculating FNV-1a 64-bit hashes in parallel across multiple GPUs using OpenCL. The system employs a **round-robin strategy** to distribute the hashing workload, ensuring maximum throughput and minimizing CPU overhead during the most I/O-intensive phase.

### 3. Top-K Match Search Optimization

Within the custom LZ4 kernel, the match-finding process uses a **Top-K adaptive search** (specifically, 8 candidates) in the hash table. This technique prioritizes finding the *best* possible match rather than just the first one, further enhancing the compression ratio without sacrificing performance due to the GPU's parallel processing capabilities.

### 4. Extremely Efficient Multi-Stage Deduplication

To minimize I/O and expensive full-file hash calculations, the deduplication process uses a highly efficient **four-stage filter** before performing the full GPU hash:
1.  **Size Check:** Group files by size.
2.  **First 2 Bytes Check:** Filter groups by the first two bytes.
3.  **Last 2 Bytes Check:** Filter remaining groups by the last two bytes.
4.  **Center 3 Bytes Check:** Filter by three bytes around the center of the file.

Only files that pass all these quick checks proceed to the final, full-file GPU hash calculation, making the deduplication process exceptionally fast.

### 5. Adaptive Compression Fallback (Read-Ahead Skip Optimization)

The system incorporates an optimization for handling incompressible data. While the compression is performed on the GPU, if the resulting compressed frame size is not smaller than the original data size, the system automatically falls back to storing the data in its **raw (uncompressed) format**. This prevents wasting time and space on data that cannot be effectively compressed, serving as an effective "read-ahead skip" mechanism for impossible-to-compress bytes.

### 6. Embedded Metadata Indexing

The system generates a comprehensive index of all files, frames, and compression parameters. This index is then **compressed using zlib** and **embedded directly into the final volume file**. A fixed 24-byte footer (`GPU_IDX1`) is appended to the last volume, allowing a metadata reader to quickly locate and extract the index without relying on external index files.

## ⚙️ How It Works (Architecture Overview)

-┌─────────────────────────────────────────────────────────────────────┐
-│                    compressor_lz4_dedup.py (Main)                   │
-├─────────────────────────────────────────────────────────────────────┤
-│  1. Scan Directory (iotools.scan_directory)                         │
-│  2. Deduplicação GPU (deduplicator.GPUFileDeduplicator)             │
-│  3. Pipeline I/O (Triple Buffer)                                    │
-│     ├─ Reader Thread → [Frame Queue] →                              │
-│     ├─ Compress Workers → [Write Queue] →                           │
-│     └─ Writer Thread                                                │
-│  4. Embed Index (iotools.embed_index_file)                          │
-└─────────────────────────────────────────────────────────────────────┘
-                              │
-        ┌─────────────────────┴─────────────────────┐
-        ▼                                           ▼
-┌───────────────────────┐               ┌───────────────────────┐
-│  GPU_LZ4_Compressor   │               │  CPU_LZ4_Compressor   │
-│  (gpu_lz4_compressor) │               │  (compressor_fallback)│
-├───────────────────────┤               ├───────────────────────┤
-│ • OpenCL Kernel       │               │ • Pure Python         │
-│ • HASH_LOG=20 (1M)    │               │ • ThreadPoolExecutor  │
-│ • 7 Candidates/Hash   │               │ • cpu_count() workers │
-│ • 3-byte offsets      │               │ • 3-byte offsets      │
-│ • 16MB window         │               │ • 16MB window         │
-│ • Batch processing    │               │ • Batch processing    │
-│ • Multi-GPU support   │               │ • 100% CPU usage      │
-└───────────────────────┘               └───────────────────────┘

The compression process follows a streamlined, high-throughput pipeline:

1.  **Directory Scan:** Recursively scan the source directory to create a list of `FileEntry` objects.
2.  **Deduplication:** Apply the four-stage filter, followed by parallel GPU hashing to mark duplicate files.
3.  **Frame Generation:** Unique files are concatenated into a single stream, which is then split into fixed-size frames (e.g., 16 MB).
4.  **GPU Compression:** Frames are processed in parallel batches by multiple GPU workers using the custom LZ4 kernel.
5.  **Volume Writing:** Compressed frames are written sequentially to multi-part volumes (`.001`, `.002`, etc.), respecting a maximum volume size.
6.  **Index Embedding:** The final index is compressed and embedded into the last volume file.

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

## Real Console Output log Compress / Extract (Sample)



E:\TheStorage\runtime\compressor>python compressor_lz4_dedup.py E:\tmp -o E:\teste.gpu

======================================================================

COMPRESSOR LZ4 GPU + DEDUP

======================================================================

Fonte: E:\tmp

Saída: E:\teste.gpu

======================================================================

Arquivos: 16496 (1292817530 bytes)


[Fase 1] Deduplicação GPU...

[Deduplicator] GPU Hashing ativado: NVIDIA GeForce GTX 1050 Ti

[Deduplicator] Iniciando busca por duplicatas em 16496 arquivos...

[Dedup Stage 1] Size Filter: 5176 grupos únicos. 13921 arquivos candidatos a duplicata.

[Dedup Stage 2] First 2 Bytes: 2283 removidos. 11638 restantes.

[Dedup Stage 3] Last 2 Bytes:  933 removidos. 10705 restantes.

[Dedup Stage 4] Center 3 Bytes: 4841 removidos. 5864 restantes para Hash Completo.

[Dedup Final] Encontradas 1532 duplicatas reais.

[Dedup Final] Economia potencial: 50.35 MB

[Dedup Final] Encontradas 1532 duplicatas reais.

[Dedup Final] Economia potencial: 50.35 MB

Tamanho efetivo: 1240023174 bytes

Economia Dedup: 50.35 MB (4.1%)


[Fase 2] Compressão LZ4 GPU...

[Compressor] Parâmetros: frame_size=16MB, max_volume=98MB

[Compressor] Detectadas 1 GPUs. Inicializando compressores...

[GPU_LZ4] Compressor LZ4 OpenCL ativado em: NVIDIA GeForce GTX 1050 Ti (Index: 0)

[GPU_LZ4] Buffers persistentes alocados: Input=16.0MB, Output=16.1MB, Hash=32768.0KB

[Compressor] Iniciando compressão com 1 workers GPU...

[LZ4] Frame 0 GPU1: 16777216 -> 12613956 (75.2%) | Economia: 4163260

[VolumeWriter] Abrindo novo volume: E:\teste.gpu.001

[VolumeWriter] Abrindo novo volume: E:\teste.gpu.002 | 519.9 MB/s

[Compressor] | LZ_EXT3_GPU=5 (62.50%) | RAW=3 (37.50%) | Redução = 62.5%

HIT: GPU1 = 8 | Tamanho real = 128MB | Tamanho atual = 48MB

Last Frame - GPU1 = 7 | Progresso atual = 10.8%

[VolumeWriter] Abrindo novo volume: E:\teste.gpu.003 | 1844.7 MB/s

[Compressor] | LZ_EXT3_GPU=9 (64.29%) | RAW=5 (35.71%) | Redução = 64.3%

HIT: GPU1 = 14 | Tamanho real = 224MB | Tamanho atual = 80MB

Last Frame - GPU1 = 13 | Progresso atual = 18.9%

[VolumeWriter] Abrindo novo volume: E:\teste.gpu.011 | 3063.4 MB/s

[Compressor] | LZ_EXT3_GPU=22 (34.38%) | RAW=42 (65.62%) | Redução = 34.4%

HIT: GPU1 = 64 | Tamanho real = 1024MB | Tamanho atual = 672MB

Last Frame - GPU1 = 63 | Progresso atual = 86.6%

[Compressor] | LZ_EXT3_GPU=32 (43.24%) | RAW=42 (56.76%) | Redução = 43.2%

HIT: GPU1 = 74 | Tamanho real = 1183MB | Tamanho atual = 672MB

Last Frame - GPU1 = 73 | Progresso atual = 100.0%

[Index] Incorporando índice comprimido (148414 bytes) em teste.gpu.011...

[Index] Footer gravado. Offset=89272883, Size=148414


======================================================================

✅ Processo Completo Finalizado!

======================================================================






E:\TheStorage\runtime\compressor>python decompressor_lz4.py E:\teste.gpu.001 -o e:\teste

Volumes encontrados: 11. Último: teste.gpu.011

Índice encontrado: Offset=89272883, Size=148414

Índice carregado e descomprimido com sucesso.

Restaurando 16496 arquivos para E:\teste...

[GPU_LZ4] Decompressor OpenCL initialized on NVIDIA GeForce GTX 1050 Ti (Index: 0)


[GPU_LZ4] GPU Decompressors initialized: 1

[GPU_LZ4] Processing 74 frames in 4 batches (size=24)

[GPU_LZ4] Worker threads: 2


Progresso: 74/74 frames | 12.0 MB/s | GPU: 32 | CPU: 42 | Fila: 0

Criando 1532 arquivos duplicados (dedup reverso)...

[Duplicatas] 1532/1532 arquivos duplicados recriados.



============================================================

Descompressão concluída!

============================================================

Tempo total:      100.0s

Frames totais:    74

  - GPU:          32 (43.2%)
  - 
  - CPU:          42 (56.8%)
  - 
Dados escritos:   1182.6 MB

Velocidade média: 11.8 MB/s

============================================================
