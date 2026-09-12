# PAQ9-CUDA

**PAQ9-CUDA** is an experimental CUDA implementation of the **PAQ9** data-compression algorithm. It moves the PAQ9 prediction and range-coding workload to the NVIDIA GPU and processes independent input chunks in parallel using CUDA threads.

The project is intended for studying and experimenting with GPU acceleration of high-ratio, context-based compression rather than for production archiving.

## Features

- CUDA implementation of PAQ9-style probabilistic compression.
- GPU-side **LZP (Lempel-Ziv Prediction)** for byte prediction.
- Context-based bit prediction with state maps, mixers, and adaptive probability maps.
- Squash/stretch probability transforms.
- Range/arithmetic-style coding for the compressed bitstream.
- Independent file chunks can be compressed/decompressed concurrently on the GPU.
- Automatic selection of the PAQ memory level from the selected chunk size.
- Runtime inspection of available CUDA device memory and CUDA device-heap capacity in the newer implementation.
- Custom `PAQ9-CUDA` archive header storing the original filename and compression configuration.
- Compression/decompression timing and throughput printed to the console.

## How it works

The implementation divides the input file into chunks. Each chunk is assigned to a CUDA thread, where a private PAQ9 model is created for that thread.

```text
Input file
    │
    ├── Chunk 0 ──► CUDA thread 0 ──► PAQ9 model ──► compressed chunk
    ├── Chunk 1 ──► CUDA thread 1 ──► PAQ9 model ──► compressed chunk
    ├── Chunk 2 ──► CUDA thread 2 ──► PAQ9 model ──► compressed chunk
    │
    └── ...

                 NVIDIA GPU
```

Each PAQ9 model contains components including:

- LZP prediction and context hashing
- state tables / state maps
- context mixers
- adaptive probability maps (APMs)
- squash/stretch transforms
- range coding

The GPU code uses device-side dynamic allocation for model data such as context tables, LZP buffers, and hash tables. The host code manages file I/O, CUDA buffers, chunk scheduling, and archive construction.

## Requirements

### Hardware

- An **NVIDIA GPU with CUDA support**
- Sufficient GPU memory for the selected chunk size and PAQ model
- The implementation is memory intensive; larger PAQ memory levels require substantially more GPU memory.

### Software

- NVIDIA CUDA Toolkit
- `nvcc`
- A C++17-capable host compiler supported by your CUDA installation

Check that CUDA is available:

```bash
nvcc --version
nvidia-smi
```

> The program requires an NVIDIA CUDA-capable GPU. It does not provide a CPU-only fallback.

## Build

The repository does not require CMake. Compile the CUDA source directly with `nvcc`.

Linux example:

```bash
nvcc -O3 -std=c++17 -arch=sm_XX -o paq9-cuda PAQ9/paq9-cuda.cu
```

Replace `sm_XX` with the compute capability supported by your NVIDIA GPU. You can also omit `-arch=sm_XX` and let your CUDA toolchain use its default architecture, although explicitly selecting the appropriate architecture is recommended for performance.

Windows example:

```bat
nvcc -O3 -std=c++17 -arch=sm_XX -o paq9-cuda.exe PAQ9\paq9-cuda.cu
```

The source file may also be compiled directly when it is in the current directory:

```bash
nvcc -O3 -std=c++17 -arch=sm_XX -o paq9-cuda paq9-cuda.cu
```

## Usage

### Compression

```text
paq9-cuda -c <archive> [-level] <input-file>
```

Example:

```bash
./paq9-cuda -c archive.paq -5 input.txt
```

The optional `-level` argument controls the input chunk size as a power of two in MiB:

| Level | Chunk size |
|---:|---:|
| `-1` | 1 MiB |
| `-2` | 2 MiB |
| `-3` | 4 MiB |
| `-4` | 8 MiB |
| `-5` | 16 MiB |
| `-6` | 32 MiB |
| `-7` | 64 MiB |
| `-8` | 128 MiB |
| `-9` | 256 MiB |
| `-10` | 512 MiB |
| `-11` | 1 GiB |

Larger chunks generally provide a larger modeling context per GPU thread, but they also increase memory consumption and reduce the number of chunks that can be processed concurrently.

If no level is supplied, the program uses its default level (`1`).

### Decompression

```text
paq9-cuda -d <archive> [output-file]
```

Example:

```bash
./paq9-cuda -d archive.paq restored.txt
```

You can also omit the output filename:

```bash
./paq9-cuda -d archive.paq
```

In that case the filename stored in the `PAQ9-CUDA` archive header is used as the output filename.

## Archive format

The generated archive starts with the `PAQ9-CUDA` signature and stores enough metadata for decompression to recreate the original data.

The current implementation writes a header containing:

```text
+----------------------+-----------------------------+
| Field                | Description                 |
+----------------------+-----------------------------+
| Magic                | "PAQ9-CUDA"                |
| Version              | Archive format version     |
| Original filename    | Null-terminated name       |
| Mode                 | 'c' for compressed data    |
| Original size        | 8-byte big-endian size     |
| Chunk size           | 4-byte big-endian value    |
| Memory level         | 4-byte big-endian value    |
| Compression level    | 4-byte big-endian value    |
| Number of chunks     | 4-byte big-endian value    |
+----------------------+-----------------------------+
```

Each compressed chunk then stores its uncompressed size, compressed size, and compressed payload. Integer fields in the archive are written in **big-endian** order.

The archive is specific to this project and should not be expected to be compatible with the original PAQ9 command-line archive format.

## Memory model

PAQ-style compression is memory hungry. This implementation creates a separate model for each active CUDA thread. A model can allocate:

- LZP history buffers
- LZP hash tables
- context state arrays
- hash tables used by the predictor
- state maps, mixers, and APM tables
- range-coder buffers

The newer implementation queries CUDA for:

1. the GPU's available memory, and
2. the maximum CUDA device malloc-heap limit accepted by the runtime.

It then reserves only a fraction of the usable memory and determines how many chunks can be processed in one device call.

Because of this design, the practical maximum chunk size depends on the GPU, CUDA runtime, and the amount of memory available at execution time.

## Parallel execution model

The main compression/decompression kernel is launched with CUDA blocks and threads. The implementation uses **256 threads per block** and assigns one input chunk to a logical CUDA thread.

For a file containing many chunks, multiple chunks can therefore be processed in parallel:

```text
             CUDA Grid
┌──────────────────────────────────────────┐
│ Block 0         Block 1         Block 2  │
│ [C0 C1 ...]     [C256 ...]      [...]    │
└──────────────────────────────────────────┘
        │
        └── one chunk / one logical thread
```

If the number of chunks is larger than the amount of GPU memory that can safely be used in one launch, the host divides the work into multiple device calls.

## PAQ model overview

The predictor follows the general PAQ philosophy of combining many statistical contexts instead of relying on a single compression model.

At a high level:

```text
Input bytes
    │
    ▼
   LZP ──────────────┐
    │                │
    ▼                ▼
Contexts        Predicted byte
    │                │
    └──────┬─────────┘
           ▼
   Context prediction
           │
           ▼
      Mixer / APM
           │
           ▼
   Probability of next bit
           │
           ▼
      Range encoder
           │
           ▼
     Compressed data
```

The CUDA port keeps the model state in device memory so that each active chunk can be modeled independently without requiring a CPU-side model for every chunk.

## Output and diagnostics

During execution the program prints information such as:

- selected memory/chunk level
- total file size
- number of chunks
- CUDA block/thread assignment
- per-device-call progress
- compressed/uncompressed byte counts
- total execution time
- calculated compression/decompression throughput

The exact diagnostic output may change as the implementation evolves.

## Performance considerations

PAQ compression trades speed and memory for compression ratio. CUDA parallelism targets the expensive model computation, but overall throughput is also affected by:

- host-to-device transfers
- device-to-host transfers
- GPU memory capacity
- device malloc-heap capacity
- number and size of chunks
- GPU architecture
- compression characteristics of the input data

For meaningful benchmarks, compare the same input files, CUDA version, GPU, compiler optimization flags, and compression level.

Useful metrics include:

```text
Compression ratio = original_size / compressed_size

Throughput = processed_uncompressed_bytes / elapsed_time
```

## Correctness testing

A simple round-trip test is:

```bash
./paq9-cuda -c test.paq -5 input.bin
./paq9-cuda -d test.paq output.bin
```

Then compare the original and decompressed files:

```bash
cmp input.bin output.bin
```

or, on Windows PowerShell:

```powershell
fc.exe /b input.bin output.bin
```

A successful comparison means the decompressed file is byte-for-byte identical to the original input.

## Project status

This is an **experimental/research-oriented CUDA port of PAQ9**. The project is actively being developed and the archive format, memory-management strategy, and GPU execution strategy may change.

In particular, GPU memory consumption is substantially higher than a simple CPU implementation because each concurrently processed chunk owns its own prediction state and working memory.

## Repository structure

A typical checkout contains the CUDA implementation under the `PAQ9` directory:

```text
PAQ9-CUDA/
├── PAQ9/
│   └── paq9-cuda.cu
├── README.md
├── .gitignore
└── .gitattributes
```

## License / attribution

PAQ is a family of context-mixing data compressors associated with Matt Mahoney and related PAQ implementations. This repository is an independent CUDA-oriented implementation/port for experimentation with GPU acceleration.

Please review the source and repository history for the exact licensing and attribution requirements of any PAQ-derived code included in this project before redistributing binaries or modified source.

## Author

**Siam Ahmed**

GitHub: [@siam35862](https://github.com/siam35862)

Repository: [siam35862/PAQ9-CUDA](https://github.com/siam35862/PAQ9-CUDA)

---

If you use this project for research or benchmarking, please document the GPU model, CUDA version, compiler flags, chunk/compression level, input dataset, and measured throughput so results are reproducible.
