"""Run PAQ9-CUDA (warp-cooperative) compression/decompression benchmarks and write a CSV report."""

# PAQ9-CUDA benchmark documentation
#
# The executable now takes two independent tuning parameters, each from 1 to 11:
#   memory_level  Controls how much GPU memory (VRAM) is used (passed after -c).
#   chunk_level   Controls compression ratio vs. speed (passed as the second flag).
#
# This script runs every combination of memory_level (1-11) and chunk_level (1-11)
# in a nested sweep -- 121 combinations in total. For each combination it:
#   1. Compresses the original file with that memory_level and chunk_level.
#   2. Decompresses the archive immediately after that compression.
#   3. Compares the original file with the decompressed file.
#   4. Prints every result in the terminal and saves every result to one CSV file.
#
# That is 121 compressions + 121 decompressions = 242 operations in total.
#
# Required command-line parameters, in this order:
#   CUDA_EXECUTABLE          Path (or file name) of the paq9_cuda_warp executable.
#   UNCOMPRESSED_FILE        Path to the original input file.
#   COMPRESSED_FILE          Path to the archive created during each compression.
#   DECOMPRESS_SOURCE_FILE   The compressed file read as the source for each
#                            decompression (normally the same path as
#                            COMPRESSED_FILE, but kept as its own argument in
#                            case the two need to differ).
#   DECOMPRESSED_FILE        Path to the output file created during decompression.
#   CSV_FILE                 Path where the benchmark report will be written.
#
# Example on Windows PowerShell:
#   python benchmark.py `
#     "E:\Tools\paq9_cuda_warp.exe" `
#     "E:\Testing FIle\enwik9" `
#     "E:\Testing FIle\Compressed\enwik9.paq9-cuda" `
#     "E:\Testing FIle\Compressed\enwik9.paq9-cuda" `
#     "E:\Testing FIle\Decompressed\enwik9" `
#     "E:\Testing FIle\Benchmark.csv"
#
# Terminal output includes live run progress, PAQ9-CUDA metrics, file comparison
# results, return codes, and the final CSV output path. The comparison reads
# both files in 1 MiB chunks so large files do not need to fit in memory.
#
# The CSV file is opened, appended to with one new row, and closed again after
# every single compression or decompression -- not just once at the end and not
# only after each combination pair. This keeps the CSV on disk always current
# and readable, even if the benchmark is interrupted mid-run.

import argparse
import csv
import re
import subprocess
import sys
import threading
import time
from pathlib import Path


MIN_LEVEL = 1
MAX_LEVEL = 6


# Add a useful explanation when the script is started without the five file paths.
class BenchmarkArgumentParser(argparse.ArgumentParser):
    """Add path explanations when required benchmark arguments are missing."""

    def error(self, message: str) -> None:
        if "the following arguments are required" in message:
            message += (
                "\n\nRequired filepath parameters:\n"
                "  cuda          Path (or file name) of the paq9_cuda_warp executable.\n"
                "  source        Original uncompressed input file.\n"
                "  archive       Compressed archive file to create.\n"
                "  decompress_source  The compressed file that will be read as the\n"
                "                     source for each decompression.\n"
                "  decompressed  Output file created by each decompression.\n"
                "  csv           CSV report file to generate.\n\n"
                "Example:\n"
                '  python benchmark.py "E:\\Tools\\paq9_cuda_warp.exe" '
                '"E:\\Testing FIle\\enwik9" '
                '"E:\\Testing FIle\\Compressed\\enwik9.paq9-cuda" '
                '"E:\\Testing FIle\\Compressed\\enwik9.paq9-cuda" '
                '"E:\\Testing FIle\\Decompressed\\enwik9" '
                '"E:\\Testing FIle\\Benchmark.csv"'
            )
        super().error(message)


CSV_FIELDS = [
    # These names become the column headings in the final Benchmark.csv file.
    "run",
    "operation",
    "requested_memory_level",
    "requested_chunk_level",
    "working_mode",
    "memory_chunk_level_mb",
    "memory_level",
    "reported_level",
    "number_of_chunks",
    "total_threads",
    "maximum_thread_at_a_time",
    "total_input_bytes",
    "total_output_bytes",
    "compression_ratio",
    "gpu_memory_allocated_bytes",
    "gpu_memory_allocated_mib",
    "cuda_malloc_bytes",
    "device_heap_bytes",
    "time_seconds",
    "speed_kb_per_second",
    "return_code",
    "status",
    "files_match",
    "comparison_status",
    "comparison_error",
]


def number(pattern: str, output: str, flags: int = 0) -> str:
    """Return the first captured value for a label, or an empty value."""
    match = re.search(pattern, output, flags)
    return match.group(1) if match else ""


def parse_output(output: str) -> dict[str, str]:
    """Parse the labelled values printed by paq9_cuda_warp."""
    # The executable prints these values as human-readable terminal messages.
    total = re.search(
        r"Total:\s*([0-9]+)\s+Byte\s*->\s*([0-9]+)\s+Byte", output
    )
    gpu = re.search(
        r"Total GPU memory allocated:\s*([0-9]+)\s+bytes\s+\(([^)]+)\)",
        output,
    )
    cuda = re.search(
        r"cudaMalloc:\s*([0-9]+)\s+bytes;\s*device heap:\s*([0-9]+)\s+bytes",
        output,
    )

    values = {
        "working_mode": number(r"Working mode:\s*(.+)", output),
        "memory_chunk_level_mb": number(
            r"Memory Chunk (?:Level|Size):\s*([^\r\n]+)", output
        ).replace("MB", "").strip(),
        "memory_level": number(r"Memory Level:\s*([^\r\n]+)", output),
        "reported_level": number(r"Level:\s*([^\r\n]+)", output),
        "number_of_chunks": number(r"Number of Chunks:\s*([^\r\n]+)", output),
        "total_threads": number(r"Total threads:\s*([^\r\n]+)", output),
        "maximum_thread_at_a_time": number(
            r"Maximum Thread at a time:\s*([^\r\n]+)", output
        ),
        "compression_ratio": number(r"Compression Ratio:\s*([^\r\n]+)", output),
        "time_seconds": number(r"Time Taken:\s*([^\r\n]+?)\s+seconds", output),
        "speed_kb_per_second": number(
            r"Compression/Decompression Speed:\s*([^\r\n]+?)\s+KB/seconds",
            output,
        ),
    }
    if total:
        values["total_input_bytes"] = total.group(1)
        values["total_output_bytes"] = total.group(2)
    if gpu:
        values["gpu_memory_allocated_bytes"] = gpu.group(1)
        values["gpu_memory_allocated_mib"] = gpu.group(2).replace("MiB", "").strip()
    if cuda:
        values["cuda_malloc_bytes"] = cuda.group(1)
        values["device_heap_bytes"] = cuda.group(2)
    return values


def compare_files(source: Path, decompressed: Path) -> tuple[str, str, str]:
    """Compare two files in chunks and return match, status, and error text."""
    try:
        # Check the size first, then compare 1 MiB at a time to avoid loading a large file into RAM.
        if source.stat().st_size != decompressed.stat().st_size:
            return "no", "different_size", ""

        with source.open("rb") as source_file, decompressed.open("rb") as decompressed_file:
            while True:
                source_chunk = source_file.read(1024 * 1024)
                decompressed_chunk = decompressed_file.read(1024 * 1024)
                if source_chunk != decompressed_chunk:
                    return "no", "different_data", ""
                if not source_chunk:
                    return "yes", "identical", ""
    except OSError as error:
        return "", "error", str(error)


def print_timed_line(message: str, benchmark_started: float) -> None:
    """Print a permanent terminal line with the benchmark elapsed time."""
    elapsed = time.monotonic() - benchmark_started
    print(f"{message} [script elapsed: {elapsed:.0f}s]", flush=True)


def run_command(
    command: list[str],
    run_number: int,
    total_runs: int,
    benchmark_started: float,
) -> tuple[str, int]:
    """Run one benchmark command, echoing and retaining its terminal output."""
    # Merge stderr into stdout so CUDA errors are visible and retained for parsing.
    started = time.monotonic()
    print_timed_line("\n$ " + subprocess.list2cmdline(command), benchmark_started)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    output_lines = []
    stop_progress = threading.Event()

    # A child process can buffer its C++ output when stdout is redirected to a pipe.
    # This heartbeat still shows that the benchmark is active during a long run.
    def report_progress() -> None:
        while not stop_progress.wait(1):
            run_elapsed = time.monotonic() - started
            script_elapsed = time.monotonic() - benchmark_started
            print(
                f"\r[progress] run {run_number}/{total_runs} still running "
                f"(run: {run_elapsed:.0f}s)"
                f" [script elapsed: {script_elapsed:.0f}s]\033[K\r",
                end="",
                flush=True,
            )

    progress_thread = threading.Thread(target=report_progress, daemon=True)
    progress_thread.start()
    assert process.stdout is not None
    for line in process.stdout:
        # Add the timestamp only to the displayed line; keep the original line for parsing.
        print_timed_line(line.rstrip("\r\n"), benchmark_started)
        output_lines.append(line)
    return_code = process.wait()
    stop_progress.set()
    progress_thread.join()
    run_elapsed = time.monotonic() - started
    script_elapsed = time.monotonic() - benchmark_started
    # Finish the temporary heartbeat line, then print the permanent status normally.
    print("\r\033[K", end="", flush=True)
    print_timed_line(
        f"[progress] run {run_number}/{total_runs} finished with exit code "
        f"{return_code} (run: {run_elapsed:.2f}s)",
        benchmark_started,
    )
    return "".join(output_lines), return_code


def append_row_to_csv(row: dict[str, str], csv_path: Path) -> None:
    """Open the CSV file, append one row, and close it immediately.

    Writes the header first if the file does not exist yet. Opening and
    closing the file around every single row (rather than keeping it open
    for the whole benchmark) means the CSV on disk is always complete and
    readable up to the last finished operation, even if the run is
    interrupted or crashes partway through.
    """
    file_is_new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        if file_is_new:
            writer.writeheader()
        writer.writerow(row)


def print_run_result(row: dict[str, str], total_runs: int, benchmark_started: float) -> None:
    """Print the parsed result immediately after one operation finishes."""
    result_fields = [
        "operation",
        "requested_memory_level",
        "requested_chunk_level",
        "working_mode",
        "memory_chunk_level_mb",
        "memory_level",
        "reported_level",
        "number_of_chunks",
        "total_threads",
        "maximum_thread_at_a_time",
        "total_input_bytes",
        "total_output_bytes",
        "compression_ratio",
        "gpu_memory_allocated_bytes",
        "gpu_memory_allocated_mib",
        "cuda_malloc_bytes",
        "device_heap_bytes",
        "time_seconds",
        "speed_kb_per_second",
        "return_code",
        "status",
        "files_match",
        "comparison_status",
        "comparison_error",
    ]
    print_timed_line(f"[result] run {row['run']}/{total_runs}", benchmark_started)
    for field in result_fields:
        value = row.get(field, "")
        if value != "":
            print_timed_line(f"  {field}: {value}", benchmark_started)


def benchmark(args: argparse.Namespace) -> None:
    # Create the CSV folder if it does not exist. Start from a fresh CSV file so
    # a leftover file from an earlier run doesn't get appended to.
    benchmark_started = time.monotonic()
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    if args.csv.exists():
        args.csv.unlink()
    run_number = 0
    levels = range(MIN_LEVEL, MAX_LEVEL + 1)
    total_combinations = len(levels) * len(levels)
    total_runs = total_combinations * 2  # one compression + one decompression each

    for memory_level in levels:
        for chunk_level in levels:
            # --- Compression: -c -<memory_level> <archive> -<chunk_level> <source> ---
            run_number += 1
            print_timed_line(
                f"\n[progress] starting run {run_number}/{total_runs}: "
                f"compression memory_level={memory_level} chunk_level={chunk_level}",
                benchmark_started,
            )
            compression_command = [
                str(args.cuda),
                "-c",
                f"-{memory_level}",
                str(args.archive),
                f"-{chunk_level}",
                str(args.source),
            ]
            compression_output, compression_code = run_command(
                compression_command, run_number, total_runs, benchmark_started
            )
            compression_row = {field: "" for field in CSV_FIELDS}
            compression_row.update(parse_output(compression_output))
            compression_row.update(
                {
                    "run": run_number,
                    "operation": "compression",
                    "requested_memory_level": memory_level,
                    "requested_chunk_level": chunk_level,
                    "return_code": compression_code,
                    "status": "ok" if compression_code == 0 else "failed",
                }
            )
            print_run_result(compression_row, total_runs, benchmark_started)
            # Open the CSV, append this compression's row, and close it right away
            # so the file on disk is up to date after every single operation.
            append_row_to_csv(compression_row, args.csv)

            # --- Decompression: -d <archive> <decompressed> ---
            run_number += 1
            print_timed_line(
                f"\n[progress] starting run {run_number}/{total_runs}: "
                f"decompression after memory_level={memory_level} chunk_level={chunk_level}",
                benchmark_started,
            )
            decompression_command = [
                str(args.cuda),
                "-d",
                str(args.decompress_source),
                str(args.decompressed),
            ]
            decompression_output, decompression_code = run_command(
                decompression_command, run_number, total_runs, benchmark_started
            )
            decompression_row = {field: "" for field in CSV_FIELDS}
            decompression_row.update(parse_output(decompression_output))
            decompression_row.update(
                {
                    "run": run_number,
                    "operation": "decompression",
                    "requested_memory_level": memory_level,
                    "requested_chunk_level": chunk_level,
                    "return_code": decompression_code,
                    "status": "ok" if decompression_code == 0 else "failed",
                }
            )
            if decompression_code == 0:
                # Verify that decompression restored the original input exactly.
                files_match, comparison_status, comparison_error = compare_files(
                    args.source, args.decompressed
                )
                decompression_row.update(
                    {
                        "files_match": files_match,
                        "comparison_status": comparison_status,
                        "comparison_error": comparison_error,
                    }
                )
            else:
                decompression_row["comparison_status"] = "skipped_decompression_failed"
            print_run_result(decompression_row, total_runs, benchmark_started)
            # Open the CSV, append this decompression's row, and close it right away.
            append_row_to_csv(decompression_row, args.csv)

    print_timed_line(f"\nBenchmark complete. CSV written to: {args.csv}", benchmark_started)


def parse_args() -> argparse.Namespace:
    parser = BenchmarkArgumentParser(
        description=(
            "Run PAQ9-CUDA (warp-cooperative) for every memory_level (1-11) x "
            "chunk_level (1-11) combination, compare every decompression, "
            "and create a CSV report."
        ),
        epilog=(
            "The six filepath arguments are, in order: the paq9_cuda_warp executable, "
            "original uncompressed file, compressed archive to create, the compressed "
            "file to read as the decompression source, decompressed output file, and "
            "CSV report file. This runs 121 memory_level/chunk_level combinations "
            "(242 operations total)."
        ),
    )
    parser.add_argument(
        "cuda", metavar="CUDA_EXECUTABLE", type=Path,
        help="path (or file name) of the paq9_cuda_warp executable",
    )
    parser.add_argument(
        "source", metavar="UNCOMPRESSED_FILE", type=Path,
        help="original uncompressed input file",
    )
    parser.add_argument(
        "archive", metavar="COMPRESSED_FILE", type=Path,
        help="compressed archive file to create during each compression",
    )
    parser.add_argument(
        "decompress_source", metavar="DECOMPRESS_SOURCE_FILE", type=Path,
        help=(
            "compressed file read as the source for each decompression "
            "(normally the same path as COMPRESSED_FILE)"
        ),
    )
    parser.add_argument(
        "decompressed", metavar="DECOMPRESSED_FILE", type=Path,
        help="output file created by each decompression",
    )
    parser.add_argument(
        "csv", metavar="CSV_FILE", type=Path,
        help="CSV report file to generate",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        benchmark(parse_args())
    except (FileNotFoundError, OSError) as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        sys.exit(1)