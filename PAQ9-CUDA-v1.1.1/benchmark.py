"""Run PAQ9-CUDA compression/decompression benchmarks and write a CSV report."""

# PAQ9-CUDA benchmark documentation
#
# This script runs 22 operations in total:
#   1. Compress the original file with levels -1 through -11.
#   2. Decompress the archive immediately after each compression.
#   3. Compare the original file with the decompressed file.
#   4. Print every result in the terminal and save every result to one CSV file.
#
# Required command-line parameters, in this order:
#   UNCOMPRESSED_FILE  Path to the original input file.
#   COMPRESSED_FILE    Path to the archive created by PAQ9-CUDA.
#   DECOMPRESSED_FILE  Path to the output file created during decompression.
#   CSV_FILE           Path where the benchmark report will be written.
#
# Example on Windows PowerShell:
#   python benchmark.py `
#     "E:\Testing FIle\enwik9" `
#     "E:\Testing FIle\Compressed\enwik9.paq9-cuda" `
#     "E:\Testing FIle\Decompressed\enwik9" `
#     "E:\Testing FIle\Benchmark.csv"
#
# The optional --executable parameter can specify a different paq9-cuda.exe.
# If it is omitted, paq9-cuda.exe is expected beside this Python script.
#
# Terminal output includes live run progress, PAQ9-CUDA metrics, file comparison
# results, return codes, and the final CSV output path. The comparison reads
# both files in 1 MiB chunks so large files do not need to fit in memory.

import argparse
import csv
import re
import subprocess
import sys
import threading
import time
from pathlib import Path


DEFAULT_EXECUTABLE = Path(__file__).with_name("paq9-cuda.exe")


# Add a useful explanation when the script is started without the four file paths.
class BenchmarkArgumentParser(argparse.ArgumentParser):
    """Add path explanations when required benchmark arguments are missing."""

    def error(self, message: str) -> None:
        if "the following arguments are required" in message:
            message += (
                "\n\nRequired filepath parameters:\n"
                "  source        Original uncompressed input file.\n"
                "  archive       Compressed archive file to create and decompress.\n"
                "  decompressed  Output file created by each decompression.\n"
                "  csv           CSV report file to generate.\n\n"
                "Example:\n"
                '  python benchmark.py "E:\\Testing FIle\\enwik9" '
                '"E:\\Testing FIle\\Compressed\\enwik9.paq9-cuda" '
                '"E:\\Testing FIle\\Decompressed\\enwik9" '
                '"E:\\Testing FIle\\Benchmark.csv"'
            )
        super().error(message)


CSV_FIELDS = [
    # These names become the column headings in the final Benchmark.csv file.
    "run",
    "operation",
    "requested_level",
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
    """Parse the labelled values printed by paq9-cuda."""
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


def print_run_result(row: dict[str, str], benchmark_started: float) -> None:
    """Print the parsed result immediately after one operation finishes."""
    result_fields = [
        "operation",
        "requested_level",
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
    print_timed_line(f"[result] run {row['run']}/{22}", benchmark_started)
    for field in result_fields:
        value = row.get(field, "")
        if value != "":
            print_timed_line(f"  {field}: {value}", benchmark_started)


def benchmark(args: argparse.Namespace) -> None:
    # Create the CSV folder if it does not exist, then run 11 compression/decompression pairs.
    benchmark_started = time.monotonic()
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    run_number = 0
    total_runs = 22

    for level in range(1, 12):
        run_number += 1
        print_timed_line(
            f"\n[progress] starting run {run_number}/{total_runs}: compression -{level}",
            benchmark_started,
        )
        compression_command = [
            str(args.executable),
            "-c",
            str(args.archive),
            f"-{level}",
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
                "requested_level": level,
                "return_code": compression_code,
                "status": "ok" if compression_code == 0 else "failed",
            }
        )
        print_run_result(compression_row, benchmark_started)
        rows.append(compression_row)

        # Decompress immediately after this level before starting the next compression level.
        run_number += 1
        print_timed_line(
            f"\n[progress] starting run {run_number}/{total_runs}: decompression after -{level}",
            benchmark_started,
        )
        decompression_command = [
            str(args.executable),
            "-d",
            str(args.archive),
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
                "requested_level": level,
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
        print_run_result(decompression_row, benchmark_started)
        rows.append(decompression_row)

    with args.csv.open("w", newline="", encoding="utf-8") as csv_file:
        # Write all 22 operation results after the complete benchmark finishes.
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print_timed_line(f"\nBenchmark complete. CSV written to: {args.csv}", benchmark_started)


def parse_args() -> argparse.Namespace:
    parser = BenchmarkArgumentParser(
        description=(
            "Run PAQ9-CUDA levels 1 through 11, compare every decompression, "
            "and create a CSV report."
        ),
        epilog=(
            "The four filepath arguments are, in order: original uncompressed file, "
            "compressed archive, decompressed output file, and CSV report file."
        ),
    )
    parser.add_argument(
        "source", metavar="UNCOMPRESSED_FILE", type=Path,
        help="original uncompressed input file",
    )
    parser.add_argument(
        "archive", metavar="COMPRESSED_FILE", type=Path,
        help="compressed archive file to create and decompress",
    )
    parser.add_argument(
        "decompressed", metavar="DECOMPRESSED_FILE", type=Path,
        help="output file created by each decompression",
    )
    parser.add_argument(
        "csv", metavar="CSV_FILE", type=Path,
        help="CSV report file to generate",
    )
    parser.add_argument(
        "--executable",
        type=Path,
        default=DEFAULT_EXECUTABLE,
        help="Path to paq9-cuda.exe (defaults to the executable beside this script)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        benchmark(parse_args())
    except (FileNotFoundError, OSError) as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        sys.exit(1)