from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lpas.experiments.gpu_throughput import (  # noqa: E402
    BENCHMARK_DIMENSIONS,
    benchmark_candidate_throughput,
    write_gpu_throughput_outputs,
)


DEFAULT_BACKENDS = ["numpy_cpu", "torch_cpu", "torch_mps", "torch_cuda", "numba_cpu"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark explicit backend candidate throughput.")
    parser.add_argument("--backends", nargs="+", default=DEFAULT_BACKENDS)
    parser.add_argument("--sizes", nargs="+", choices=sorted(BENCHMARK_DIMENSIONS), default=["small", "medium", "large"])
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    size_names = ["small"] if args.quick else args.sizes
    records = []
    for size_name in size_names:
        size = BENCHMARK_DIMENSIONS[size_name]
        K_values = [size["K"][0]] if args.quick else size["K"]
        for K in K_values:
            for backend in args.backends:
                try:
                    record = benchmark_candidate_throughput(
                        backend=backend,
                        n=size["n"],
                        m=size["m"],
                        K=K,
                        chunk_size=args.chunk_size,
                        seed=args.seed,
                        dtype=args.dtype,
                    )
                except RuntimeError as exc:
                    print(f"Skipping {backend} for {size_name} K={K}: {exc}")
                    continue
                records.append(record)
                print(_format_record(record))
    paths = write_gpu_throughput_outputs(records, args.output_dir, config_payload=vars(args))
    print("Saved GPU throughput benchmark outputs:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


def _format_record(record) -> str:
    return (
        f"{record.backend} n={record.n} m={record.m} K={record.K} "
        f"samples/s={record.samples_per_second:.2f} elapsed={record.elapsed_seconds:.4f}s"
    )


if __name__ == "__main__":
    main()
