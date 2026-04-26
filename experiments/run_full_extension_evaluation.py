from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lpas.experiments.certificate_validation import write_certificate_validation_outputs  # noqa: E402
from lpas.experiments.corner_discovery import run_corner_discovery_experiment, write_corner_discovery_outputs  # noqa: E402
from lpas.experiments.gpu_throughput import (  # noqa: E402
    BENCHMARK_DIMENSIONS,
    DEFAULT_BENCHMARK_CHUNK_SIZE,
    benchmark_candidate_throughput,
    run_gpu_throughput_suite,
    write_gpu_throughput_outputs,
)
from lpas.experiments.scaling_by_dimension import run_scaling_experiment, write_scaling_outputs  # noqa: E402
from lpas.experiments.solver_seeding_total_time import (  # noqa: E402
    run_solver_seeding_total_time_benchmark,
    write_solver_seeding_outputs,
)
from lpas.utils.config import SamplerConfig, SolverConfig  # noqa: E402


FULL_SCALING_DIMENSIONS = (2, 5, 10, 12, 15, 18, 20, 30, 50)
FULL_SCALING_RATIOS = (2, 5, 10)
QUICK_SCALING_DIMENSIONS = (5, 10, 15, 20)
QUICK_SCALING_RATIOS = (2, 5)

FULL_CORNER_DIMENSIONS = (2, 5, 10, 12, 15, 18, 20, 30, 50)
FULL_CORNER_RATIOS = (2, 5, 10)
QUICK_CORNER_DIMENSIONS = (5, 10, 15, 20)
QUICK_CORNER_RATIOS = (2, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full benchmark and certification extension.")
    parser.add_argument("--stage", choices=["gpu-benchmark", "scaling", "corners", "seeding", "certificates", "all"], default="all")
    parser.add_argument("--backend", default="numpy_cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def build_config(seed: int, quick: bool) -> SolverConfig:
    batch_size = 96 if quick else 256
    iterations = 10 if quick else 24
    return SolverConfig(
        batch_size=batch_size,
        max_iter=iterations,
        patience=max(4, iterations // 2),
        seed=seed,
        sampler=SamplerConfig(
            seed=seed,
            sigma_init=1.35,
            primal_init_mean=0.8,
            dual_init_mean=0.8,
        ),
    )


def _run_gpu_stage(args: argparse.Namespace) -> None:
    if args.quick:
        small = BENCHMARK_DIMENSIONS["small"]
        records = [
            benchmark_candidate_throughput(
                backend=args.backend,
                n=int(small["n"]),
                m=int(small["m"]),
                K=int(small["K"][0]),
                chunk_size=DEFAULT_BENCHMARK_CHUNK_SIZE,
                seed=args.seed,
            )
        ]
    else:
        records = run_gpu_throughput_suite(
            backends=[args.backend],
            size_names=BENCHMARK_DIMENSIONS.keys(),
            seed=args.seed,
            chunk_size=DEFAULT_BENCHMARK_CHUNK_SIZE,
        )
    write_gpu_throughput_outputs(records, args.output_dir, config_payload=vars(args))


def _run_scaling_stage(args: argparse.Namespace, config: SolverConfig) -> None:
    records = run_scaling_experiment(
        config=config,
        dimensions=QUICK_SCALING_DIMENSIONS if args.quick else FULL_SCALING_DIMENSIONS,
        ratios=QUICK_SCALING_RATIOS if args.quick else FULL_SCALING_RATIOS,
        seeds=range(5),
        backend="numpy_cpu",
    )
    write_scaling_outputs(records, args.output_dir, config_payload=vars(args))


def _run_corner_stage(args: argparse.Namespace, config: SolverConfig) -> None:
    records = run_corner_discovery_experiment(
        config=config,
        dimensions=QUICK_CORNER_DIMENSIONS if args.quick else FULL_CORNER_DIMENSIONS,
        ratios=QUICK_CORNER_RATIOS if args.quick else FULL_CORNER_RATIOS,
        seeds=range(5),
    )
    write_corner_discovery_outputs(records, args.output_dir, config_payload=vars(args))


def _run_seeding_stage(args: argparse.Namespace, config: SolverConfig) -> None:
    records = run_solver_seeding_total_time_benchmark(
        config=config,
        dimensions=[2] if args.quick else (2, 3, 4),
        ratios=[2] if args.quick else (2, 4),
        seeds=[args.seed] if args.quick else range(3),
        backend="numpy_cpu",
    )
    write_solver_seeding_outputs(records, args.output_dir, config_payload=vars(args))


def _run_certificate_stage(args: argparse.Namespace) -> None:
    write_certificate_validation_outputs(args.output_dir)


def main() -> None:
    args = parse_args()
    config = build_config(args.seed, args.quick)
    if args.stage in {"gpu-benchmark", "all"}:
        _run_gpu_stage(args)
    if args.stage in {"scaling", "all"}:
        _run_scaling_stage(args, config)
    if args.stage in {"corners", "all"}:
        _run_corner_stage(args, config)
    if args.stage in {"seeding", "all"}:
        _run_seeding_stage(args, config)
    if args.stage in {"certificates", "all"}:
        _run_certificate_stage(args)


if __name__ == "__main__":
    main()
