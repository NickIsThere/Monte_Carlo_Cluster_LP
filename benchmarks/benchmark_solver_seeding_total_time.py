from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lpas.experiments.solver_seeding_total_time import (  # noqa: E402
    run_solver_seeding_total_time_benchmark,
    write_solver_seeding_outputs,
)
from lpas.utils.config import SamplerConfig, SolverConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark total-time vs solver-only-time for sampler seeding.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dimensions", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--ratios", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--backend", choices=["numpy_cpu"], default="numpy_cpu")
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


def main() -> None:
    args = parse_args()
    config = build_config(args.seed, args.quick)
    records = run_solver_seeding_total_time_benchmark(
        config=config,
        dimensions=[2] if args.quick else args.dimensions,
        ratios=[2] if args.quick else args.ratios,
        seeds=[args.seed] if args.quick else range(3),
        backend=args.backend,
    )
    paths = write_solver_seeding_outputs(records, args.output_dir, config_payload=vars(args))
    print("Saved solver seeding outputs:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
