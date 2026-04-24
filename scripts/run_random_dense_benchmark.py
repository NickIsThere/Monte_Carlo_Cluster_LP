from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lpas.experiments.benchmark_runner import run_random_dense_suite
from lpas.experiments.random_dense import build_random_dense_suite
from lpas.experiments.reporting import write_random_dense_outputs
from lpas.utils.config import SamplerConfig, SolverConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the random dense LP benchmark suite.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-instances", type=int, default=6)
    parser.add_argument("--n-variables", nargs="+", type=int, default=[2, 5, 10])
    parser.add_argument("--n-constraints", nargs="+", type=int, default=[10, 25, 50])
    parser.add_argument("--n-iterations", type=int, default=32)
    parser.add_argument("--samples-per-iteration", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> SolverConfig:
    samples = 96 if args.quick else args.samples_per_iteration
    iterations = 12 if args.quick else args.n_iterations
    return SolverConfig(
        batch_size=samples,
        max_iter=iterations,
        patience=max(6, iterations // 2),
        seed=args.seed,
        sampler=SamplerConfig(
            seed=args.seed,
            sigma_init=1.35,
            primal_init_mean=0.8,
            dual_init_mean=0.8,
        ),
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    n_variables = [2, 5] if args.quick else args.n_variables
    n_constraints = [8, 12] if args.quick else args.n_constraints
    n_instances = 2 if args.quick else args.n_instances
    instances = build_random_dense_suite(
        n_variables=n_variables,
        n_constraints=n_constraints,
        n_instances=n_instances,
        seed=args.seed,
    )
    results = run_random_dense_suite(instances, config=config, capture_samples=False)
    paths = write_random_dense_outputs(results, args.output_dir, config_payload=asdict(config))
    print("Saved random dense benchmark outputs:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
