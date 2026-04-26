from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lpas.experiments.scaling_by_dimension import run_scaling_experiment, write_scaling_outputs  # noqa: E402
from lpas.utils.config import SamplerConfig, SolverConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scaling experiments across dimensions and constraint ratios.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dimensions", nargs="+", type=int, default=[2, 5, 10, 12, 15, 18, 20, 30, 50])
    parser.add_argument("--ratios", nargs="+", type=int, default=[2, 5, 10])
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
    records = run_scaling_experiment(
        config=config,
        dimensions=[5, 10, 15, 20] if args.quick else args.dimensions,
        ratios=[2, 5] if args.quick else args.ratios,
        seeds=range(5),
        backend=args.backend,
    )
    paths = write_scaling_outputs(records, args.output_dir, config_payload=vars(args))
    print("Saved scaling outputs:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
