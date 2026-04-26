from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lpas.experiments.corner_discovery import run_corner_discovery_experiment, write_corner_discovery_outputs  # noqa: E402
from lpas.utils.config import SamplerConfig, SolverConfig  # noqa: E402


DEFAULT_DIMENSIONS = [2, 5, 10, 12, 15, 18, 20, 30, 50]
DEFAULT_RATIOS = [2, 5, 10]
QUICK_DIMENSIONS = [5, 10, 15, 20]
QUICK_RATIOS = [2, 5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-corner discovery experiments.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dimensions", nargs="+", type=int, default=DEFAULT_DIMENSIONS)
    parser.add_argument("--ratios", nargs="+", type=int, default=DEFAULT_RATIOS)
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
    records = run_corner_discovery_experiment(
        config=config,
        dimensions=QUICK_DIMENSIONS if args.quick else args.dimensions,
        ratios=QUICK_RATIOS if args.quick else args.ratios,
        seeds=range(5),
    )
    paths = write_corner_discovery_outputs(records, args.output_dir, config_payload=vars(args))
    print("Saved corner discovery outputs:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
