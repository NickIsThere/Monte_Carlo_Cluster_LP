from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lpas.experiments.reporting import load_report_json, make_random_dense_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate random dense benchmark plots from results.json.")
    parser.add_argument(
        "--results-json",
        type=Path,
        default=Path("outputs/benchmarks/random_dense/results.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures/benchmarks"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_report_json(args.results_json)
    outputs = make_random_dense_plots(payload, args.output_dir)
    print("Regenerated benchmark plots:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
