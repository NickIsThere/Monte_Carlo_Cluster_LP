from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lpas.experiments.benchmark_runner import (
    GEOMETRY_AWARE_METHOD,
    GEOMETRY_AWARE_POLISHED_METHOD,
    NAIVE_MONTE_CARLO_METHOD,
    run_problem_comparison,
)
from lpas.experiments.toy_lps import simple_3d_case, triangle_unique_optimum
from lpas.experiments.visualization import (
    plot_3d_projections,
    plot_active_constraints,
    plot_convergence,
    plot_elite_samples,
    plot_feasible_region,
    plot_naive_vs_geometry,
    plot_sampling_scores,
)
from lpas.utils.config import SamplerConfig, SolverConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run toy 2D/3D LP visualization experiments.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples-per-iteration", type=int, default=192)
    parser.add_argument("--n-iterations", type=int, default=28)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> SolverConfig:
    samples = 96 if args.quick else args.samples_per_iteration
    iterations = 14 if args.quick else args.n_iterations
    return SolverConfig(
        batch_size=samples,
        max_iter=iterations,
        patience=max(6, iterations // 2),
        seed=args.seed,
        sampler=SamplerConfig(
            seed=args.seed,
            sigma_init=1.25,
            primal_init_mean=0.75,
            dual_init_mean=0.75,
        ),
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    output_dir = args.output_dir / "figures" / "toy"
    output_dir.mkdir(parents=True, exist_ok=True)

    toy2d = triangle_unique_optimum()
    traces = {
        trace.method: trace
        for trace in run_problem_comparison(
            problem_name=toy2d.name,
            family="toy",
            problem=toy2d.problem,
            config=config,
            capture_samples=True,
        )
    }
    geometry_raw = traces[GEOMETRY_AWARE_METHOD]
    geometry = traces.get(GEOMETRY_AWARE_POLISHED_METHOD, geometry_raw)
    naive = traces[NAIVE_MONTE_CARLO_METHOD]

    plot_feasible_region(toy2d, geometry, output_dir / "toy2d_feasible_region.png")
    plot_sampling_scores(toy2d, geometry_raw, output_dir / "toy2d_sampling_scores.png")
    plot_elite_samples(toy2d, geometry_raw, output_dir / "toy2d_elite_samples.png")
    plot_active_constraints(toy2d, geometry, output_dir / "toy2d_active_constraints.png")
    plot_naive_vs_geometry(toy2d, naive, geometry, output_dir / "toy2d_naive_vs_geometry.png")
    plot_convergence(naive, geometry, output_dir / "toy2d_convergence.png")

    toy3d = simple_3d_case(seed=args.seed)
    toy3d_map = {
        result.method: result
        for result in run_problem_comparison(
            problem_name=toy3d.name,
            family="toy",
            problem=toy3d.problem,
            config=config,
            capture_samples=True,
        )
    }
    plot_3d_projections(
        toy3d,
        toy3d_map[NAIVE_MONTE_CARLO_METHOD],
        toy3d_map.get(GEOMETRY_AWARE_POLISHED_METHOD, toy3d_map[GEOMETRY_AWARE_METHOD]),
        output_dir / "toy3d_projection.png",
    )

    print("Saved toy visual outputs to", output_dir)
    print("Configuration:", asdict(config))


if __name__ == "__main__":
    main()
