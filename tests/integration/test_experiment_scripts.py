from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _run_script(script: str, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, script, "--quick", "--output-dir", str(output_dir)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_quick_toy_visual_script_runs_and_writes_outputs(tmp_path: Path) -> None:
    _run_script("scripts/run_toy_visuals.py", tmp_path)
    figure_dir = tmp_path / "figures" / "toy"
    expected = [
        figure_dir / "toy2d_feasible_region.png",
        figure_dir / "toy2d_sampling_scores.png",
        figure_dir / "toy2d_elite_samples.png",
        figure_dir / "toy2d_active_constraints.png",
        figure_dir / "toy2d_naive_vs_geometry.png",
        figure_dir / "toy2d_convergence.png",
        figure_dir / "toy3d_projection.png",
    ]
    for path in expected:
        assert path.exists()


def test_quick_random_dense_benchmark_runs_and_writes_outputs(tmp_path: Path) -> None:
    _run_script("scripts/run_random_dense_benchmark.py", tmp_path)
    assert (tmp_path / "benchmarks" / "random_dense" / "results.csv").exists()
    assert (tmp_path / "benchmarks" / "random_dense" / "results.json").exists()
    assert (tmp_path / "benchmarks" / "random_dense" / "summary.md").exists()
    assert (tmp_path / "figures" / "benchmarks" / "active_set_recovery_vs_iteration.png").exists()


def test_quick_solver_hint_experiment_runs_and_writes_outputs(tmp_path: Path) -> None:
    _run_script("scripts/run_solver_hint_experiment.py", tmp_path)
    assert (tmp_path / "benchmarks" / "solver_hints" / "results.csv").exists()
    assert (tmp_path / "benchmarks" / "solver_hints" / "results.json").exists()
    assert (tmp_path / "benchmarks" / "solver_hints" / "summary.md").exists()
    assert (tmp_path / "figures" / "solver_hints" / "hint_active_set_jaccard_distribution.png").exists()
