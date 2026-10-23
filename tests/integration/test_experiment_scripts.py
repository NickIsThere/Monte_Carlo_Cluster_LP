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


def test_quick_gpu_throughput_script_runs_and_writes_outputs(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "benchmarks/benchmark_gpu_throughput.py", "--quick", "--backends", "numpy_cpu", "--output-dir", str(tmp_path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert (tmp_path / "results" / "gpu_benchmarks" / "gpu_throughput_results.csv").exists()
    assert (tmp_path / "results" / "gpu_benchmarks" / "gpu_throughput_results.json").exists()
    assert (tmp_path / "figures" / "gpu_benchmarks" / "samples_per_second_vs_K.png").exists()


def test_quick_full_extension_gpu_stage_runs_and_writes_outputs(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "experiments/run_full_extension_evaluation.py", "--quick", "--stage", "gpu-benchmark", "--output-dir", str(tmp_path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert (tmp_path / "results" / "gpu_benchmarks" / "gpu_throughput_results.csv").exists()
    assert (tmp_path / "results" / "gpu_benchmarks" / "gpu_throughput_results.json").exists()
    assert (tmp_path / "figures" / "gpu_benchmarks" / "samples_per_second_vs_K.png").exists()


def test_quick_scaling_script_runs_and_writes_outputs(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "experiments/scaling_by_dimension.py", "--quick", "--output-dir", str(tmp_path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert (tmp_path / "results" / "scaling" / "scaling_results.csv").exists()
    assert (tmp_path / "results" / "scaling" / "scaling_summary_by_dimension.csv").exists()
    assert (tmp_path / "figures" / "scaling" / "success_rate_vs_dimension.png").exists()


def test_quick_solver_seeding_benchmark_runs_and_writes_outputs(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "benchmarks/benchmark_solver_seeding_total_time.py", "--quick", "--output-dir", str(tmp_path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert (tmp_path / "results" / "solver_seeding" / "solver_seeding_total_time_results.csv").exists()
    assert (tmp_path / "results" / "solver_seeding" / "solver_seeding_total_time_results.json").exists()
    assert (tmp_path / "figures" / "solver_seeding" / "time_breakdown_stacked_bar.png").exists()


def test_quick_corner_discovery_script_runs_and_writes_outputs(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "experiments/corner_discovery.py", "--quick", "--output-dir", str(tmp_path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert (tmp_path / "results" / "corner_discovery" / "corner_discovery_results.csv").exists()
    assert (tmp_path / "results" / "corner_discovery" / "corner_discovery_results.json").exists()
    assert (tmp_path / "figures" / "corner_discovery" / "active_set_jaccard_vs_dimension.png").exists()


def test_certificate_validation_script_runs_and_writes_outputs(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "experiments/certificate_validation.py", "--output-dir", str(tmp_path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert (tmp_path / "results" / "certificates" / "certificate_separation_examples.json").exists()
    assert (tmp_path / "results" / "certificates" / "certificate_separation_summary.md").exists()
    assert (tmp_path / "figures" / "certificates" / "certificate_gap_comparison.png").exists()


def test_primal_dual_comparison_writes_three_method_results(tmp_path: Path) -> None:
    import json

    _run_script("scripts/run_primal_dual_comparison.py", tmp_path)
    payload = json.loads((tmp_path / "results.json").read_text())
    assert {row["method"] for row in payload["records"]} == {"sampling", "pdhg", "hybrid"}
    assert (tmp_path / "comparison.png").exists()
    assert (tmp_path / "summary.md").exists()
    for row in payload["records"]:
        assert row["total_time_seconds"] >= row["directed_time_seconds"]
        if not row["feasible"]:
            assert row["relative_objective_error"] is None


def test_primal_dual_followup_exports_raw_and_recovered_controls(tmp_path: Path) -> None:
    import json

    _run_script("scripts/run_primal_dual_followup.py", tmp_path)
    payload = json.loads((tmp_path / "results.json").read_text())
    assert {r["profile"] for r in payload["records"]} == {"raw", "recovered"}
    assert {r["method"] for r in payload["records"]} == {"sampling", "pdhg", "hybrid"}
    assert (tmp_path / "comparison.png").exists()
    for record in payload["records"]:
        assert len(record["history"]) <= 64
        assert record["recovery_enabled"] == (record["profile"] == "recovered")
