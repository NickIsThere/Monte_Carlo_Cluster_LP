from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl-lpas"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from lpas.certificates.bounds import build_bound_certificate
from lpas.experiments.generators import tiny_known_lp
from lpas.solver.result import PolishedVertex
from lpas.solver.scipy_handoff import solve_with_scipy


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if hasattr(value, "__dict__"):
        return {key: _jsonify(item) for key, item in value.__dict__.items()}
    return value


def write_certificate_validation_outputs(output_root: Path) -> dict[str, Path]:
    problem = tiny_known_lp()
    scipy_reference = solve_with_scipy(problem)
    polished = PolishedVertex(
        x=np.array([2.0, 2.0]),
        objective=10.0,
        active_indices=(0, 1),
        feasible=True,
        primal_violation=0.0,
        reconstruction_residual=0.0,
        condition_number=1.0,
        source_sample_index=0,
        original_active_mask=np.array([True, True, False]),
        nonneg_active_mask=np.array([False, False]),
    )
    diagnostic_certificate = build_bound_certificate(
        problem,
        raw_x=np.array([3.0, 3.0]),
        raw_y=np.array([0.0, 0.0, 0.0]),
        raw_primal_objective=15.0,
        raw_dual_objective=0.0,
        scipy_result=scipy_reference,
    )
    certified_certificate = build_bound_certificate(
        problem,
        primal_x=np.array([2.0, 2.0]),
        primal_objective=10.0,
        dual_y=np.array([2.0, 1.0, 0.0]),
        dual_objective=10.0,
        polished_vertex=polished,
        scipy_result=scipy_reference,
    )

    results_dir = output_root / "results" / "certificates"
    figures_dir = output_root / "figures" / "certificates"
    _ensure_directory(results_dir)
    _ensure_directory(figures_dir)

    json_path = results_dir / "certificate_separation_examples.json"
    markdown_path = results_dir / "certificate_separation_summary.md"
    figure_path = figures_dir / "certificate_gap_comparison.png"
    json_path.write_text(
        json.dumps(
            {
                "raw_diagnostic_only": _jsonify(diagnostic_certificate),
                "valid_certificate": _jsonify(certified_certificate),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    markdown_path.write_text(
        "\n".join(
            [
                "# Certificate Separation",
                "",
                "Raw sampled gaps are recorded as diagnostic values only.",
                "",
                f"- Diagnostic raw gap: {diagnostic_certificate.raw_gap}",
                f"- Diagnostic certified gap: {diagnostic_certificate.certified_gap}",
                f"- Valid certified gap: {certified_certificate.certified_gap}",
                f"- Valid lower bound: {certified_certificate.best_feasible_primal_lower_bound}",
                f"- Valid upper bound: {certified_certificate.best_feasible_dual_upper_bound}",
            ]
        ),
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["raw diagnostic gap", "certified gap"]
    values = [
        abs(float(diagnostic_certificate.raw_gap or 0.0)),
        abs(float(certified_certificate.certified_gap or 0.0)),
    ]
    ax.bar(labels, values)
    ax.set_title("Diagnostic vs certified gap")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    return {
        "results_json": json_path,
        "summary_md": markdown_path,
        "figure": figure_path,
    }
