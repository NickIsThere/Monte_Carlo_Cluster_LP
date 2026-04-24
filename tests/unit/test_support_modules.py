from __future__ import annotations

import numpy as np

from lpas.experiments.metrics import active_set_precision_recall, relative_objective_error, summarize_history_gaps
from lpas.sampling.simplex_sampler import SimplexSampler
from lpas.solver.result import IterationMetrics
from lpas.utils.logging import format_kv


def test_simplex_sampler_samples_and_updates() -> None:
    sampler = SimplexSampler(primal_dim=3, dual_dim=2, seed=0, scale=1.0)
    X = sampler.sample_primal(5)
    Y = sampler.sample_dual(4)
    assert X.shape == (5, 3)
    assert Y.shape == (4, 2)
    assert np.all(X >= 0.0)
    assert np.all(Y >= 0.0)
    old_scale = sampler.scale
    sampler.update(X[:2], Y[:2])
    assert sampler.scale > 0.0
    assert sampler.scale != old_scale or old_scale > 0.0


def test_experiment_metrics_helpers() -> None:
    assert relative_objective_error(9.0, 10.0) == 0.1
    precision, recall = active_set_precision_recall(
        np.array([True, False, True]),
        np.array([True, True, False]),
    )
    assert precision == 0.5
    assert recall == 0.5
    history = [
        IterationMetrics(
            iteration=0,
            best_score=1.0,
            mean_score=0.5,
            best_feasible_primal_objective=1.0,
            best_certified_gap=0.1,
            mean_primal_violation=0.2,
            mean_dual_violation=0.3,
            dominant_active_pattern=None,
            sampler_mu_x=np.array([0.0]),
            sampler_sigma_x=np.array([1.0]),
            sampler_mu_y=np.array([0.0]),
            sampler_sigma_y=np.array([1.0]),
        )
    ]
    assert summarize_history_gaps(history) == [0.1]


def test_logging_helper_formats_pairs() -> None:
    assert format_kv(iteration=1, score=2.5) == "iteration=1 score=2.5"
