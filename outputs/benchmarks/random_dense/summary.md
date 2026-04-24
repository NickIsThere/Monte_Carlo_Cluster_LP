# Random Dense Benchmark

This project investigates whether geometry-aware adaptive sampling can recover useful active-set structure in dense linear programs faster than naive Monte Carlo sampling. The method is evaluated as a diagnostic and solver-hint mechanism, not as a replacement for mature LP solvers such as HiGHS.

## Configuration

```json
{
  "batch_size": 96,
  "max_iter": 12,
  "elite_fraction": 0.1,
  "seed": 0,
  "active_tol": 1e-06,
  "feasibility_tol": 1e-07,
  "gap_tol": 1e-06,
  "patience": 6,
  "time_limit_seconds": null,
  "archive_limit_multiplier": 5,
  "variance_collapse_factor": 1.0001,
  "scoring": {
    "w_primal_obj": 1.0,
    "w_dual_obj": 0.25,
    "w_gap": 1.5,
    "w_pviol": 2.0,
    "w_dviol": 2.0,
    "w_comp": 1.0,
    "w_geo": 0.2,
    "w_active": 0.2,
    "geometry_sigma": 1.0,
    "geometry_dual_weight": 0.5,
    "active_similarity_beta": 0.5,
    "cluster_smoothing": 0.0
  },
  "sampler": {
    "seed": 0,
    "alpha": 0.7,
    "sigma_init": 1.35,
    "sigma_min": 1e-06,
    "sigma_max": 10.0,
    "primal_init_mean": 0.8,
    "dual_init_mean": 0.8,
    "max_retries": 8,
    "primal_violation_threshold": 1.0,
    "dual_violation_threshold": 1.0
  },
  "warm_start": {
    "feasibility_tol": 1e-07,
    "max_combinations": 512
  }
}
```

## Overall Method Comparison

| method | count | active_set_recovery_accuracy_mean | time_to_identify_optimal_active_constraints_mean | objective_gap_to_highs_mean | wall_clock_seconds_mean |
| --- | --- | --- | --- | --- | --- |
| Geometry-aware | 8 | 1.0000 | 0.0031 | 0.2222 | 0.0229 |
| Naive Monte Carlo | 8 | 0.9542 | 0.0033 | 0.3126 | 0.0132 |

## Family Breakdown

| method | family | count | active_set_recovery_accuracy_mean | objective_gap_to_highs_mean | wall_clock_seconds_mean |
| --- | --- | --- | --- | --- | --- |
| Geometry-aware | bounded_dense | 4 | 1.0000 | 0.1467 | 0.0215 |
| Geometry-aware | controlled_optimum | 4 | 1.0000 | 0.2976 | 0.0242 |
| Naive Monte Carlo | bounded_dense | 4 | 0.9500 | 0.1933 | 0.0117 |
| Naive Monte Carlo | controlled_optimum | 4 | 0.9583 | 0.4320 | 0.0148 |

## Interpretation

Geometry-aware sampling recovered the optimal active set earlier or more reliably on average under the same sample budget.