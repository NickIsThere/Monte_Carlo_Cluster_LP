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
  },
  "vertex_polishing": {
    "enabled": true,
    "elite_fraction": 0.05,
    "tau": 0.01,
    "method": "rbf",
    "max_ranked_constraints": null,
    "max_candidates_per_sample": 100,
    "max_total_candidates": 5000,
    "feasibility_tol": 1e-08,
    "residual_tol": 1e-08,
    "feasibility": {
      "absolute_tolerance": 1e-07,
      "relative_tolerance": 1e-07,
      "nonnegativity_tolerance": 1e-08
    }
  },
  "feasibility": {
    "absolute_tolerance": 1e-07,
    "relative_tolerance": 1e-07,
    "nonnegativity_tolerance": 1e-08
  },
  "search_mode": "sampling",
  "pdhg": {
    "steps_per_iteration": 5,
    "step_scale": 0.9
  },
  "stop_on_variance_collapse": true
}
```

## Overall Method Comparison

| method | count | raw_best_objective_mean | polished_objective_mean | active_set_recovery_accuracy_mean | polishing_wall_clock_seconds_mean | time_to_identify_optimal_active_constraints_mean | objective_gap_to_highs_mean | wall_clock_seconds_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Geometry-aware (raw) | 8 | 4.9777 | - | 0.0250 | 0.0000 | - | 0.1771 | 0.0204 |
| Geometry-aware + polishing | 8 | 4.9777 | 5.1548 | 1.0000 | 0.0132 | 0.0336 | 0.0000 | 0.0336 |
| Naive Monte Carlo | 8 | 4.8421 | - | 0.0250 | 0.0000 | - | 0.3126 | 0.0158 |

## Family Breakdown

| method | family | count | raw_best_objective_mean | polished_objective_mean | active_set_recovery_accuracy_mean | objective_gap_to_highs_mean | wall_clock_seconds_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Geometry-aware (raw) | bounded_dense | 4 | 2.4204 | - | 0.0500 | 0.1107 | 0.0200 |
| Geometry-aware (raw) | controlled_optimum | 4 | 7.5349 | - | 0.0000 | 0.2435 | 0.0208 |
| Geometry-aware + polishing | bounded_dense | 4 | 2.4204 | 2.5311 | 1.0000 | -0.0000 | 0.0344 |
| Geometry-aware + polishing | controlled_optimum | 4 | 7.5349 | 7.7784 | 1.0000 | 0.0000 | 0.0328 |
| Naive Monte Carlo | bounded_dense | 4 | 2.3378 | - | 0.0500 | 0.1933 | 0.0150 |
| Naive Monte Carlo | controlled_optimum | 4 | 7.3465 | - | 0.0000 | 0.4320 | 0.0166 |

## Interpretation

Vertex polishing reduced the mean objective gap after geometry-aware sampling while keeping the scientific framing as a solver-hint refinement step.