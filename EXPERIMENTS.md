# Experiments

This project investigates whether geometry-aware adaptive sampling can recover useful active-set structure in dense linear programs faster than naive Monte Carlo sampling. The method is evaluated as a diagnostic and solver-hint mechanism, not as a replacement for mature LP solvers such as HiGHS.

## Layers

1. Toy 2D/3D visual experiments
   They generate feasible-region plots, scored sample clouds, elite samples, raw-versus-polished solution views, active-constraint views, naive-vs-geometry comparisons, and convergence curves.
2. Random dense LP benchmarks
   They compare naive Monte Carlo, raw geometry-aware sampling, and geometry-aware sampling plus vertex polishing on bounded dense LP families using HiGHS as the reference solver.
3. Solver-hint experiments
   They reconstruct candidate corners from sampled active sets and measure whether the sampled support contains useful solver hints.

## Quick Runs

Run these from the repository root after activating the virtual environment:

```bash
python scripts/run_toy_visuals.py --quick
python scripts/run_random_dense_benchmark.py --quick
python scripts/run_solver_hint_experiment.py --quick
python scripts/make_benchmark_plots.py
```

## Full Runs

Each script supports common controls:

- `--seed`
- `--n-instances`
- `--n-variables`
- `--n-constraints`
- `--n-iterations`
- `--samples-per-iteration`
- `--output-dir`
- `--quick`

Examples:

```bash
python scripts/run_random_dense_benchmark.py --seed 7 --n-variables 5 10 20 --n-constraints 25 50 100 --n-instances 6
python scripts/run_solver_hint_experiment.py --seed 7 --n-variables 5 10 --n-constraints 25 50 --n-instances 4
python scripts/run_toy_visuals.py --seed 7 --samples-per-iteration 256 --n-iterations 30
```

## Metrics

- `active_set_recovery_accuracy`
  Jaccard overlap between a sampled active set and the HiGHS active set.
- `best_feasible_objective`
  Best objective among primal-feasible sampled candidates.
- `raw_best_objective` and `polished_objective`
  Best objective before and after the deterministic vertex-polishing pass.
- `best_objective_any_candidate`
  Best objective among all sampled primal candidates, feasible or not.
- `best_primal_violation`
  Lowest primal feasibility violation norm observed so far.
- `best_dual_violation`
  Lowest dual feasibility violation norm observed so far.
- `best_gap`
  Lowest duality-gap style score observed so far.
- `best_complementarity_error`
  Lowest complementarity error observed so far.
- `time_to_identify_optimal_active_constraints`
  First wall-clock time when active-set Jaccard reaches the recovery threshold.
- `constraints_in_top_k_support`
  Fraction of HiGHS active constraints covered by the top-ranked sampled support set.
- `objective_gap_to_highs`
  Difference between the HiGHS objective and the sampled or reconstructed objective.
- `polishing_candidates_generated`, `vertices_reconstructed`, `vertices_feasible`
  Bounded reconstruction diagnostics for the polishing pass.

Active-set recovery is the central research metric because the project is testing whether the sampling process discovers useful LP structure earlier than a naive baseline, not whether it replaces a mature solver.

## Vertex Polishing Flow

The geometry-aware experiments now include a deterministic polishing stage after sampling:

1. Build augmented constraints `[A; -I] x <= [b; 0]`.
2. Score candidate active constraints from slack magnitudes with a soft kernel.
3. Form bounded active-set candidates of size `n`.
4. Reconstruct vertices by solving the induced equality systems.
5. Reject infeasible or numerically invalid reconstructions.
6. Compare the best feasible polished vertex against the raw best sampled point and against HiGHS.

## Outputs

Toy figures:

- `outputs/figures/toy/toy2d_feasible_region.png`
- `outputs/figures/toy/toy2d_sampling_scores.png`
- `outputs/figures/toy/toy2d_elite_samples.png`
- `outputs/figures/toy/toy2d_active_constraints.png`
- `outputs/figures/toy/toy2d_naive_vs_geometry.png`
- `outputs/figures/toy/toy2d_convergence.png`
- `outputs/figures/toy/toy3d_projection.png`

Random dense benchmark:

- `outputs/benchmarks/random_dense/results.csv`
- `outputs/benchmarks/random_dense/results.json`
- `outputs/benchmarks/random_dense/summary.csv`
- `outputs/benchmarks/random_dense/summary.md`
- `outputs/figures/benchmarks/`

Solver hints:

- `outputs/benchmarks/solver_hints/results.csv`
- `outputs/benchmarks/solver_hints/results.json`
- `outputs/benchmarks/solver_hints/summary.csv`
- `outputs/benchmarks/solver_hints/summary.md`
- `outputs/figures/solver_hints/`

## Reporting Guidance

Use the reports as research evidence. Accept negative results and report them directly.

Acceptable framing:

- Geometry-aware sampling recovered the optimal active set earlier than naive sampling under the same sample budget.
- Geometry-aware sampling improved active-set overlap but did not improve final objective quality on this benchmark family.

Avoid overclaiming:

- Do not describe the project as beating or replacing HiGHS.
- Do not describe the solver-hint experiment as a true HiGHS warm start unless the solver interface actually supports it.
