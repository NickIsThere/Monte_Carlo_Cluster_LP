# Geometry-Aware LP Adaptive Sampling

This repository explores geometry-aware primal-dual adaptive sampling for LP active-set discovery and vertex reconstruction. It samples primal and dual candidates, scores them with feasibility and duality diagnostics, tracks recurring active-set patterns, and compares the resulting corner candidates against SciPy/HiGHS.

The current optimization pipeline is:

`sample -> soft active-set inference -> vertex reconstruction -> feasibility verification -> objective evaluation`

The polishing step is a deterministic local refinement stage applied to elite primal samples. It is intended to turn near-corner samples into verified LP vertices, not to replace HiGHS.

High-objective infeasible reconstructed vertices are not valid LP solutions. The evaluation code therefore ranks reconstructed vertices feasibility-first and treats infeasible high-objective candidates as diagnostics rather than successful reconstructions.

## What It Is

- A Python package built around dense LPs of the form `Ax <= b`, `x >= 0`.
- A research prototype for testing whether adaptive sampling, active-set statistics, and polishing can identify useful LP structure and corner candidates.
- A hybrid tool that can surface candidate solutions, dual certificates, and active-set hints alongside a classical solver baseline.
- A bounded vertex-polishing layer that reconstructs exact candidate vertices from soft active-set guesses.

## What It Is Not

- It is not a replacement for mature LP solvers such as HiGHS.
- It does not implement true HiGHS warm-start support in this version.
- It is not optimized for large sparse LPs or production-scale workloads.
- The new parallel backend is not a replacement for Simplex, Interior-Point, HiGHS, or PDLP.


## Package Layout

- `src/lpas/core`: LP representation, feasibility, primal-dual metrics, active sets, certificates, scoring.
- `src/lpas/sampling`: Gaussian adaptive sampler, truncated feasibility-aware wrapper, simplex-style experimental sampler.
- `src/lpas/geometry`: Jaccard similarity, cluster support, kernel density rewards.
- `src/lpas/solver`: adaptive loop, warm-start reconstruction, SciPy handoff, result objects.
- `src/lpas/backends`: explicit `numpy_cpu`, `torch_cpu`, `torch_cuda`, `torch_mps`, and `numba_cpu` backend selection.
- `src/lpas/gpu`: Torch batch evaluation, active-set helpers, chunked device-oriented solver entry points, and memory estimates.
- `src/lpas/cpu_accel`: explicit Numba kernels and score helpers for `backend="numba_cpu"`.
- `src/lpas/experiments`: LP generators, benchmark helpers, summary metrics.
- `tests`: unit, integration, and regression coverage.
- `examples`, `scripts`, `notebooks`: runnable demos and lightweight experiment entry points.

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Optional parallel-backend extras:

```bash
pip install -e '.[gpu]'
pip install -e '.[numba]'
pip install -e '.[parallel]'
```

PyTorch installation remains explicit because CUDA and Apple Silicon wheel selection is platform-specific. Numba is also explicit so the base environment stays lightweight.

Apple Silicon MPS remediation:

```bash
rm -rf .venv
/opt/homebrew/bin/python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e '.[gpu]'
python -c 'import torch; print(torch.backends.mps.is_available())'
python -c 'import torch; print(torch.ones(1, device="mps"))'
```

If an Apple Silicon environment reports `mps_built=True` but `mps_available=False`, or `torch.ones(..., device="mps")` fails with an OS-version check, rebuild the environment with Homebrew Python 3.12 before relying on `torch_mps` or `auto_torch`.
If the same checks still fail on macOS 26.x after that rebuild, treat it as an upstream PyTorch MPS compatibility issue on that OS release and expect `auto_torch` to fall back to CPU until PyTorch ships a fix.

## Parallel Backends

The new batch-oriented solver lives alongside the original adaptive solver and is exposed as `lpas.ParallelLPSolver`.

Supported backend names:

- `numpy_cpu`
- `torch_cpu`
- `torch_cuda`
- `torch_mps`
- `numba_cpu`
- `auto_torch`

Backend policy:

- `torch_cuda` raises if CUDA is unavailable.
- `torch_mps` raises if MPS is unavailable.
- `auto_torch` may choose MPS, then CPU.
- `auto_torch` never auto-selects CUDA.
- `auto_torch` never chooses Numba.
- `numba_cpu` is only used when explicitly requested.

This keeps benchmarking and debugging reproducible. CUDA remains available as an explicit backend, while `auto_torch` is reserved for Apple Silicon MPS when available and otherwise falls back to CPU.

## Running The Parallel Solver

Tiny 2D LP:

```bash
. .venv/bin/activate
python examples/gpu_tiny_2d_lp.py --backend numpy_cpu --samples 20000 --chunk-size 5000 --iterations 20 --seed 42
```

Apple Silicon MPS:

```bash
python examples/gpu_tiny_2d_lp.py --backend torch_mps --samples 100000 --chunk-size 25000 --iterations 40
```

CUDA:

```bash
python examples/gpu_random_lp.py --backend torch_cuda --samples 100000 --chunk-size 10000 --iterations 10
```

Explicit Torch CPU:

```bash
python examples/gpu_random_lp.py --backend torch_cpu --samples 50000 --chunk-size 5000 --iterations 10
```

Explicit Numba CPU:

```bash
python examples/compare_numpy_torch_numba.py --backend numba_cpu --samples 20000 --chunk-size 5000 --iterations 6
```

## Tests

Base test suite:

```bash
. .venv/bin/activate
pytest
```

Torch-specific tests require PyTorch to be installed. Numba-specific tests require Numba to be installed. When those optional packages are missing, the backend-specific tests are skipped rather than silently redirecting to another backend.

## Benchmarks

Run the backend benchmark matrix:

```bash
. .venv/bin/activate
python benchmarks/benchmark_backends.py --sizes small medium large --iterations 4
```

The benchmark reports:

- backend name
- device name
- elapsed time
- samples per second
- chunk memory estimate
- best score improvement

Do not expect GPU wins on tiny problems. The Torch GPU backends are meant for sufficiently large batch sizes where batched matrix multiplication dominates overhead.

## Vertex Polishing

After the adaptive sampling loop, the solver can run a bounded polishing pass:

1. Augment the primal system to `[A; -I] x <= [b; 0]`.
2. Compute soft activity scores from slack magnitudes.
3. Generate bounded active-set candidates of size `n`.
4. Solve the candidate equality systems.
5. Keep only reconstructed vertices that satisfy `Ax <= b` and `x >= 0` within tolerance.
6. Compare the best feasible reconstructed vertex against the raw best sampled point.

`SolverResult` now reports both the raw sampled point and the polished vertex when available, including whether polishing improved the objective and whether the polished point is certified primal-feasible.

## Running Tests

```bash
pytest tests/unit
pytest tests/integration
pytest tests/regression
pytest --cov=src/lpas
```

## Running Examples

```bash
python examples/tiny_2d_lp.py
python examples/random_vs_scipy.py
python scripts/run_benchmark.py
python examples/gpu_tiny_2d_lp.py --backend numpy_cpu
python examples/gpu_random_lp.py --backend torch_cpu
python examples/compare_numpy_torch_numba.py --backend all
```


Quick commands:

```bash
python scripts/run_toy_visuals.py --quick
python scripts/run_random_dense_benchmark.py --quick
python scripts/run_solver_hint_experiment.py --quick
python scripts/make_benchmark_plots.py
```

Default outputs:

- `outputs/figures/toy/`
- `outputs/figures/benchmarks/`
- `outputs/figures/solver_hints/`
- `outputs/benchmarks/random_dense/`
- `outputs/benchmarks/solver_hints/`

Key metrics:

- `active_set_recovery_accuracy`: Jaccard overlap between sampled active constraints and the HiGHS active set.
- `best_feasible_objective`: best primal-feasible objective seen during sampling.
- `raw_best_objective` and `polished_objective`: the best raw sampled objective and the best feasible polished vertex objective.
- `polishing_candidates_generated`, `vertices_reconstructed`, and `vertices_feasible`: reconstruction diagnostics for the bounded polishing pass.
- `best_primal_violation` and `best_dual_violation`: lowest violation norms observed across all candidates.
- `best_gap`: best duality-gap style score observed during sampling.
- `best_complementarity_error`: lowest complementarity error observed across all candidates.
- `time_to_identify_optimal_active_constraints`: first wall-clock time where active-set Jaccard reaches the configured recovery threshold.

See [EXPERIMENTS.md](/Users/nickgrebe/Projects/LP_Experiment/EXPERIMENTS.md) for the full experiment workflow, output files, and reporting conventions.

## Metrics

- `primal_violation`: total violation of `Ax <= b` plus `x >= 0`.
- `dual_violation`: total violation of `A^T y >= c` plus `y >= 0`.
- `gap`: `b^T y - c^T x`; the raw sampled gap is only a certificate when both primal and dual points are feasible.
- `complementarity_error`: `|| y ⊙ (b - Ax) ||_1`.
- `geometry_support`: kernel reward from nearby elite primal-dual samples.
- `active-set support`: frequency-based support for repeated active-set patterns among elite candidates.
- `samples_per_second`: throughput metric for the batch-oriented backend.

## Experimental Status

This prototype is intended to be correct, modular, and reproducible rather than aggressively optimized. The important scientific question is whether geometry-aware sampling discovers useful active sets faster than naive Monte Carlo, not whether it replaces HiGHS, Simplex, or Interior-Point methods.
Current limitations:

- The parallel solver currently targets dense LPs and dense batch evaluation.
- Active-set rewards are elite-frequency based; it does not compute full pairwise Jaccard over all candidates.
- Vertex polishing uses bounded candidate generation rather than exhaustive active-set enumeration.
- The explicit Numba backend prioritizes correctness and comparable metrics over aggressive kernel fusion.
- Torch MPS support is currently limited to `float32`.


## Note on LLms in the creation of this experiment
LLMs were used in the code creation and implementation as a quick way to be able to get to actually experimenting with my idea.
If interested I saved the md files that I created with the implementation guidelines and am happy to share those!
