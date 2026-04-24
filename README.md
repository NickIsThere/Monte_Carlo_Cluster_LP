# Geometry-Aware LP Adaptive Sampling

This repository contains a research prototype for geometry-aware primal-dual adaptive sampling for linear programming. It samples primal and dual candidates, scores them with feasibility and duality metrics, tracks recurring active-set patterns, and can compare or hand off diagnostics to SciPy/HiGHS.

## What It Is

- A Python package built around dense LPs of the form `Ax <= b`, `x >= 0`.
- A research prototype for testing whether active-set geometry helps adaptive sampling find useful LP structure.
- A hybrid tool that can surface candidate solutions, dual certificates, and active-set hints alongside a classical solver baseline.

## What It Is Not

- It is not a replacement for mature LP solvers such as HiGHS.
- It does not implement true HiGHS warm-start support in this version.
- It is not optimized for large sparse LPs or production-scale workloads.


## Package Layout

- `src/lpas/core`: LP representation, feasibility, primal-dual metrics, active sets, certificates, scoring.
- `src/lpas/sampling`: Gaussian adaptive sampler, truncated feasibility-aware wrapper, simplex-style experimental sampler.
- `src/lpas/geometry`: Jaccard similarity, cluster support, kernel density rewards.
- `src/lpas/solver`: adaptive loop, warm-start reconstruction, SciPy handoff, result objects.
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
```

## Metrics

- `primal_violation`: total violation of `Ax <= b` plus `x >= 0`.
- `dual_violation`: total violation of `A^T y >= c` plus `y >= 0`.
- `gap`: `b^T y - c^T x`; only meaningful as a certificate when both primal and dual points are feasible.
- `complementarity_error`: `|| y ⊙ (b - Ax) ||_1`.
- `geometry_support`: kernel reward from nearby elite primal-dual samples.
- `active-set support`: frequency-based support for repeated active-set patterns among elite candidates.

## Experimental Status

This prototype is intended to be correct, modular, and reproducible rather than aggressively optimized. The important scientific question is whether geometry-aware sampling discovers useful active sets faster than naive Monte Carlo, not whether it beats HiGHS directly.

## Note on LLms in the creation of this experiment
LLMs were used in the code creation and implementation as a quick way to be able to get to actually experimenting with my idea.
If interested I saved the md files that I created with the implementation guidelines and am happy to share those!
