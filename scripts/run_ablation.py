from __future__ import annotations

from lpas.experiments.runners import run_ablation_study
from lpas.utils.config import ScoringConfig, SolverConfig


def main() -> None:
    base = SolverConfig()
    configs = {
        "full": base,
        "no_geometry": SolverConfig(scoring=ScoringConfig(w_geo=0.0, w_active=0.0)),
        "no_dual_terms": SolverConfig(scoring=ScoringConfig(w_gap=0.0, w_dviol=0.0, w_comp=0.0)),
    }
    results = run_ablation_study(configs)
    for name, result in results.items():
        print(name, {"status": result.status.value, "objective": result.best_primal_objective, "gap": result.best_gap})


if __name__ == "__main__":
    main()
