from __future__ import annotations

import cProfile
import pstats

from lpas.experiments.generators import tiny_known_lp
from lpas.solver.adaptive_solver import AdaptiveLPSolver


def main() -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    AdaptiveLPSolver().solve(tiny_known_lp())
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats(20)


if __name__ == "__main__":
    main()
