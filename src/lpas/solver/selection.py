from __future__ import annotations

from lpas.solver.result import ArchiveEntry


def archive_entry_key(entry: ArchiveEntry) -> tuple[bool, float, float]:
    """Keep feasible incumbents; compare infeasible points by fixed KKT residual.

    Within-batch rank scores train the sampler, but are not comparable across
    iterations and must not decide which old candidates survive.
    """
    return (
        entry.primal_feasible,
        entry.primal_objective if entry.primal_feasible else -entry.kkt_residual,
        -entry.kkt_residual,
    )
