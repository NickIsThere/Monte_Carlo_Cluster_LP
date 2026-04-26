"""Experiment generators and benchmark helpers."""

from lpas.experiments.certificate_validation import write_certificate_validation_outputs
from lpas.experiments.corner_discovery import (
    CornerDiscoveryRecord,
    run_corner_discovery_experiment,
    write_corner_discovery_outputs,
)
from lpas.experiments.gpu_throughput import (
    BENCHMARK_DIMENSIONS,
    GPUThroughputRecord,
    benchmark_candidate_throughput,
    run_gpu_throughput_suite,
    write_gpu_throughput_outputs,
)
from lpas.experiments.scaling_by_dimension import (
    ScalingRecord,
    run_scaling_experiment,
    write_scaling_outputs,
)
from lpas.experiments.solver_seeding_total_time import (
    SolverSeedingRecord,
    benchmark_solver_seeding_problem,
    run_solver_seeding_total_time_benchmark,
    write_solver_seeding_outputs,
)

__all__ = [
    "BENCHMARK_DIMENSIONS",
    "CornerDiscoveryRecord",
    "GPUThroughputRecord",
    "ScalingRecord",
    "SolverSeedingRecord",
    "benchmark_candidate_throughput",
    "benchmark_solver_seeding_problem",
    "run_corner_discovery_experiment",
    "run_gpu_throughput_suite",
    "run_scaling_experiment",
    "run_solver_seeding_total_time_benchmark",
    "write_certificate_validation_outputs",
    "write_corner_discovery_outputs",
    "write_gpu_throughput_outputs",
    "write_scaling_outputs",
    "write_solver_seeding_outputs",
]
