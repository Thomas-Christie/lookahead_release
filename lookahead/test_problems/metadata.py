"""Bencher benchmarks metadata."""

from dataclasses import dataclass

from bencherscaffold.protoclasses.bencher_pb2 import ValueType


@dataclass(frozen=True)  # frozen=True makes them immutable
class BenchmarkMetadata:
    """Metadata that benchmarks must provide."""

    input_dim: int
    input_type: int  # protobuf ValueType enum (not a PEP-compliant type)
    default_n_initial_points: int
    default_n_total_evaluations: int


BENCHMARK_METADATA: dict[str, BenchmarkMetadata] = {
    "rover": BenchmarkMetadata(
        input_dim=60,
        input_type=ValueType.CONTINUOUS,
        default_n_initial_points=30,
        default_n_total_evaluations=1000,
    ),
    "mopta08": BenchmarkMetadata(
        input_dim=124,
        input_type=ValueType.CONTINUOUS,
        default_n_initial_points=30,
        default_n_total_evaluations=1000,
    ),
    "lasso-dna": BenchmarkMetadata(
        input_dim=180,
        input_type=ValueType.CONTINUOUS,
        default_n_initial_points=30,
        default_n_total_evaluations=1000,
    ),
    "svm": BenchmarkMetadata(
        input_dim=388,
        input_type=ValueType.CONTINUOUS,
        default_n_initial_points=30,
        default_n_total_evaluations=1000,
    ),
    "mujoco-ant": BenchmarkMetadata(
        input_dim=888,
        input_type=ValueType.CONTINUOUS,
        default_n_initial_points=30,
        default_n_total_evaluations=1000,
    ),
    "mujoco-humanoid": BenchmarkMetadata(
        input_dim=6392,
        input_type=ValueType.CONTINUOUS,
        default_n_initial_points=30,
        default_n_total_evaluations=1000,
    ),
}

