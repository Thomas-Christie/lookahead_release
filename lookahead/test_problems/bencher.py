import numpy as np
from lookahead.test_problems.metadata import BenchmarkMetadata, BENCHMARK_METADATA
from bencherscaffold.client import BencherClient
from bencherscaffold.protoclasses.bencher_pb2 import Value

class BencherBenchmark:
    """Benchmark implementation using Bencher.

    Args:
        name: Name of the benchmark.
        metadata: Benchmark metadata. If None, will be looked up in BENCHMARK_METADATA
          using the name.
        address: Address of the Bencher server. Can be a hostname (e.g. "127.0.0.1")
          or a Unix socket path (e.g. "unix:///path/to/bencher.sock").
        port: Port of the Bencher server. Ignored when address is a Unix socket.
    """

    def __init__(
        self,
        name: str,
        metadata: BenchmarkMetadata | None = None,
        address: str = "127.0.0.1",
        port: int = 50051,
    ) -> None:
        if metadata is None:
            metadata = BENCHMARK_METADATA.get(name)
            if metadata is None:
                msg = f"Metadata not found for benchmark '{name}'."
                raise ValueError(msg)
        self.name = name
        self.metadata = metadata
        self.bencher_client = BencherClient(address=address, port=port)

    def evaluate_points(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the benchmark at the given points.

        Args:
            points: (N, D) array of points to evaluate.

        Returns:
            (N, ) shaped array of evaluations at the given points.
        """
        evaluations = []
        for point in points:
            bencher_point = [
                Value(type=self.metadata.input_type, value=float(d)) for d in point
            ]
            value = self.bencher_client.evaluate_point(
                benchmark_name=self.name,
                point=bencher_point,
            )
            evaluations.append(value)
        return np.array(evaluations)  # keep as minimisation problem