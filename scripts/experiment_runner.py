import argparse
import os

import numpy as np
import random

from lookahead.runners.runners import ExpectedImprovementRunner
from lookahead.test_problems.bencher import BencherBenchmark
from lookahead.model.domain import TensorProductDomain, ClosedInterval
from lookahead.test_problems.metadata import BENCHMARK_METADATA
import logging

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
 
def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Bayesian Optimization Benchmark")

    parser.add_argument(
        "--benchmark",
        type=str,
        default="lasso-dna",
        help="Name of the benchmark to run (e.g., 'svm')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed",
    )

    return parser.parse_args()

def main():
    args = parse_args()
    seed_everything(args.seed)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )

    results_dir = f"results/{args.benchmark}"
    xs_path = f"{results_dir}/seed_{args.seed}_xs.npy"
    ys_path = f"{results_dir}/seed_{args.seed}_ys.npy"
    if os.path.exists(xs_path) or os.path.exists(ys_path):
            raise FileExistsError(
                f"Results already exist for benchmark={args.benchmark}, seed={args.seed}: {xs_path} / {ys_path}"
            )


    problem_metadata = BENCHMARK_METADATA[args.benchmark]
    bencher_benchmark = BencherBenchmark(name=args.benchmark)
    bencher_callable = bencher_benchmark.evaluate_points
    dim = problem_metadata.input_dim
    domain_bounds = [ClosedInterval(0, 1) for _ in range(dim)]
    search_space = TensorProductDomain(domain_bounds)

    runner = ExpectedImprovementRunner(search_space=search_space)
    initial_xs = np.load(f"init_points/{args.benchmark}/seed_{args.seed}_initial_points.npy")
    xs, ys = runner.run(f=bencher_callable, seed=args.seed, budget_minus_initialization=970, initial_xs=initial_xs)
    os.makedirs(results_dir, exist_ok=True)
    np.save(xs_path, xs)
    np.save(ys_path, ys)

if __name__ == "__main__":
    main()
