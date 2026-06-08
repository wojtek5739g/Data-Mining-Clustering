import argparse
import time

import numpy as np

from ClusteringAlgorithms.clustering_module import CLARANS, ROCK
from DataLoader.data_loader import DataLoader
from Evaluation.evaluation_module import evaluate_clustering
from Experiments.experiment_runner import percentile_theta


def run_algorithm(name, algorithm, X, y):
    start = time.perf_counter()
    model = algorithm.fit(X)
    elapsed = time.perf_counter() - start

    result = evaluate_clustering(X, model.labels_, y, algorithm=name)
    values = result.as_dict()
    values["time"] = elapsed
    return values


def print_result(dataset_name, result):
    print(
        f"{dataset_name:<8} {result['algorithm']:<8} "
        f"k={result['n_clusters']:<3} "
        f"ARI={result['ari']:.3f} "
        f"NMI={result['nmi']:.3f} "
        f"purity={result['purity']:.3f} "
        f"sil={_format_metric(result['silhouette'])} "
        f"time={result['time']:.2f}s"
    )


def _format_metric(value):
    return "N/A" if value is None else f"{value:.3f}"


def main():
    parser = argparse.ArgumentParser(
        description="Run CLARANS and ROCK on deric/clustering-benchmark datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["jain", "flame", "spiral"],
        help="Dataset names to run. Available defaults: jain flame spiral.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for CLARANS.")
    parser.add_argument("--clarans-local", type=int, default=4)
    parser.add_argument("--clarans-neighbor", type=int, default=60)
    parser.add_argument(
        "--rock-theta",
        type=float,
        default=None,
        help="Fixed ROCK distance threshold. By default ROCK estimates one automatically.",
    )
    parser.add_argument(
        "--rock-theta-percentile",
        type=float,
        default=None,
        help="Use this percentile of pairwise distances as ROCK theta.",
    )
    args = parser.parse_args()

    loader = DataLoader()
    print("Dataset  Algorithm Result")
    print("-" * 54)

    for dataset_name in args.datasets:
        dataset = loader.load(dataset_name, download=True)
        k = len(np.unique(dataset.y))
        rock_theta = args.rock_theta
        if args.rock_theta_percentile is not None:
            rock_theta = percentile_theta(dataset.X, args.rock_theta_percentile)

        algorithms = [
            (
                "CLARANS",
                CLARANS(
                    k=k,
                    numlocal=args.clarans_local,
                    maxneighbor=args.clarans_neighbor,
                    random_state=args.seed,
                ),
            ),
            ("ROCK", ROCK(k=k, theta=rock_theta)),
        ]

        for algorithm_name, algorithm in algorithms:
            result = run_algorithm(algorithm_name, algorithm, dataset.X, dataset.y)
            print_result(dataset.name, result)


if __name__ == "__main__":
    main()
