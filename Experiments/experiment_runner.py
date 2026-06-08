import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from ClusteringAlgorithms import CLARANS, ROCK
from DataLoader import DataLoader
from Evaluation import evaluate_clustering


class ExperimentRunner:
    def __init__(self, loader=None, output_dir=None):
        project_root = Path(__file__).resolve().parents[1]
        self.loader = loader if loader is not None else DataLoader()
        self.output_dir = Path(output_dir) if output_dir else project_root / "results"

    def run(
        self,
        datasets=None,
        seed=42,
        clarans_numlocal=4,
        clarans_maxneighbor=60,
        rock_theta=None,
        rock_theta_percentile=None,
        standardize=True,
        download=True,
    ):
        dataset_names = datasets if datasets is not None else self.loader.list_datasets()
        rows = []

        for dataset_name in dataset_names:
            dataset = self.loader.load(
                dataset_name,
                standardize=standardize,
                download=download,
            )
            k = len(np.unique(dataset.y))
            rows.extend(
                self._run_dataset(
                    dataset,
                    k,
                    seed,
                    clarans_numlocal,
                    clarans_maxneighbor,
                    rock_theta,
                    rock_theta_percentile,
                )
            )

        return pd.DataFrame(rows)

    def save(self, results, filename="evaluation_results.csv"):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / filename
        results.to_csv(path, index=False)
        return path

    def run_and_save(self, **kwargs):
        filename = kwargs.pop("filename", "evaluation_results.csv")
        results = self.run(**kwargs)
        path = self.save(results, filename)
        return results, path

    def _run_dataset(
        self,
        dataset,
        k,
        seed,
        clarans_numlocal,
        clarans_maxneighbor,
        rock_theta,
        rock_theta_percentile,
    ):
        rock_resolved_theta = rock_theta
        if rock_theta_percentile is not None:
            rock_resolved_theta = percentile_theta(dataset.X, rock_theta_percentile)

        experiments = [
            (
                "CLARANS",
                CLARANS(
                    k=k,
                    numlocal=clarans_numlocal,
                    maxneighbor=clarans_maxneighbor,
                    random_state=seed,
                ),
                {
                    "k": k,
                    "numlocal": clarans_numlocal,
                    "maxneighbor": clarans_maxneighbor,
                    "random_state": seed,
                },
            ),
            (
                "ROCK",
                ROCK(k=k, theta=rock_resolved_theta),
                {
                    "k": k,
                    "theta": rock_resolved_theta,
                    "theta_percentile": rock_theta_percentile,
                },
            ),
        ]

        return [
            self._run_algorithm(dataset, algorithm_name, algorithm, parameters)
            for algorithm_name, algorithm, parameters in experiments
        ]

    def _run_algorithm(self, dataset, algorithm_name, algorithm, parameters):
        start = time.perf_counter()
        model = algorithm.fit(dataset.X)
        runtime_seconds = time.perf_counter() - start

        evaluation = evaluate_clustering(
            dataset.X,
            model.labels_,
            dataset.y,
            dataset=dataset.name,
            algorithm=algorithm_name,
        )
        row = evaluation.as_dict()
        row["runtime_seconds"] = runtime_seconds
        row["parameters"] = json.dumps(parameters, sort_keys=True)
        row["source_file"] = str(dataset.path)
        return row


def percentile_theta(X, percentile):
    distances = pdist(X)
    distances = distances[distances > 0]
    if distances.size == 0:
        return 0.0
    return float(np.percentile(distances, percentile))


def main():
    parser = argparse.ArgumentParser(
        description="Run clustering experiments and save evaluation metrics."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset names to run. By default all configured benchmark datasets are used.",
    )
    parser.add_argument("--output", default="evaluation_results.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clarans-local", type=int, default=4)
    parser.add_argument("--clarans-neighbor", type=int, default=60)
    parser.add_argument("--rock-theta", type=float, default=None)
    parser.add_argument("--rock-theta-percentile", type=float, default=None)
    args = parser.parse_args()

    runner = ExperimentRunner()
    results, path = runner.run_and_save(
        datasets=args.datasets,
        seed=args.seed,
        clarans_numlocal=args.clarans_local,
        clarans_maxneighbor=args.clarans_neighbor,
        rock_theta=args.rock_theta,
        rock_theta_percentile=args.rock_theta_percentile,
        filename=args.output,
    )

    print(results.to_string(index=False))
    print(f"\nSaved results to: {path}")


if __name__ == "__main__":
    main()
