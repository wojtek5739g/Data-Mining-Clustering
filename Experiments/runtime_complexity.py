import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ClusteringAlgorithms import CLARANS, ROCK
from DataLoader import DataLoader
from Evaluation import evaluate_clustering
from Experiments.experiment_runner import percentile_theta


class RuntimeComplexityRunner:
    def __init__(self, loader=None, output_dir=None):
        project_root = Path(__file__).resolve().parents[1]
        self.loader = loader if loader is not None else DataLoader()
        self.output_dir = Path(output_dir) if output_dir else project_root / "results"

    def run(
        self,
        datasets=None,
        sample_sizes=None,
        repeats=3,
        seed=42,
        clarans_numlocal=4,
        clarans_maxneighbor=60,
        rock_theta=None,
        rock_theta_percentile=None,
    ):
        dataset_names = datasets if datasets is not None else self.loader.list_datasets()
        sample_sizes = sample_sizes if sample_sizes is not None else [100, 200, 400, 800]
        rows = []

        for dataset_name in dataset_names:
            dataset = self.loader.load(dataset_name, download=True)
            valid_sizes = [size for size in sample_sizes if size <= len(dataset.X)]

            for sample_size in valid_sizes:
                for repeat in range(repeats):
                    rng = np.random.default_rng(seed + repeat)
                    X_sample, y_sample = self._sample_dataset(dataset.X, dataset.y, sample_size, rng)
                    k = len(np.unique(y_sample))

                    rows.extend(
                        self._run_sample(
                            dataset_name=dataset.name,
                            X=X_sample,
                            y=y_sample,
                            sample_size=sample_size,
                            repeat=repeat,
                            k=k,
                            seed=seed,
                            clarans_numlocal=clarans_numlocal,
                            clarans_maxneighbor=clarans_maxneighbor,
                            rock_theta=rock_theta,
                            rock_theta_percentile=rock_theta_percentile,
                        )
                    )

        return pd.DataFrame(rows)

    def save(self, results, filename="runtime_complexity.csv"):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / filename
        results.to_csv(path, index=False)
        return path

    def run_and_save(self, **kwargs):
        filename = kwargs.pop("filename", "runtime_complexity.csv")
        results = self.run(**kwargs)
        path = self.save(results, filename)
        return results, path

    def _run_sample(
        self,
        dataset_name,
        X,
        y,
        sample_size,
        repeat,
        k,
        seed,
        clarans_numlocal,
        clarans_maxneighbor,
        rock_theta,
        rock_theta_percentile,
    ):
        resolved_theta = rock_theta
        if rock_theta_percentile is not None:
            resolved_theta = percentile_theta(X, rock_theta_percentile)

        experiments = [
            (
                "CLARANS",
                CLARANS(
                    k=k,
                    numlocal=clarans_numlocal,
                    maxneighbor=clarans_maxneighbor,
                    random_state=seed + repeat,
                ),
                {
                    "k": k,
                    "numlocal": clarans_numlocal,
                    "maxneighbor": clarans_maxneighbor,
                    "random_state": seed + repeat,
                },
            ),
            (
                "ROCK",
                ROCK(k=k, theta=resolved_theta),
                {
                    "k": k,
                    "theta": resolved_theta,
                    "theta_percentile": rock_theta_percentile,
                },
            ),
        ]

        return [
            self._run_algorithm(
                dataset_name=dataset_name,
                X=X,
                y=y,
                sample_size=sample_size,
                repeat=repeat,
                algorithm_name=algorithm_name,
                algorithm=algorithm,
                parameters=parameters,
            )
            for algorithm_name, algorithm, parameters in experiments
        ]

    def _run_algorithm(
        self,
        dataset_name,
        X,
        y,
        sample_size,
        repeat,
        algorithm_name,
        algorithm,
        parameters,
    ):
        start = time.perf_counter()
        model = algorithm.fit(X)
        runtime_seconds = time.perf_counter() - start

        evaluation = evaluate_clustering(
            X,
            model.labels_,
            y,
            dataset=dataset_name,
            algorithm=algorithm_name,
        )
        row = evaluation.as_dict()
        row["sample_size"] = sample_size
        row["repeat"] = repeat
        row["runtime_seconds"] = runtime_seconds
        row["parameters"] = json.dumps(parameters, sort_keys=True)
        return row

    def _sample_dataset(self, X, y, sample_size, rng):
        if sample_size >= len(X):
            return X.copy(), y.copy()

        selected = []
        available = np.arange(len(X))
        labels = np.unique(y)

        if sample_size >= len(labels):
            for label in labels:
                label_indices = np.flatnonzero(y == label)
                selected.append(int(rng.choice(label_indices)))

        remaining_slots = sample_size - len(selected)
        remaining = np.setdiff1d(available, selected, assume_unique=False)
        selected.extend(rng.choice(remaining, size=remaining_slots, replace=False).tolist())

        selected = np.array(selected)
        rng.shuffle(selected)
        return X[selected], y[selected]


def _parse_int_list(values):
    if values is None:
        return None
    return [int(value) for value in values]


def main():
    parser = argparse.ArgumentParser(
        description="Run runtime complexity experiments for CLARANS and ROCK."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to run. By default all configured benchmark datasets are used.",
    )
    parser.add_argument("--sample-sizes", nargs="+", default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clarans-local", type=int, default=4)
    parser.add_argument("--clarans-neighbor", type=int, default=60)
    parser.add_argument("--rock-theta", type=float, default=None)
    parser.add_argument("--rock-theta-percentile", type=float, default=None)
    parser.add_argument("--output", default="runtime_complexity.csv")
    args = parser.parse_args()

    runner = RuntimeComplexityRunner()
    results, path = runner.run_and_save(
        datasets=args.datasets,
        sample_sizes=_parse_int_list(args.sample_sizes),
        repeats=args.repeats,
        seed=args.seed,
        clarans_numlocal=args.clarans_local,
        clarans_maxneighbor=args.clarans_neighbor,
        rock_theta=args.rock_theta,
        rock_theta_percentile=args.rock_theta_percentile,
        filename=args.output,
    )

    print(results.to_string(index=False))
    print(f"\nSaved runtime complexity results to: {path}")


if __name__ == "__main__":
    main()
