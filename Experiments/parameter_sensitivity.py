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


class ParameterSensitivityRunner:
    def __init__(self, loader=None, output_dir=None):
        project_root = Path(__file__).resolve().parents[1]
        self.loader = loader if loader is not None else DataLoader()
        self.output_dir = Path(output_dir) if output_dir else project_root / "results"

    def run_clarans(
        self,
        datasets=None,
        numlocal_values=None,
        maxneighbor_values=None,
        seed=42,
    ):
        dataset_names = datasets if datasets is not None else self.loader.list_datasets()
        numlocal_values = numlocal_values if numlocal_values is not None else [1, 3, 5]
        maxneighbor_values = (
            maxneighbor_values if maxneighbor_values is not None else [20, 60, 100]
        )

        rows = []
        for dataset_name in dataset_names:
            dataset = self.loader.load(dataset_name, download=True)
            k = len(np.unique(dataset.y))

            for numlocal in numlocal_values:
                for maxneighbor in maxneighbor_values:
                    parameters = {
                        "k": k,
                        "numlocal": numlocal,
                        "maxneighbor": maxneighbor,
                        "random_state": seed,
                    }
                    algorithm = CLARANS(
                        k=k,
                        numlocal=numlocal,
                        maxneighbor=maxneighbor,
                        random_state=seed,
                    )
                    rows.append(
                        self._run_algorithm(
                            dataset=dataset,
                            algorithm_name="CLARANS",
                            algorithm=algorithm,
                            parameters=parameters,
                        )
                    )

        return pd.DataFrame(rows)

    def run_rock(
        self,
        datasets=None,
        theta_percentiles=None,
    ):
        dataset_names = datasets if datasets is not None else self.loader.list_datasets()
        theta_percentiles = (
            theta_percentiles if theta_percentiles is not None else [1, 2, 3, 5, 7, 10]
        )

        rows = []
        for dataset_name in dataset_names:
            dataset = self.loader.load(dataset_name, download=True)
            k = len(np.unique(dataset.y))

            for theta_percentile in theta_percentiles:
                theta = percentile_theta(dataset.X, theta_percentile)
                parameters = {
                    "k": k,
                    "theta": theta,
                    "theta_percentile": theta_percentile,
                }
                algorithm = ROCK(k=k, theta=theta)
                rows.append(
                    self._run_algorithm(
                        dataset=dataset,
                        algorithm_name="ROCK",
                        algorithm=algorithm,
                        parameters=parameters,
                    )
                )

        return pd.DataFrame(rows)

    def run_all(
        self,
        datasets=None,
        numlocal_values=None,
        maxneighbor_values=None,
        theta_percentiles=None,
        seed=42,
    ):
        clarans_results = self.run_clarans(
            datasets=datasets,
            numlocal_values=numlocal_values,
            maxneighbor_values=maxneighbor_values,
            seed=seed,
        )
        rock_results = self.run_rock(
            datasets=datasets,
            theta_percentiles=theta_percentiles,
        )
        return clarans_results, rock_results

    def save(self, results, filename):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / filename
        results.to_csv(path, index=False)
        return path

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
        row.update(parameters)
        return row


def _parse_int_list(values):
    if values is None:
        return None
    return [int(value) for value in values]


def _parse_float_list(values):
    if values is None:
        return None
    return [float(value) for value in values]


def main():
    parser = argparse.ArgumentParser(
        description="Run CLARANS and ROCK parameter sensitivity experiments."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to run. By default all configured benchmark datasets are used.",
    )
    parser.add_argument(
        "--algorithm",
        choices=["all", "clarans", "rock"],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--numlocal-values", nargs="+", default=None)
    parser.add_argument("--maxneighbor-values", nargs="+", default=None)
    parser.add_argument("--theta-percentiles", nargs="+", default=None)
    args = parser.parse_args()

    runner = ParameterSensitivityRunner()
    saved_paths = []

    if args.algorithm in {"all", "clarans"}:
        clarans_results = runner.run_clarans(
            datasets=args.datasets,
            numlocal_values=_parse_int_list(args.numlocal_values),
            maxneighbor_values=_parse_int_list(args.maxneighbor_values),
            seed=args.seed,
        )
        saved_paths.append(
            runner.save(clarans_results, "clarans_parameter_sensitivity.csv")
        )
        print(clarans_results.to_string(index=False))

    if args.algorithm in {"all", "rock"}:
        rock_results = runner.run_rock(
            datasets=args.datasets,
            theta_percentiles=_parse_float_list(args.theta_percentiles),
        )
        saved_paths.append(runner.save(rock_results, "rock_parameter_sensitivity.csv"))
        print(rock_results.to_string(index=False))

    print("\nSaved parameter sensitivity results:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
