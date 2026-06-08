import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ClusteringAlgorithms import CLARANS, ROCK
from DataLoader import DataLoader
from Experiments.experiment_runner import percentile_theta


class ResultsVisualizer:
    EXTERNAL_METRICS = ["ari", "nmi", "purity"]
    INTERNAL_METRICS = ["silhouette", "calinski_harabasz", "davies_bouldin"]

    def __init__(self, results_path=None, output_dir=None, loader=None):
        project_root = Path(__file__).resolve().parents[1]
        self.results_path = (
            Path(results_path)
            if results_path
            else project_root / "results" / "evaluation_results.csv"
        )
        self.output_dir = (
            Path(output_dir) if output_dir else project_root / "results" / "plots"
        )
        self.loader = loader if loader is not None else DataLoader()

    def load_results(self):
        if not self.results_path.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_path}")

        results = pd.read_csv(self.results_path)
        self._validate_results(results)
        return results

    def create_all_plots(self):
        results = self.load_results()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        plot_paths = [
            self.plot_external_metrics(results),
            self.plot_internal_metrics(results),
            self.plot_runtime(results),
        ]
        return plot_paths

    def create_cluster_plots(
        self,
        datasets=None,
        seed=42,
        clarans_numlocal=4,
        clarans_maxneighbor=60,
        rock_theta=None,
        rock_theta_percentile=None,
    ):
        dataset_names = datasets if datasets is not None else self.loader.list_datasets()
        paths = []

        for dataset_name in dataset_names:
            dataset = self.loader.load(dataset_name, download=True)
            if dataset.X.shape[1] < 2:
                raise ValueError(f"Dataset '{dataset.name}' must have at least two features.")

            k = len(np.unique(dataset.y))
            resolved_theta = rock_theta
            if rock_theta_percentile is not None:
                resolved_theta = percentile_theta(dataset.X, rock_theta_percentile)

            clarans = CLARANS(
                k=k,
                numlocal=clarans_numlocal,
                maxneighbor=clarans_maxneighbor,
                random_state=seed,
            ).fit(dataset.X)
            rock = ROCK(k=k, theta=resolved_theta).fit(dataset.X)

            paths.append(
                self.plot_dataset_clusters(
                    dataset=dataset,
                    clarans_labels=clarans.labels_,
                    rock_labels=rock.labels_,
                    clarans_medoids=clarans.medoids_vectors_,
                )
            )

        return paths

    def plot_dataset_clusters(
        self,
        dataset,
        clarans_labels,
        rock_labels,
        clarans_medoids=None,
    ):
        figure, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 4.5), squeeze=False)
        x_values = dataset.X[:, 0]
        y_values = dataset.X[:, 1]

        panels = [
            ("Ground Truth", dataset.y, None),
            ("CLARANS", clarans_labels, clarans_medoids),
            ("ROCK", rock_labels, None),
        ]

        for axis, (title, labels, medoids) in zip(axes[0], panels):
            scatter = axis.scatter(
                x_values,
                y_values,
                c=labels,
                cmap="tab10",
                s=24,
                alpha=0.85,
                edgecolors="none",
            )
            if medoids is not None:
                axis.scatter(
                    medoids[:, 0],
                    medoids[:, 1],
                    c="black",
                    marker="X",
                    s=120,
                    linewidths=1,
                    label="Medoids",
                )
                axis.legend(loc="best")

            axis.set_title(title)
            axis.set_xlabel("Feature 1")
            axis.set_ylabel("Feature 2")
            axis.grid(alpha=0.2)
            axis.set_aspect("equal", adjustable="box")

        figure.suptitle(f"Cluster Assignments: {dataset.name}")
        figure.tight_layout()
        return self._save_figure(figure, f"clusters_{dataset.name}.png")

    def plot_external_metrics(self, results):
        return self._plot_metric_group(
            results=results,
            metrics=self.EXTERNAL_METRICS,
            title="External Metrics",
            ylabel="Score",
            filename="external_metrics.png",
        )

    def plot_internal_metrics(self, results):
        return self._plot_metric_group(
            results=results,
            metrics=self.INTERNAL_METRICS,
            title="Internal Metrics",
            ylabel="Metric value",
            filename="internal_metrics.png",
        )

    def plot_runtime(self, results):
        figure, axis = plt.subplots(figsize=(9, 5))
        labels = self._experiment_labels(results)
        axis.bar(labels, results["runtime_seconds"], color=self._algorithm_colors(results))
        axis.set_title("Runtime Comparison")
        axis.set_ylabel("Runtime [seconds]")
        axis.set_xlabel("Dataset / Algorithm")
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()

        return self._save_figure(figure, "runtime_comparison.png")

    def _plot_metric_group(self, results, metrics, title, ylabel, filename):
        existing_metrics = [metric for metric in metrics if metric in results.columns]
        if not existing_metrics:
            raise ValueError(f"No requested metrics found for plot: {metrics}")

        figure, axes = plt.subplots(
            nrows=1,
            ncols=len(existing_metrics),
            figsize=(5 * len(existing_metrics), 5),
            squeeze=False,
        )
        labels = self._experiment_labels(results)
        colors = self._algorithm_colors(results)

        for axis, metric in zip(axes[0], existing_metrics):
            axis.bar(labels, results[metric], color=colors)
            axis.set_title(metric.replace("_", " ").title())
            axis.set_ylabel(ylabel)
            axis.tick_params(axis="x", rotation=35)
            axis.grid(axis="y", alpha=0.25)

        figure.suptitle(title)
        figure.tight_layout()
        return self._save_figure(figure, filename)

    def _save_figure(self, figure, filename):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / filename
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        return path

    def _experiment_labels(self, results):
        return results["dataset"] + " / " + results["algorithm"]

    def _algorithm_colors(self, results):
        palette = {
            "CLARANS": "#4C78A8",
            "ROCK": "#F58518",
        }
        return [palette.get(algorithm, "#72B7B2") for algorithm in results["algorithm"]]

    def _validate_results(self, results):
        required_columns = {"dataset", "algorithm", "runtime_seconds"}
        missing = required_columns - set(results.columns)
        if missing:
            missing_names = ", ".join(sorted(missing))
            raise ValueError(f"Results file is missing required columns: {missing_names}")


def main():
    parser = argparse.ArgumentParser(
        description="Create plots from experiment result metrics."
    )
    parser.add_argument(
        "--results",
        default=None,
        help="Path to evaluation_results.csv. Defaults to results/evaluation_results.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where plots will be saved. Defaults to results/plots.",
    )
    parser.add_argument(
        "--clusters",
        action="store_true",
        help="Also plot cluster assignments for each selected dataset.",
    )
    parser.add_argument(
        "--only-clusters",
        action="store_true",
        help="Create only cluster assignment plots, skipping metric plots.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to use for cluster plots. By default all configured datasets are used.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clarans-local", type=int, default=4)
    parser.add_argument("--clarans-neighbor", type=int, default=60)
    parser.add_argument("--rock-theta", type=float, default=None)
    parser.add_argument("--rock-theta-percentile", type=float, default=None)
    args = parser.parse_args()

    visualizer = ResultsVisualizer(results_path=args.results, output_dir=args.output_dir)
    plot_paths = []

    if not args.only_clusters:
        plot_paths.extend(visualizer.create_all_plots())
    if args.clusters or args.only_clusters:
        plot_paths.extend(
            visualizer.create_cluster_plots(
                datasets=args.datasets,
                seed=args.seed,
                clarans_numlocal=args.clarans_local,
                clarans_maxneighbor=args.clarans_neighbor,
                rock_theta=args.rock_theta,
                rock_theta_percentile=args.rock_theta_percentile,
            )
        )

    print("Saved plots:")
    for path in plot_paths:
        print(path)


if __name__ == "__main__":
    main()
