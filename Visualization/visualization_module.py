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
        self.project_root = Path(__file__).resolve().parents[1]
        self.results_path = (
            Path(results_path)
            if results_path
            else self.project_root / "results" / "evaluation_results.csv"
        )
        self.output_dir = (
            Path(output_dir) if output_dir else self.project_root / "results" / "plots"
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

    def create_sensitivity_plots(self, clarans_path=None, rock_path=None):
        clarans_results = self._load_sensitivity_results(
            clarans_path,
            "clarans_parameter_sensitivity.csv",
            {"dataset", "numlocal", "maxneighbor", "ari", "runtime_seconds"},
        )
        rock_results = self._load_sensitivity_results(
            rock_path,
            "rock_parameter_sensitivity.csv",
            {"dataset", "theta_percentile", "ari", "runtime_seconds"},
        )

        return [
            self.plot_clarans_sensitivity_heatmap(
                clarans_results,
                metric="ari",
                title="CLARANS Parameter Sensitivity: ARI",
                filename="clarans_sensitivity_ari.png",
            ),
            self.plot_clarans_sensitivity_heatmap(
                clarans_results,
                metric="runtime_seconds",
                title="CLARANS Parameter Sensitivity: Runtime",
                filename="clarans_sensitivity_runtime.png",
                value_format="{:.3f}",
            ),
            self.plot_rock_sensitivity_lines(
                rock_results,
                metric="ari",
                title="ROCK Theta Sensitivity: ARI",
                ylabel="ARI",
                filename="rock_sensitivity_ari.png",
            ),
            self.plot_rock_sensitivity_lines(
                rock_results,
                metric="runtime_seconds",
                title="ROCK Theta Sensitivity: Runtime",
                ylabel="Runtime [seconds]",
                filename="rock_sensitivity_runtime.png",
            ),
        ]

    def create_runtime_complexity_plots(self, runtime_path=None):
        results = self._load_runtime_complexity_results(runtime_path)
        return [
            self.plot_runtime_complexity(
                results,
                filename="runtime_complexity.png",
                log_scale=False,
            ),
            self.plot_runtime_complexity(
                results,
                filename="runtime_complexity_log.png",
                log_scale=True,
            ),
        ]

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

    def plot_clarans_sensitivity_heatmap(
        self,
        results,
        metric,
        title,
        filename,
        value_format="{:.2f}",
    ):
        datasets = sorted(results["dataset"].unique())
        figure, axes = plt.subplots(
            nrows=1,
            ncols=len(datasets),
            figsize=(5 * len(datasets), 4.5),
            squeeze=False,
        )

        for axis, dataset_name in zip(axes[0], datasets):
            subset = results[results["dataset"] == dataset_name]
            pivot = subset.pivot_table(
                index="numlocal",
                columns="maxneighbor",
                values=metric,
                aggfunc="mean",
            ).sort_index().sort_index(axis=1)

            image = axis.imshow(pivot.values, cmap="viridis", aspect="auto")
            axis.set_title(dataset_name)
            axis.set_xlabel("maxneighbor")
            axis.set_ylabel("numlocal")
            axis.set_xticks(range(len(pivot.columns)))
            axis.set_xticklabels(pivot.columns)
            axis.set_yticks(range(len(pivot.index)))
            axis.set_yticklabels(pivot.index)

            for row_index, numlocal in enumerate(pivot.index):
                for column_index, maxneighbor in enumerate(pivot.columns):
                    value = pivot.loc[numlocal, maxneighbor]
                    axis.text(
                        column_index,
                        row_index,
                        value_format.format(value),
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8,
                    )

            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

        figure.suptitle(title)
        figure.tight_layout()
        return self._save_figure(figure, filename)

    def plot_rock_sensitivity_lines(self, results, metric, title, ylabel, filename):
        figure, axis = plt.subplots(figsize=(8, 5))

        for dataset_name in sorted(results["dataset"].unique()):
            subset = results[results["dataset"] == dataset_name].sort_values(
                "theta_percentile"
            )
            axis.plot(
                subset["theta_percentile"],
                subset[metric],
                marker="o",
                linewidth=2,
                label=dataset_name,
            )

        axis.set_title(title)
        axis.set_xlabel("theta percentile")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(title="Dataset")
        figure.tight_layout()
        return self._save_figure(figure, filename)

    def plot_runtime_complexity(self, results, filename, log_scale=False):
        summary = (
            results.groupby(["dataset", "algorithm", "sample_size"], as_index=False)
            .agg(
                runtime_mean=("runtime_seconds", "mean"),
                runtime_std=("runtime_seconds", "std"),
            )
            .fillna({"runtime_std": 0.0})
        )
        datasets = sorted(summary["dataset"].unique())
        figure, axes = plt.subplots(
            nrows=1,
            ncols=len(datasets),
            figsize=(5 * len(datasets), 4.5),
            squeeze=False,
        )

        for axis, dataset_name in zip(axes[0], datasets):
            subset = summary[summary["dataset"] == dataset_name]
            for algorithm in sorted(subset["algorithm"].unique()):
                algorithm_data = subset[subset["algorithm"] == algorithm].sort_values(
                    "sample_size"
                )
                x_values = algorithm_data["sample_size"].to_numpy()
                y_values = algorithm_data["runtime_mean"].to_numpy()
                error = algorithm_data["runtime_std"].to_numpy()

                axis.plot(
                    x_values,
                    y_values,
                    marker="o",
                    linewidth=2,
                    label=algorithm,
                )
                axis.fill_between(
                    x_values,
                    np.maximum(y_values - error, 0),
                    y_values + error,
                    alpha=0.15,
                )

            axis.set_title(dataset_name)
            axis.set_xlabel("Number of objects (n)")
            axis.set_ylabel("Runtime [seconds]")
            axis.grid(alpha=0.25)
            axis.legend()
            if log_scale:
                axis.set_yscale("log")

        title = "Runtime Complexity"
        if log_scale:
            title += " (log scale)"
        figure.suptitle(title)
        figure.tight_layout()
        return self._save_figure(figure, filename)

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

    def _load_sensitivity_results(self, path, default_filename, required_columns):
        results_path = (
            Path(path) if path else self.project_root / "results" / default_filename
        )
        if not results_path.exists():
            raise FileNotFoundError(f"Sensitivity results file not found: {results_path}")

        results = pd.read_csv(results_path)
        missing = required_columns - set(results.columns)
        if missing:
            missing_names = ", ".join(sorted(missing))
            raise ValueError(
                f"Sensitivity results file is missing required columns: {missing_names}"
            )
        return results

    def _load_runtime_complexity_results(self, path):
        results_path = (
            Path(path) if path else self.project_root / "results" / "runtime_complexity.csv"
        )
        if not results_path.exists():
            raise FileNotFoundError(f"Runtime complexity file not found: {results_path}")

        results = pd.read_csv(results_path)
        required_columns = {"dataset", "algorithm", "sample_size", "runtime_seconds"}
        missing = required_columns - set(results.columns)
        if missing:
            missing_names = ", ".join(sorted(missing))
            raise ValueError(
                f"Runtime complexity file is missing required columns: {missing_names}"
            )
        return results


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
        "--sensitivity",
        action="store_true",
        help="Also plot parameter sensitivity results.",
    )
    parser.add_argument(
        "--complexity",
        action="store_true",
        help="Also plot runtime complexity results.",
    )
    parser.add_argument(
        "--only-clusters",
        action="store_true",
        help="Create only cluster assignment plots, skipping metric plots.",
    )
    parser.add_argument(
        "--only-sensitivity",
        action="store_true",
        help="Create only parameter sensitivity plots, skipping metric plots.",
    )
    parser.add_argument(
        "--only-complexity",
        action="store_true",
        help="Create only runtime complexity plots, skipping metric plots.",
    )
    parser.add_argument(
        "--clarans-sensitivity-results",
        default=None,
        help="Path to clarans_parameter_sensitivity.csv.",
    )
    parser.add_argument(
        "--rock-sensitivity-results",
        default=None,
        help="Path to rock_parameter_sensitivity.csv.",
    )
    parser.add_argument(
        "--runtime-complexity-results",
        default=None,
        help="Path to runtime_complexity.csv.",
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

    if not args.only_clusters and not args.only_sensitivity and not args.only_complexity:
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
    if args.sensitivity or args.only_sensitivity:
        plot_paths.extend(
            visualizer.create_sensitivity_plots(
                clarans_path=args.clarans_sensitivity_results,
                rock_path=args.rock_sensitivity_results,
            )
        )
    if args.complexity or args.only_complexity:
        plot_paths.extend(
            visualizer.create_runtime_complexity_plots(
                runtime_path=args.runtime_complexity_results,
            )
        )

    print("Saved plots:")
    for path in plot_paths:
        print(path)


if __name__ == "__main__":
    main()
