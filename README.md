# Data-Mining-Clustering

Final EDAMI data mining project implementation of two clustering algorithms:

- CLARANS
- ROCK

The test data comes from the required benchmark repository:
https://github.com/deric/clustering-benchmark

Three small ARFF datasets are cached in `data/benchmark` for reproducible local
runs: `jain`, `flame`, and `spiral`.

## Project Structure

- `ClusteringAlgorithms/`: CLARANS and ROCK implementations.
- `DataLoader/`: benchmark ARFF loading and preprocessing.
- `Evaluation/`: internal and external clustering metrics.
- `Experiments/`: experiment runners for baseline evaluation, parameter sensitivity, and runtime complexity.
- `Visualization/`: metric, cluster assignment, sensitivity, and runtime plots.
- `data/benchmark/`: cached benchmark datasets.
- `results/`: generated CSV result tables and plot images.
- `main.py`: quick baseline run for both algorithms.
- `testing.ipynb`: earlier exploratory notebook.

## Run

```bash
python main.py
```

The program loads each benchmark dataset, runs both algorithms, and prints ARI
and NMI scores against the provided benchmark labels.

## Evaluation

`Evaluation/evaluation_module.py` contains reusable metrics for experiments:

- external metrics: ARI, NMI, purity
- internal metrics: Silhouette, Calinski-Harabasz, Davies-Bouldin

ROCK uses a distance threshold (`theta`) to build its neighbor graph. In
`main.py`, leaving `theta` empty makes the implementation choose a conservative
default from the pairwise distances. For experiments, pass a fixed threshold or
derive it from a percentile of pairwise distances:

```bash
python main.py --datasets spiral --rock-theta-percentile 3
```

To run only selected datasets:

```bash
python main.py --datasets jain flame
```

## Experiments

Run the experiment runner to evaluate all configured datasets and save a CSV
table:

```bash
python -m Experiments.experiment_runner
```

By default, results are saved to `results/evaluation_results.csv`. The table
contains the dataset, algorithm, parameters, runtime, external metrics, and
internal metrics.

Run parameter sensitivity experiments:

```bash
python -m Experiments.parameter_sensitivity
```

This saves:

- `results/clarans_parameter_sensitivity.csv`
- `results/rock_parameter_sensitivity.csv`

For a smaller ROCK threshold sweep:

```bash
python -m Experiments.parameter_sensitivity --algorithm rock --datasets spiral --theta-percentiles 1 2 3 5
```

Run runtime complexity experiments:

```bash
python -m Experiments.runtime_complexity
```

This saves `results/runtime_complexity.csv`. The experiment measures both
algorithms on increasing sample sizes and repeats each measurement to reduce
noise.

## Visualization

Create plots from the experiment results:

```bash
python -m Visualization.visualization_module
```

By default, plots are saved to `results/plots`:

- `external_metrics.png`
- `internal_metrics.png`
- `runtime_comparison.png`

To plot cluster assignments as well:

```bash
python -m Visualization.visualization_module --clusters
```

To generate only cluster plots for selected datasets:

```bash
python -m Visualization.visualization_module --only-clusters --datasets jain spiral
```

To generate parameter sensitivity plots from the sensitivity CSV files:

```bash
python -m Visualization.visualization_module --only-sensitivity
```

This creates:

- `clarans_sensitivity_ari.png`
- `clarans_sensitivity_runtime.png`
- `rock_sensitivity_ari.png`
- `rock_sensitivity_runtime.png`

To generate runtime complexity plots:

```bash
python -m Visualization.visualization_module --only-complexity
```

This creates:

- `runtime_complexity.png`
- `runtime_complexity_log.png`
