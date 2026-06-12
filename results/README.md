# Results

This folder contains generated experiment outputs.

CSV files:

- `evaluation_results.csv`: baseline CLARANS and ROCK evaluation.
- `clarans_parameter_sensitivity.csv`: CLARANS parameter sweep results.
- `rock_parameter_sensitivity.csv`: ROCK theta sensitivity results.
- `runtime_complexity.csv`: runtime measurements for increasing sample sizes.

Plots are saved in `plots/`.

Regenerate outputs with:

```bash
python -m Experiments.experiment_runner
python -m Experiments.parameter_sensitivity
python -m Experiments.runtime_complexity
python -m Visualization.visualization_module --clusters --sensitivity --complexity
```
