from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class BenchmarkDataset:
    name: str
    X: object
    y: object
    path: Path
    source_url: str


class DataLoader:
    """
    Loader for datasets from deric/clustering-benchmark.

    The project feedback requires benchmark data instead of generated toy data.
    Files are cached locally in data/benchmark and parsed from ARFF format.
    """

    BASE_URL = (
        "https://raw.githubusercontent.com/deric/clustering-benchmark/master/"
        "src/main/resources/datasets/artificial"
    )
    DATASETS = {
        "jain": "jain.arff",
        "flame": "flame.arff",
        "spiral": "spiral.arff",
    }

    def __init__(self, cache_dir=None):
        project_root = Path(__file__).resolve().parents[1]
        self.cache_dir = Path(cache_dir) if cache_dir else project_root / "data" / "benchmark"

    def list_datasets(self):
        return sorted(self.DATASETS)

    def load(self, name, standardize=True, download=True):
        if name not in self.DATASETS:
            available = ", ".join(self.list_datasets())
            raise ValueError(f"Unknown dataset '{name}'. Available datasets: {available}")

        filename = self.DATASETS[name]
        source_url = f"{self.BASE_URL}/{filename}"
        path = self.cache_dir / filename

        if download and not path.exists():
            self._download(source_url, path)
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {path}. Enable download or place the file there."
            )

        X, y = self._read_arff(path)
        if standardize:
            X = StandardScaler().fit_transform(X)

        return BenchmarkDataset(name=name, X=X, y=y, path=path, source_url=source_url)

    def load_all(self, standardize=True, download=True):
        return [
            self.load(name, standardize=standardize, download=download)
            for name in self.list_datasets()
        ]

    def _download(self, url, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        request = Request(url, headers={"User-Agent": "Data-Mining-Clustering"})
        with urlopen(request, timeout=30) as response:
            path.write_bytes(response.read())

    def _read_arff(self, path):
        data, _ = arff.loadarff(path)
        frame = pd.DataFrame(data)

        for column in frame.columns:
            if frame[column].dtype == object:
                frame[column] = frame[column].map(
                    lambda value: value.decode("utf-8") if isinstance(value, bytes) else value
                )

        label_column = "class" if "class" in frame.columns else frame.columns[-1]
        y = pd.factorize(frame[label_column])[0]
        X = frame.drop(columns=[label_column]).apply(pd.to_numeric).to_numpy(float)

        return X, y
