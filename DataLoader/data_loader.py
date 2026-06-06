import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

class DataLoader:
    """
    Module for loading and preprocessing test datasets (toy datasets).
    """
    def __init__(self, n_samples=500):
        self.n_samples = n_samples

    def get_circles(self, noise=0.05, factor=0.5):
        X, y = datasets.make_circles(n_samples=self.n_samples, factor=factor, noise=noise, random_state=42)
        return self._preprocess(X), y

    def get_moons(self, noise=0.05):
        X, y = datasets.make_moons(n_samples=self.n_samples, noise=noise, random_state=42)
        return self._preprocess(X), y

    def get_blobs(self, random_state=8):
        X, y = datasets.make_blobs(n_samples=self.n_samples, random_state=random_state)
        return self._preprocess(X), y

    def get_no_structure(self):
        X = np.random.rand(self.n_samples, 2)
        y = np.zeros(self.n_samples)
        return self._preprocess(X), y

    def _preprocess(self, X):
        # Normalization is crucial for distance-based algorithms
        return StandardScaler().fit_transform(X)