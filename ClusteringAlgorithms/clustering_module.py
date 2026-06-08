import numpy as np
from abc import ABC, abstractmethod
from scipy.spatial.distance import cdist, pdist, squareform
import heapq


class BaseClustering(ABC):
    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class CLARANS(BaseClustering):
    def __init__(self, k, numlocal=5, maxneighbor=50, random_state=None):
        self.k = k
        self.numlocal = numlocal
        self.maxneighbor = maxneighbor
        self.random_state = random_state
        self.medoids_indices_ = None
        self.medoids_vectors_ = None
        self.labels_ = None
        self.cost_ = None

    def _cost(self, X, medoids_indices):
        distances = cdist(X, X[medoids_indices], metric="euclidean")
        return np.sum(np.min(distances, axis=1))

    def fit(self, X):
        X = self._validate_input(X)
        n_samples = X.shape[0]
        self._validate_parameters(n_samples)

        rng = np.random.default_rng(self.random_state)
        best_cost = float("inf")
        best_node = None

        for _ in range(self.numlocal):
            current_node = rng.choice(n_samples, size=self.k, replace=False).tolist()
            current_cost = self._cost(X, current_node)
            j = 0

            while j < self.maxneighbor:
                neighbor_node = current_node.copy()
                swap_out_idx = rng.integers(0, self.k)
                non_medoids = np.setdiff1d(np.arange(n_samples), current_node)
                swap_in_idx = int(rng.choice(non_medoids))
                neighbor_node[swap_out_idx] = swap_in_idx

                neighbor_cost = self._cost(X, neighbor_node)

                if neighbor_cost < current_cost:
                    current_node = neighbor_node
                    current_cost = neighbor_cost
                    j = 0
                else:
                    j += 1

            if current_cost < best_cost:
                best_cost = current_cost
                best_node = current_node

        self.medoids_indices_ = best_node
        self.medoids_vectors_ = X[best_node]
        self.cost_ = best_cost
        self.labels_ = self.predict(X)
        return self

    def predict(self, X):
        if self.medoids_vectors_ is None:
            raise ValueError("Model has not been fitted yet.")
        X = self._validate_input(X)
        distances = cdist(X, self.medoids_vectors_, metric="euclidean")
        return np.argmin(distances, axis=1)

    def _validate_parameters(self, n_samples):
        if not 1 <= self.k < n_samples:
            raise ValueError("k must be at least 1 and smaller than the number of samples.")
        if self.numlocal < 1:
            raise ValueError("numlocal must be at least 1.")
        if self.maxneighbor < 1:
            raise ValueError("maxneighbor must be at least 1.")

    def _validate_input(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")
        return X


class ROCK(BaseClustering):
    def __init__(self, k, theta=None, alpha=0.5):
        self.k = k
        self.theta = theta
        self.alpha = alpha
        self.labels_ = None
        self.clusters_ = None
        self.n_clusters_ = None
        self.theta_ = None

    def fit(self, X):
        X = self._validate_input(X)
        n_samples = X.shape[0]
        self._validate_parameters(n_samples)

        dist_mat = squareform(pdist(X, "euclidean"))
        self.theta_ = self._resolve_theta(dist_mat)

        neighbors = (dist_mat <= self.theta_).astype(int)
        np.fill_diagonal(neighbors, 0)
        links = neighbors @ neighbors

        clusters = {i: [i] for i in range(n_samples)}
        active_clusters = set(range(n_samples))
        cl_links = {}

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if links[i, j] > 0:
                    cl_links[(i, j)] = links[i, j]
                    cl_links[(j, i)] = links[i, j]

        def compute_goodness(links, size_i, size_j):
            power = 1.0 + 2.0 * self.alpha
            expected = (size_i + size_j) ** power - size_i**power - size_j**power
            return links / expected if expected > 0 else 0

        pq = []
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if links[i, j] > 0:
                    g = compute_goodness(links[i, j], 1, 1)
                    heapq.heappush(pq, (-g, i, j))

        valid_clusters = {i: True for i in range(n_samples)}
        next_cluster_id = n_samples

        while len(active_clusters) > self.k and pq:
            neg_g, c1, c2 = heapq.heappop(pq)

            if not valid_clusters.get(c1, False) or not valid_clusters.get(c2, False):
                continue

            new_c = next_cluster_id
            next_cluster_id += 1

            clusters[new_c] = clusters[c1] + clusters[c2]
            valid_clusters[c1], valid_clusters[c2] = False, False
            active_clusters.remove(c1)
            active_clusters.remove(c2)
            valid_clusters[new_c] = True
            active_clusters.add(new_c)

            size_new = len(clusters[new_c])

            for other_c in active_clusters:
                if other_c == new_c:
                    continue

                link_count = cl_links.get((c1, other_c), 0) + cl_links.get(
                    (c2, other_c), 0
                )
                if link_count > 0:
                    cl_links[(new_c, other_c)] = link_count
                    cl_links[(other_c, new_c)] = link_count

                    size_other = len(clusters[other_c])
                    g = compute_goodness(link_count, size_new, size_other)
                    heapq.heappush(pq, (-g, min(new_c, other_c), max(new_c, other_c)))

        self.labels_ = np.zeros(n_samples, dtype=int)
        ordered_clusters = sorted(active_clusters, key=lambda cluster_id: min(clusters[cluster_id]))
        for label_idx, cluster_id in enumerate(ordered_clusters):
            for pt_idx in clusters[cluster_id]:
                self.labels_[pt_idx] = label_idx

        self.clusters_ = [clusters[cluster_id] for cluster_id in ordered_clusters]
        self.n_clusters_ = len(self.clusters_)
        return self

    def predict(self, X):
        if self.labels_ is None:
            raise ValueError("Model has not been fitted yet.")
        return self.labels_

    def _resolve_theta(self, dist_mat):
        if self.theta is not None:
            return self.theta

        distances = dist_mat[np.triu_indices_from(dist_mat, k=1)]
        distances = distances[distances > 0]
        if distances.size == 0:
            return 0.0
        return float(np.percentile(distances, 10))

    def _validate_parameters(self, n_samples):
        if not 1 <= self.k <= n_samples:
            raise ValueError("k must be at least 1 and no larger than the number of samples.")
        if self.theta is not None and self.theta <= 0:
            raise ValueError("theta must be positive when provided.")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive.")

    def _validate_input(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")
        return X
