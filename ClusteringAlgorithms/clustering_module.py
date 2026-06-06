import numpy as np
from abc import ABC, abstractmethod
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse import csr_matrix
import random
import heapq


class BaseClustering(ABC):
    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class CLARANS(BaseClustering):
    def __init__(self, k, numlocal, maxneighbor):
        self.k = k
        self.numlocal = numlocal
        self.maxneighbor = maxneighbor
        self.medoids_indices_ = None
        self.medoids_vectors_ = None
        self.labels_ = None

    def _cost(self, X, medoids_indices):
        # Calculating distances from all points to their nearest medoid
        distances = cdist(X, X[medoids_indices], metric="euclidean")
        return np.sum(np.min(distances, axis=1))

    def fit(self, X):
        n_samples = X.shape[0]
        best_cost = float("inf")
        best_node = None

        for _ in range(self.numlocal):
            current_node = random.sample(range(n_samples), self.k)
            current_cost = self._cost(X, current_node)
            j = 0

            while j < self.maxneighbor:
                # Change one medoid to a random non-medoid point (random neighbor)
                neighbor_node = current_node.copy()
                swap_out_idx = random.randint(0, self.k - 1)
                non_medoids = list(set(range(n_samples)) - set(current_node))
                swap_in_idx = random.choice(non_medoids)
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
        self.labels_ = self.predict(X)
        return self

    def predict(self, X):
        if self.medoids_vectors_ is None:
            raise ValueError("Model has not been fitted yet.")
        distances = cdist(X, self.medoids_vectors_, metric="euclidean")
        return np.argmin(distances, axis=1)


class ROCK(BaseClustering):
    def __init__(self, k, theta, alpha=0.5):
        self.k = k
        self.theta = theta
        self.alpha = alpha
        self.labels_ = None

    def fit(self, X):
        n_samples = X.shape[0]

        # Describing the neighborhood using a similarity threshold (theta)
        dist_mat = squareform(pdist(X, "euclidean"))
        # 1 for distances <= theta, 0 otherwise
        A_dense = (dist_mat <= self.theta).astype(int)
        np.fill_diagonal(A_dense, 0)

        # Sparse representation (Sparse Matrix) for the adjacency matrix
        A_sparse = csr_matrix(A_dense)

        # Calculating the number of links between pairs of points (sparse matrix multiplication)
        L_sparse = A_sparse.dot(A_sparse)
        L_dense = L_sparse.toarray()

        # Clusters initilization
        clusters = {i: [i] for i in range(n_samples)}
        active_clusters = set(range(n_samples))

        cl_links = {}
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if L_dense[i, j] > 0:
                    cl_links[(i, j)] = L_dense[i, j]
                    cl_links[(j, i)] = L_dense[j, i]

        def compute_goodness(links, size_i, size_j):
            # Parameter scaling the goodness function
            power = 1.0 + 2.0 * self.alpha
            expected = (size_i + size_j) ** power - size_i**power - size_j**power
            return links / expected if expected > 0 else 0

        # Building the priority queue with initial cluster pairs (-goodness because heapq is a min-heap and we want max-heap behavior)
        pq = []
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if L_dense[i, j] > 0:
                    g = compute_goodness(L_dense[i, j], 1, 1)
                    heapq.heappush(pq, (-g, i, j))

        valid_clusters = {i: True for i in range(n_samples)}
        next_cluster_id = n_samples

        # Agglomerative merging of clusters until we have k clusters left
        while len(active_clusters) > self.k and pq:
            neg_g, c1, c2 = heapq.heappop(pq)

            # Verification if the clusters have not already been merged (lazy deletion)
            if not valid_clusters.get(c1, False) or not valid_clusters.get(c2, False):
                continue

            # Merging chosen clusters
            new_c = next_cluster_id
            next_cluster_id += 1

            clusters[new_c] = clusters[c1] + clusters[c2]
            valid_clusters[c1], valid_clusters[c2] = False, False
            active_clusters.remove(c1)
            active_clusters.remove(c2)
            valid_clusters[new_c] = True
            active_clusters.add(new_c)

            size_new = len(clusters[new_c])

            # Actualization of links and priority queue after merging clusters
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

        # Assigning final labels to points based on the resulting clusters
        self.labels_ = np.zeros(n_samples, dtype=int)
        for label_idx, cluster_id in enumerate(active_clusters):
            for pt_idx in clusters[cluster_id]:
                self.labels_[pt_idx] = label_idx

        return self
    
    def predict(self, X):
        # ROCK is a hierarchical/agglomerative algorithm. The labels are a direct result of the fit() method.
        return self.labels_
