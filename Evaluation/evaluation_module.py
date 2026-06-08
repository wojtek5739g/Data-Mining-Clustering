from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)


@dataclass(frozen=True)
class EvaluationResult:
    dataset: str
    algorithm: str
    n_samples: int
    n_clusters: int
    ari: float | None
    nmi: float | None
    purity: float | None
    silhouette: float | None
    calinski_harabasz: float | None
    davies_bouldin: float | None

    def as_dict(self):
        return asdict(self)


def evaluate_clustering(X, labels_pred, labels_true=None, dataset="", algorithm=""):
    """
    Compute external and internal clustering quality metrics.

    External metrics use ground-truth labels and are returned only when
    labels_true is provided. Internal metrics use only X and predicted labels.
    """
    X = np.asarray(X, dtype=float)
    labels_pred = np.asarray(labels_pred)
    labels_true = None if labels_true is None else np.asarray(labels_true)

    _validate_inputs(X, labels_pred, labels_true)

    n_samples = X.shape[0]
    n_clusters = len(np.unique(labels_pred))
    external = _external_metrics(labels_true, labels_pred)
    internal = _internal_metrics(X, labels_pred, n_clusters)

    return EvaluationResult(
        dataset=dataset,
        algorithm=algorithm,
        n_samples=n_samples,
        n_clusters=n_clusters,
        ari=external["ari"],
        nmi=external["nmi"],
        purity=external["purity"],
        silhouette=internal["silhouette"],
        calinski_harabasz=internal["calinski_harabasz"],
        davies_bouldin=internal["davies_bouldin"],
    )


def purity_score(labels_true, labels_pred):
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    if labels_true.shape[0] != labels_pred.shape[0]:
        raise ValueError("labels_true and labels_pred must have the same length.")

    total_correct = 0
    for cluster_label in np.unique(labels_pred):
        mask = labels_pred == cluster_label
        true_labels, counts = np.unique(labels_true[mask], return_counts=True)
        if true_labels.size > 0:
            total_correct += counts.max()

    return total_correct / labels_pred.shape[0]


def _external_metrics(labels_true, labels_pred):
    if labels_true is None:
        return {"ari": None, "nmi": None, "purity": None}

    return {
        "ari": adjusted_rand_score(labels_true, labels_pred),
        "nmi": normalized_mutual_info_score(labels_true, labels_pred),
        "purity": purity_score(labels_true, labels_pred),
    }


def _internal_metrics(X, labels_pred, n_clusters):
    if not 1 < n_clusters < X.shape[0]:
        return {
            "silhouette": None,
            "calinski_harabasz": None,
            "davies_bouldin": None,
        }

    return {
        "silhouette": silhouette_score(X, labels_pred),
        "calinski_harabasz": calinski_harabasz_score(X, labels_pred),
        "davies_bouldin": davies_bouldin_score(X, labels_pred),
    }


def _validate_inputs(X, labels_pred, labels_true):
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")
    if labels_pred.ndim != 1:
        raise ValueError("labels_pred must be a one-dimensional array.")
    if X.shape[0] != labels_pred.shape[0]:
        raise ValueError("X and labels_pred must contain the same number of samples.")
    if labels_true is not None and labels_true.shape[0] != labels_pred.shape[0]:
        raise ValueError("labels_true and labels_pred must have the same length.")
