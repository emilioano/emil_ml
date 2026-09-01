"""Projects CNN embeddings to 2D and quantifies approved/failed separability.

IMPORTANT — visualization only, not an inference step: nothing here is
reused when a component is trained or when an image is inspected. UMAP in
particular is fit jointly over the whole batch of points it's given;
transforming a single new point later does not reliably place it where it
"belongs" relative to the already-fitted cloud, so this is not a
transform any predictor could safely reuse. core/detection,
core/classification, and core/anomaly do not import this module, and
nothing here writes a model artifact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

DEFAULT_METHOD = "umap"
VALID_METHODS = ("umap", "pca")

# UMAP's two most impactful parameters. n_neighbors: how many nearby points
# define "local" structure — lower values chase fine-grained local detail
# (noisier with few images), higher values favor broad/global structure.
# min_dist: how tightly points are allowed to pack together — lower values
# cluster more aggressively, which can visually exaggerate separation.
DEFAULT_UMAP_N_NEIGHBORS = 15
DEFAULT_UMAP_MIN_DIST = 0.1

# UMAP builds a neighbor graph and degenerates (or raises deep inside its own
# numpy code, e.g. "zero-size array to reduction operation maximum") once the
# dataset is barely bigger than n_neighbors. PCA has no such floor, so this
# only gates the UMAP branch below.
MIN_POINTS_FOR_UMAP = 4


@dataclass(frozen=True)
class ProjectionResult:
    """2D points plus separability numbers for two labeled classes."""

    points: np.ndarray  # (N, 2) float32
    method: str
    silhouette: float | None  # None if either class has < 2 examples, or only one class exists
    mean_intra_class_distance: float  # avg distance between same-class points
    mean_inter_class_distance: float | None  # avg distance between approved/failed points; None if only one class


def project(
    embeddings: np.ndarray,
    labels: list[str],
    *,
    method: str = DEFAULT_METHOD,
    n_neighbors: int = DEFAULT_UMAP_N_NEIGHBORS,
    min_dist: float = DEFAULT_UMAP_MIN_DIST,
    seed: int = 0,
) -> ProjectionResult:
    """Reduce `embeddings` to 2D with `method` and score class separability.

    Raises ValueError for fewer than 2 embeddings (nothing to project) or an
    unknown method — both are caller/config errors, not data conditions the
    diagnostics page should silently paper over.
    """
    if method not in VALID_METHODS:
        raise ValueError(f"Unknown method {method!r}; must be one of {VALID_METHODS}")
    if len(embeddings) < 2:
        raise ValueError("Need at least 2 embeddings to project.")

    if method == "pca":
        points = PCA(n_components=2, random_state=seed).fit_transform(embeddings)
    else:
        if len(embeddings) < MIN_POINTS_FOR_UMAP:
            raise ValueError(
                f"UMAP needs at least {MIN_POINTS_FOR_UMAP} images to build a meaningful "
                f"neighbor graph; got {len(embeddings)}. Try PCA instead, or add more "
                "training images."
            )
        import umap  # heavy import (numba JIT-compiles on first use) — deferred so PCA stays fast

        # n_neighbors can't exceed the number of other points available.
        effective_neighbors = max(2, min(n_neighbors, len(embeddings) - 1))
        reducer = umap.UMAP(
            n_neighbors=effective_neighbors, min_dist=min_dist, n_components=2, random_state=seed
        )
        points = reducer.fit_transform(embeddings)

    silhouette, intra, inter = _separability(points, labels)
    return ProjectionResult(
        points=points.astype(np.float32),
        method=method,
        silhouette=silhouette,
        mean_intra_class_distance=intra,
        mean_inter_class_distance=inter,
    )


def _pairwise_mean_distance(a: np.ndarray, b: np.ndarray | None = None) -> float:
    """Mean Euclidean distance between all pairs in `a` (or between `a` and `b`)."""
    if b is None:
        if len(a) < 2:
            return 0.0
        d = np.linalg.norm(a[:, None, :] - a[None, :, :], axis=-1)
        iu = np.triu_indices(len(a), k=1)
        return float(d[iu].mean())
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float(d.mean())


def _separability(points: np.ndarray, labels: list[str]) -> tuple[float | None, float, float | None]:
    labels_arr = np.asarray(labels)
    classes = np.unique(labels_arr)

    silhouette = None
    if len(classes) == 2 and all((labels_arr == c).sum() >= 2 for c in classes):
        silhouette = float(silhouette_score(points, labels_arr))

    intra_dists = [
        _pairwise_mean_distance(points[labels_arr == c]) for c in classes if (labels_arr == c).sum() >= 2
    ]
    mean_intra = float(np.mean(intra_dists)) if intra_dists else 0.0

    mean_inter = None
    if len(classes) == 2:
        mean_inter = _pairwise_mean_distance(points[labels_arr == classes[0]], points[labels_arr == classes[1]])

    return silhouette, mean_intra, mean_inter


@dataclass(frozen=True)
class Interpretation:
    headline: str
    explanation: str
    recommendation: str | None  # None when there isn't enough data to recommend anything


def interpret(result: ProjectionResult) -> Interpretation:
    """A short, honest read of the separability numbers — a hint, not a verdict.

    Thresholds (silhouette > 0.25, or inter/intra distance ratio > 1.5) are
    deliberately loose rules of thumb, not calibrated statistics — with the
    small datasets this tool is meant for, exact cutoffs would be false
    precision. Read the plot; treat the numbers as a second opinion.
    """
    if result.mean_inter_class_distance is None:
        return Interpretation(
            headline="Only one class present.",
            explanation=(
                "All available training images are the same class (usually: no failed "
                "examples uploaded yet). Separability can't be assessed until both approved "
                "and failed examples exist for this component."
            ),
            recommendation=None,
        )

    ratio = (
        result.mean_inter_class_distance / result.mean_intra_class_distance
        if result.mean_intra_class_distance > 0
        else float("inf")
    )
    well_separated = (result.silhouette is not None and result.silhouette > 0.25) or ratio > 1.5

    if well_separated:
        return Interpretation(
            headline="Classes look separable in whole-image features.",
            explanation=(
                "Approved and failed images form mostly distinct clouds in this projection. "
                "A whole-image classifier should be able to learn this signal — if it still "
                "struggles in practice, the likely cause is training setup or data volume, "
                "not that the signal is missing from whole-image features."
            ),
            recommendation=None,
        )

    return Interpretation(
        headline="Classes overlap heavily in whole-image features.",
        explanation=(
            "Approved and failed images mix together in this projection, which suggests the "
            "difference between them isn't visible in a whole-image feature summary — "
            "typically because the defect is small and localized relative to the whole image. "
            "A method that looks at specific regions rather than the whole image tends to do "
            "better here."
        ),
        recommendation=(
            "Classes overlap heavily here — consider YOLO (localizable defects, needs "
            "bounding-box annotations) instead of a whole-image classifier or autoencoder "
            "for this component."
        ),
    )
