"""Configures the sklearn IsolationForest used for anomaly detection on CNN embeddings."""

from __future__ import annotations

from typing import Union

from sklearn.ensemble import IsolationForest


def build_isolation_forest(
    *,
    n_estimators: int,
    contamination: Union[str, float],
    max_features: float,
    random_state: int = 0,
) -> IsolationForest:
    return IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_features=max_features,
        random_state=random_state,
    )
