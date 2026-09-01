"""Common interface for turning raw input into whatever a predictor expects.

Each modality (image now, text later) owns "how do I read and preprocess an
input item for this component" — nothing about detection/analysis logic
lives here. `pipeline.inspect` calls a handler to get a prepared object, then
hands that straight to the predictor; it never inspects the raw input itself,
and never branches on modality — that dispatch lives in `core.registry_factory`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from emil_ml.config.registry import Component


class BaseModalityHandler(ABC):
    """Loads and preprocesses raw input for one component's modality.

    Bound to a component at construction (mirrors `BasePredictor`), since
    preprocessing depends on component config (e.g. image_size).
    """

    def __init__(self, component: Component) -> None:
        self.component = component

    @abstractmethod
    def load(self, raw_input: Any) -> Any:
        """Turn raw input (a path, bytes, ...) into what this component's predictor expects."""
        ...
