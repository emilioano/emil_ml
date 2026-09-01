"""Common interface every analysis method (autoencoder, isolation forest,
classifier, ...) implements.

`pipeline.inspect` and the Streamlit UI only ever talk to `BaseTrainer` /
`BasePredictor` and the result types below — they never know which concrete
method (or modality) is behind them. Adding a new method means implementing
these two classes and registering them in `core/registry_factory.py`;
nothing else in the system changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from emil_ml.config.registry import Component


@dataclass
class TrainResult:
    """Common shape returned by every trainer, regardless of method."""

    model_path: Path | None
    threshold: float | None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Common shape returned by every trainer's evaluate(), regardless of
    method — deliberately minimal and generic, the same "uniform envelope,
    method-specific content inside" split TrainResult/PredictionResult
    already use. Model-type-specific content (a classification report, YOLO
    mAP, an ROC curve, ...) lives in `metrics` (JSON-serializable summary
    numbers, already written to `artifacts_dir/metrics.json` by the trainer
    itself — see core/evaluation/io.py) and as image files under
    `artifacts_dir` (see core/evaluation/plots.py for the shared plotting
    helpers every trainer's evaluate() draws from, so a confusion matrix
    looks the same regardless of which method produced it).

    `artifacts_dir` is None only when evaluation produced nothing at all
    (e.g. a frozen, never-fine-tuned coarse detector with no local labeled
    data to evaluate against — see core/classification/resnet_coarse and
    core/detection/yolo_coco's evaluate() overrides) — that's a valid,
    honest outcome (see `notes`), not an error.
    """

    artifacts_dir: Path | None
    metrics: dict[str, Any] = field(default_factory=dict)
    plot_files: list[str] = field(default_factory=list)  # filenames relative to artifacts_dir
    notes: list[str] = field(default_factory=list)  # e.g. "no labeled test set available — X was skipped"


@dataclass
class PredictionResult:
    """Common shape returned by every predictor, regardless of method.

    `verdict` and `score` are always populated. `threshold` is None for
    methods that don't use one (e.g. a classifier). Method-specific extras
    (predicted class, per-class probabilities, embedding distances, ...) go
    in `details`.
    """

    verdict: str  # "approved" | "failed"
    score: float
    threshold: float | None
    details: dict[str, Any] = field(default_factory=dict)


class BaseTrainer(ABC):
    """Trains a component's model from its training images."""

    @abstractmethod
    def train(self, component: Component) -> TrainResult: ...

    @abstractmethod
    def evaluate(self, component: Component, train_result: TrainResult, *, output_dir: Path) -> EvaluationResult:
        """Generate and save this method's own meaningful evaluation
        artifacts (metrics + plots) into `output_dir` (already created by
        the caller — see training/onboard.py's train_component(), the only
        caller, gated by the component's generate_evaluation_report
        setting). Abstract, not a default no-op: every model_type must
        deliberately decide what "evaluated" means for it — forcing the
        same artifacts on every method produces meaningless or crashing
        output (e.g. an "accuracy over epochs" curve for YOLO, which
        doesn't train on accuracy at all — see YoloTrainer's own
        evaluate(), which reports mAP instead). A method with genuinely
        nothing to evaluate (e.g. a frozen, never-fine-tuned coarse
        detector) still must implement this — just returning an
        EvaluationResult with `artifacts_dir=None` and an explanatory note,
        the same "say so honestly" principle used elsewhere in this project
        (see core/reporting/prompt.py's rules) rather than silently
        producing nothing or raising.

        Called once per successful train() (never on a failed one — see
        train_component()), never from core/search/grid_search.py's sweep
        trials (full evaluation reporting per trial would be prohibitively
        slow and isn't the trial-ranking mechanism grid_search.py already
        has via TrainResult.details["metrics"]).
        """
        ...


class BasePredictor(ABC):
    """Scores a single already-prepared input against a trained component model.

    Predictors are bound to one component at construction time (they load
    that component's model). `predict` takes whatever the component's
    modality handler produced (see `core/modality/base.py`) — never raw
    input — so a predictor never has to know how to load/decode anything.
    """

    def __init__(self, component: Component) -> None:
        self.component = component

    @abstractmethod
    def predict(self, prepared_input: Any) -> PredictionResult: ...
