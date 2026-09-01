"""No-op trainer for the ResNet coarse classifier — there are no weights
to fit (frozen ImageNet-1k pretrained ResNet-50; see predictor.py).

Still goes through the normal train_component() flow (training/onboard.py)
like every other model_type, rather than special-casing "this model_type
never needs training" anywhere else in the system — a component with
model_type='resnet_classifier' is created, then "trained" (this no-op,
effectively instant) exactly like any other component, and only then
becomes status='ready' and usable by the cascade. This keeps
registry_factory.py's dispatch the only place that knows this model_type
is different, instead of leaking a resnet_classifier-specific branch into
training/onboard.py or the Onboard page.
"""

from __future__ import annotations

from pathlib import Path

from emil_ml.config.registry import Component
from emil_ml.core.base import BaseTrainer, EvaluationResult, TrainResult


class ResNetCoarseTrainer(BaseTrainer):
    def train(self, component: Component) -> TrainResult:
        return TrainResult(
            model_path=None,
            threshold=None,
            details={"note": "pretrained ImageNet-1k ResNet-50; no training performed"},
        )

    def evaluate(self, component: Component, train_result: TrainResult, *, output_dir: Path) -> EvaluationResult:
        """Nothing meaningful to evaluate: frozen, pretrained weights never
        fine-tuned on this component's own data, so there is no local
        train/val split or labeled test set to evaluate against (same
        reasoning as core/detection/yolo_coco's CocoCoarseTrainer.evaluate()
        — both are pretrained coarse methods for core/cascade, not
        component-specific trained models). Says so explicitly rather than
        fabricating a report from data that doesn't exist.
        """
        return EvaluationResult(
            artifacts_dir=None,
            notes=[
                "model_type='resnet_classifier' uses frozen, pretrained ImageNet-1k weights, never "
                "fine-tuned on this component's own data — there is no local train/val split or "
                "labeled test set to evaluate against, so no evaluation artifacts were generated."
            ],
        )
