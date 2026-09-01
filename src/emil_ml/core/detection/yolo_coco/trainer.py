"""No-op trainer for the COCO-YOLO coarse detector — there are no weights
to fit (frozen, stock COCO-pretrained checkpoint; see predictor.py).

Same reasoning as core/classification/resnet_coarse/trainer.py: still
goes through the normal train_component() flow (training/onboard.py)
like every other model_type, rather than special-casing "this model_type
never needs training" anywhere else in the system.
"""

from __future__ import annotations

from pathlib import Path

from emil_ml.config.registry import Component
from emil_ml.core.base import BaseTrainer, EvaluationResult, TrainResult


class CocoCoarseTrainer(BaseTrainer):
    def train(self, component: Component) -> TrainResult:
        return TrainResult(
            model_path=None,
            threshold=None,
            details={"note": "stock COCO-pretrained YOLO checkpoint; no training performed"},
        )

    def evaluate(self, component: Component, train_result: TrainResult, *, output_dir: Path) -> EvaluationResult:
        """Nothing meaningful to evaluate locally: this is a frozen, stock
        COCO checkpoint, never fine-tuned on this component's own data.
        Re-evaluating it against the full COCO validation set would be
        both wildly disproportionate for a course/demo project (a 5+ GB
        download this project has no other use for) and would just
        reproduce Ultralytics' own already-published benchmark numbers,
        not anything specific to this component. Says so explicitly
        rather than either downloading COCO or fabricating a report.
        """
        return EvaluationResult(
            artifacts_dir=None,
            notes=[
                "model_type='coco_detector' uses a frozen, stock COCO-pretrained checkpoint, never "
                "fine-tuned on this component's own data — there is no local train/val split to "
                "evaluate against, so no evaluation artifacts were generated. (Ultralytics' own "
                "published COCO benchmark numbers for this checkpoint apply regardless of which "
                "component uses it — see the yolo_model_variant setting.)"
            ],
        )
