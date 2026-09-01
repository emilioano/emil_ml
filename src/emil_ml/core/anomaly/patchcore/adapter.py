"""Thin adapter around anomalib's PatchCore — config, training, inference.

Everything in this file that touches anomalib/torch is deferred to inside
function bodies, never imported at module top level. That's deliberate, not
an oversight: `trainer.py`/`predictor.py` (and this module) must remain
importable even when anomalib isn't installed, since `registry_factory`
imports every method's trainer/predictor class eagerly at app startup —
importing anomalib there would make it a hard dependency for the whole app,
not the optional one it's meant to be (see the `patchcore` extra in
pyproject.toml). Only actually training or running inference on a PatchCore
component requires anomalib to be installed.

anomalib's own PostProcessor normalizes every score against calibration
statistics gathered from the training run (min/max observed anomaly scores,
plus an F1-optimal threshold when labeled failed examples are available) such
that, with its default sensitivity, the calibrated decision boundary always
lands at exactly 0.5 in normalized score space — see
`anomalib.post_processing.post_processor.PostProcessor.normalized_image_threshold`.
This adapter relies on that: it reports the *normalized* score/threshold
(0-1, matching every other method in this app) as the primary result, and
keeps the *raw* threshold value only as supplementary detail.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

NORMALIZED_THRESHOLD = 0.5

_truststore_injected = False


class PatchCoreUnavailableError(ImportError):
    """anomalib isn't installed — PatchCore needs the optional `patchcore` extra."""

    def __init__(self) -> None:
        super().__init__(
            "PatchCore requires the 'anomalib' package, which isn't installed. "
            "It's an optional dependency (heavy: PyTorch + Lightning + friends) so "
            "installing EMIL doesn't force it on users who don't need PatchCore. "
            "Install it with: pip install -e '.[patchcore]' "
            "(or: pip install anomalib>=2.5)"
        )


def _import_anomalib() -> tuple[Any, Any, Any, Any]:
    """Import and return (Folder, Engine, Patchcore, PredictDataset), or raise clearly."""
    global _truststore_injected
    try:
        from anomalib.data import Folder, PredictDataset
        from anomalib.engine import Engine
        from anomalib.models import Patchcore
    except ImportError as exc:
        raise PatchCoreUnavailableError() from exc

    if not _truststore_injected:
        # Same corporate-TLS-interception workaround as YOLO's weight
        # pre-fetch (core/detection/yolo/model.py): anomalib downloads the
        # pretrained backbone from Hugging Face Hub (httpx-based) on first
        # use, which fails plain certificate verification on this network.
        # truststore verifies against the OS cert store instead and patches
        # Python's global ssl module, so this only needs to run once.
        import truststore

        truststore.inject_into_ssl()
        _truststore_injected = True

    return Folder, Engine, Patchcore, PredictDataset


@dataclass(frozen=True)
class TrainOutcome:
    """What adapter.train() hands back to PatchCoreTrainer."""

    checkpoint_path: Path
    raw_threshold: float
    metrics: dict[str, float] = field(default_factory=dict)  # e.g. {"image_AUROC": ..., "image_F1Score": ...}


def train(
    *,
    normal_dir: Path,
    abnormal_dir: Path | None,
    results_dir: Path,
    backbone: str,
    coreset_sampling_ratio: float,
    num_neighbors: int,
    batch_size: int,
    seed: int = 0,
) -> TrainOutcome:
    """Train PatchCore on `normal_dir`'s images and return where its checkpoint landed.

    `abnormal_dir` (optional, matching the autoencoder's approved-only
    philosophy) is never trained on — anomalib's `Folder` datamodule only
    ever places abnormal-labeled images in its validation/test splits, used
    to calibrate the threshold and, if present, compute image-level AUROC.
    Without it, the threshold still gets computed (from the extreme end of
    the normal images' own score distribution — anomalib warns about this
    itself, the same "less reliable without labeled failures" tradeoff the
    autoencoder's percentile threshold has), just without a meaningful AUROC.
    """
    Folder, Engine, Patchcore, _ = _import_anomalib()

    if results_dir.exists():
        shutil.rmtree(results_dir)

    datamodule = Folder(
        name="component",
        root=normal_dir.parent,
        normal_dir=normal_dir.name,
        abnormal_dir=abnormal_dir.name if abnormal_dir is not None else None,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=0,  # small datasets; avoids Windows multiprocessing overhead/pitfalls
    )
    model = Patchcore(
        backbone=backbone,
        coreset_sampling_ratio=coreset_sampling_ratio,
        num_neighbors=num_neighbors,
    )
    engine = Engine(default_root_dir=str(results_dir), accelerator="cpu", devices=1)
    engine.fit(model=model, datamodule=datamodule)

    metrics: dict[str, float] = {}
    if abnormal_dir is not None:
        # AUROC/F1 are only meaningful with labeled failed examples in the
        # test split — anomalib still "succeeds" without them, but returns a
        # degenerate 0.0 that would misleadingly look like terrible
        # performance rather than "not measurable" (confirmed empirically).
        test_results = engine.test(model=model, datamodule=datamodule)
        if test_results:
            metrics = {k: float(v) for k, v in test_results[0].items()}

    checkpoint_path = Path(engine.trainer.checkpoint_callback.best_model_path)
    raw_threshold = float(model.post_processor.image_threshold)
    return TrainOutcome(checkpoint_path=checkpoint_path, raw_threshold=raw_threshold, metrics=metrics)


def load_model(checkpoint_path: Path):  # noqa: ANN201 - return type is anomalib.models.Patchcore, imported lazily
    """Load a trained PatchCore model from a checkpoint, ready for inference.

    `weights_only=False`: this checkpoint is always one EMIL itself produced
    (via `train()` above, into this component's own models/ directory), not
    a third-party download — PyTorch 2.6+'s stricter default `weights_only`
    loading rejects anomalib's checkpoints (they embed a small enum type,
    `anomalib.PrecisionType`, alongside the tensors) unless disabled for a
    source you trust.
    """
    _, _, Patchcore, _ = _import_anomalib()
    return Patchcore.load_from_checkpoint(str(checkpoint_path), weights_only=False)


@dataclass(frozen=True)
class PredictOutcome:
    normalized_score: float  # 0-1, decision boundary always at NORMALIZED_THRESHOLD
    heatmap: np.ndarray  # (H, W) float32, normalized 0-1, same aspect as the model's internal resize


def predict(model, image: Image.Image) -> PredictOutcome:  # noqa: ANN001 - model is anomalib.models.Patchcore
    """Run one image through a loaded PatchCore model.

    Goes through a temp file + anomalib's own `PredictDataset`/`Engine.predict`
    rather than hand-rolling the preprocessing tensor pipeline ourselves —
    keeps this adapter thin and guarantees inference sees exactly the same
    transform pipeline as training did (baked into the model's own
    pre_processor), which hand-rolled preprocessing could subtly drift from.
    """
    _, Engine, _, PredictDataset = _import_anomalib()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "input.png"
        image.convert("RGB").save(tmp_path)

        engine = Engine(accelerator="cpu", devices=1)
        predict_dataset = PredictDataset(path=tmp_path)
        predictions = engine.predict(model=model, dataset=predict_dataset)

    batch = predictions[0]
    normalized_score = float(batch.pred_score[0])
    heatmap = batch.anomaly_map[0].detach().cpu().numpy().astype(np.float32)

    if np.isnan(normalized_score):
        # Degenerate calibration (every normal training image scored
        # identically, so min==max and normalization divides by zero) —
        # observed only with unrealistically uniform synthetic test images,
        # but guard against it anyway rather than let a NaN silently compare
        # as "not >= threshold" (always False -> always "approved", exactly
        # backwards for an anomaly detector). Fall back to the boundary
        # itself: neither confidently normal nor anomalous.
        normalized_score = NORMALIZED_THRESHOLD

    return PredictOutcome(
        normalized_score=normalized_score,
        heatmap=np.nan_to_num(heatmap, nan=0.0),
    )
