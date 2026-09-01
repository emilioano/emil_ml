"""Onboarding and training orchestration.

Creates a component (registry row + folder tree), ingests uploaded training
images, and runs training. Contains all onboarding business logic; the
Streamlit onboarding page is a thin view over these functions. No Streamlit
imports here — this must also work from a future FastAPI backend unchanged.
"""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from emil_ml.config.registry import SETTINGS_DEFAULTS, Component, ComponentRegistry
from emil_ml.config.settings import DEFAULT_MODALITY, DEFAULT_MODEL_TYPE
from emil_ml.core import registry_factory
from emil_ml.core.base import TrainResult
from emil_ml.core.detection.yolo import annotation as yolo_annotation
from emil_ml.core.detection.yolo.annotation import YoloBox
from emil_ml.core.inspections import store as inspection_store
from emil_ml.core.inspections.store import InspectionRecord
from emil_ml.core.training_runs import store as training_runs_store
from emil_ml.utils import image_io
from emil_ml.utils.paths import ComponentPaths, for_component

logger = logging.getLogger(__name__)

_BACKUP_TIMESTAMP_FMT = "%y%m%d%H%M"
_BACKUP_STEM_RE = re.compile(r"_bak\d{10}(_\d+)?$")


def create_component(
    display_name: str,
    *,
    modality: str = DEFAULT_MODALITY,
    model_type: str = DEFAULT_MODEL_TYPE,
    registry: ComponentRegistry | None = None,
    **settings: Any,
) -> Component:
    """Register a new component and create its folder tree. Status starts at 'created'.

    Any setting from `ComponentRegistry.SETTINGS_DEFAULTS` (image_size, epochs,
    base_model, score_method, ...) can be overridden by keyword.
    """
    registry = registry or ComponentRegistry()
    component = registry.create(display_name, modality=modality, model_type=model_type, **settings)
    for_component(component.name).create_all()
    return component


def add_training_images(
    component_name: str,
    *,
    approved: list[tuple[str, bytes]] | None = None,
    failed: list[tuple[str, bytes]] | None = None,
) -> None:
    """Save uploaded (filename, data) pairs into the component's training folders."""
    paths = for_component(component_name)
    for filename, data in approved or []:
        image_io.save_upload(data, paths.training_approved_dir, filename)
    for filename, data in failed or []:
        image_io.save_upload(data, paths.training_failed_dir, filename)


def train_component(
    component_name: str,
    *,
    registry: ComponentRegistry | None = None,
    incorporated_correction_ids: list[int] | None = None,
) -> TrainResult:
    """Train the component's model (via its model_type's trainer) and update the registry.

    `incorporated_correction_ids` is purely for traceability — pass the ids
    of any verified corrections that were just incorporated into training/
    (see incorporate_verified_corrections()) immediately before calling
    this, so the resulting training_runs row (and its evaluation report,
    if generated) records exactly which corrections a retrain was
    responding to. Optional: a plain, non-correction-loop retrain passes
    nothing, same as before this parameter existed.
    """
    registry = registry or ComponentRegistry()
    component = registry.get(component_name)
    if component is None:
        raise KeyError(f"No component named {component_name!r}")

    settings_snapshot = {key: getattr(component, key) for key in SETTINGS_DEFAULTS}

    registry.update_status(component.name, "training")
    try:
        trainer = registry_factory.get_trainer(component.modality, component.model_type)
        result = trainer.train(component)
    except Exception as exc:
        registry.update_status(component.name, "failed")
        # Logged even on failure — see training_runs/store.py's module docstring:
        # "all performance data for every model we train" means every attempt
        # that was actually run, not just the ones that produced a model.
        training_runs_store.create(
            component.name,
            display_name=component.display_name,
            modality=component.modality,
            model_type=component.model_type,
            status="failed",
            error=str(exc),
            settings=settings_snapshot,
        )
        raise

    # Stored relative to the component's own root (e.g. "models/autoencoder.keras"),
    # not absolute — an absolute path baked in at training time is tied to
    # whatever machine/filesystem trained it (confirmed the hard way: a
    # component trained under WSL is unusable from a Windows checkout of
    # the same emil.db, and vice versa). Every predictor resolves this via
    # ComponentPaths.resolve_model_path(), which is also what makes this
    # field an actual functioning pointer rather than a write-only cache
    # of something already re-derivable from name + model_type.
    paths = for_component(component.name)
    relative_model_path = (
        result.model_path.relative_to(paths.root).as_posix() if result.model_path is not None else None
    )
    registry.update_training_result(
        component.name,
        anomaly_threshold=result.threshold,
        model_path=relative_model_path,
        status="ready",
    )

    # Model_type-aware evaluation report — see core/base.py's
    # BaseTrainer.evaluate() and core/evaluation/. Gated by the
    # component's own setting (fast-iteration escape hatch); best-effort
    # when on — a failure here must never turn an otherwise-successful
    # training into a failed one, since evaluation is transparency, not
    # correctness (see BaseTrainer.evaluate()'s own docstring).
    evaluation_dir_relative: str | None = None
    evaluation_details: dict[str, Any] = {}
    if component.generate_evaluation_report:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        eval_output_dir = paths.evaluation_run_dir(timestamp)
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        try:
            evaluation = trainer.evaluate(component, result, output_dir=eval_output_dir)
            if evaluation.artifacts_dir is not None:
                evaluation_dir_relative = evaluation.artifacts_dir.relative_to(paths.root).as_posix()
            evaluation_details = {
                "dir": evaluation_dir_relative,
                "metrics": evaluation.metrics,
                "plot_files": evaluation.plot_files,
                "notes": evaluation.notes,
            }
        except Exception:  # noqa: BLE001 - evaluation is best-effort transparency, never fatal to a successful training
            logger.exception(
                "component=%s evaluation reporting failed (training itself still succeeded)", component.name
            )

    details_with_evaluation = dict(result.details)
    if evaluation_details:
        details_with_evaluation["evaluation"] = evaluation_details
    if incorporated_correction_ids:
        details_with_evaluation["incorporated_correction_ids"] = incorporated_correction_ids

    training_runs_store.create(
        component.name,
        display_name=component.display_name,
        modality=component.modality,
        model_type=component.model_type,
        status="success",
        threshold=result.threshold,
        model_path=relative_model_path,
        settings=settings_snapshot,
        metrics=result.details.get("metrics", {}),
        details=details_with_evaluation,
        evaluation_dir=evaluation_dir_relative,
    )
    return result


# --- Model backup / rollback -------------------------------------------------
#
# A generic, model_type-agnostic safety net for retraining: a component's
# "current model" is always whatever file component.model_path points at
# (see ComponentPaths.resolve_model_path() in utils/paths.py) — one file,
# regardless of whether it's an autoencoder's .keras, a YOLO best.pt, or
# anything else, so one implementation covers every model_type. Backups
# aren't tracked in the registry/DB, same as the live model file itself —
# they're plain files sitting in models/, found by filename pattern.


def backup_current_model(component_name: str, *, registry: ComponentRegistry | None = None) -> str | None:
    """Copy a component's current model file to a timestamped backup
    within models/, meant to be called right before a retrain overwrites
    it. Returns the backup's component-root-relative path, or None if
    there's no current model file yet (e.g. this component has never
    been trained).

    Filename pattern: "<original stem>_bak<YYMMDDHHMM><original suffix>" —
    same timestamp-suffix-for-uniqueness idea core/inspections/lifecycle.py's
    archive() uses, just minute resolution rather than microsecond: this is
    a deliberate, occasional action a person takes before clicking Train,
    not a high-frequency automated one, so collisions within the same
    minute are the only real risk — see the loop below, which appends a
    counter suffix on the rare case two backups land in the same minute.
    """
    registry = registry or ComponentRegistry()
    component = registry.get(component_name)
    if component is None:
        raise KeyError(f"No component named {component_name!r}")
    if not component.model_path:
        return None
    paths = for_component(component_name)
    current = paths.resolve_model_path(component.model_path)
    if not current.exists():
        return None

    timestamp = datetime.now().strftime(_BACKUP_TIMESTAMP_FMT)
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    backup_path = paths.models_dir / f"{current.stem}_bak{timestamp}{current.suffix}"
    suffix_n = 2
    while backup_path.exists():
        backup_path = paths.models_dir / f"{current.stem}_bak{timestamp}_{suffix_n}{current.suffix}"
        suffix_n += 1
    shutil.copy2(current, backup_path)
    logger.info("component=%s backed up current model %s -> %s", component_name, current.name, backup_path.name)
    return backup_path.relative_to(paths.root).as_posix()


def list_model_backups(component_name: str) -> list[dict[str, str]]:
    """Every backup model file in this component's models/ dir, newest
    first — for the Onboard page's rollback UI. Pure filesystem listing,
    matched by backup_current_model()'s own naming pattern, so a manually
    dropped-in file that happens to sit in models/ without that suffix is
    never mistaken for a backup.
    """
    paths = for_component(component_name)
    if not paths.models_dir.exists():
        return []
    backups = [p for p in paths.models_dir.iterdir() if p.is_file() and _BACKUP_STEM_RE.search(p.stem)]
    backups.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [
        {
            "filename": p.name,
            "path": p.relative_to(paths.root).as_posix(),
            "modified": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        }
        for p in backups
    ]


def restore_model_backup(
    component_name: str, backup_relative_path: str, *, registry: ComponentRegistry | None = None
) -> None:
    """Make a previously-saved backup the component's active model again.

    No file copy needed: the backup already sits in models/ and every
    predictor resolves the active model via
    ComponentPaths.resolve_model_path(component.model_path) (see
    utils/paths.py), so simply repointing model_path at the backup file
    makes it active immediately. Leaves anomaly_threshold untouched — a
    threshold tuned for whatever model trained most recently may not fit
    the restored one, but that's a manual retune (see the Inspect page's
    "Save as this component's default threshold"), not something this
    function should guess at.
    """
    registry = registry or ComponentRegistry()
    component = registry.get(component_name)
    if component is None:
        raise KeyError(f"No component named {component_name!r}")
    paths = for_component(component_name)
    backup_abs = paths.root / backup_relative_path
    if not backup_abs.exists():
        raise FileNotFoundError(f"Backup file not found: {backup_abs}")
    if backup_abs.resolve().parent != paths.models_dir.resolve():
        raise ValueError(f"Refusing to restore a file outside {paths.models_dir}: {backup_abs}")

    registry.update_training_result(
        component_name,
        anomaly_threshold=component.anomaly_threshold,
        model_path=backup_relative_path,
        status="ready",
    )
    logger.info("component=%s restored model backup %s as the active model", component_name, backup_abs.name)


# --- YOLO annotation onboarding --------------------------------------------
#
# Three annotation paths (pre-made YOLO labels, mask-to-box conversion,
# manual drawing) all converge on `add_yolo_annotation`, the single point
# that writes into the component's images/+labels/ pool. YoloTrainer only
# ever reads that pool — it has no idea which path produced any given label,
# and nothing outside this section branches on annotation source.


def create_yolo_component(
    display_name: str,
    *,
    class_names: list[str],
    registry: ComponentRegistry | None = None,
    **settings: Any,
) -> Component:
    """Register a new YOLO component and record its class list. Status starts at 'created'."""
    if not class_names:
        raise ValueError("A YOLO component needs at least one class name.")
    component = create_component(display_name, model_type="yolo", registry=registry, **settings)
    paths = for_component(component.name)
    yolo_annotation.write_classes_file(paths.yolo_classes_file, class_names)
    return component


def get_yolo_classes(component_name: str) -> list[str]:
    return yolo_annotation.read_classes_file(for_component(component_name).yolo_classes_file)


def add_yolo_classes(component_name: str, new_class_names: list[str]) -> list[str]:
    """Append new class name(s) to an existing YOLO component's class list.

    Every existing class keeps its index — see
    yolo_annotation.append_classes_file()'s own docstring for why that's
    the one invariant this can never break. Callers (onboarding UI,
    annotation paths) should re-fetch get_yolo_classes() after this rather
    than reuse a class list read before the call. Returns the full,
    updated class list.
    """
    paths = for_component(component_name)
    return yolo_annotation.append_classes_file(paths.yolo_classes_file, new_class_names)


def _resolve_annotation_filename(images_dir: Path, filename: str, image_bytes: bytes) -> str:
    """Picks the filename to save this upload's image (and matching label) under.

    If `filename` doesn't exist in the pool yet, or an existing file with
    that name has the *same* bytes (the user re-saving/correcting an
    annotation for an image already in the pool), reuse it as-is. If a
    *different* image happens to share the name — e.g. two separate upload
    batches that each used generic sequential filenames like 000.png,
    001.png (common for exported camera frames) — reusing it would silently
    overwrite the earlier image and its annotation with no warning. A
    numeric suffix is appended instead so nothing already saved is ever
    clobbered.
    """
    existing = images_dir / filename
    if not existing.exists() or existing.read_bytes() == image_bytes:
        return filename
    stem, suffix = Path(filename).stem, Path(filename).suffix
    n = 2
    while True:
        candidate = f"{stem}_{n}{suffix}"
        candidate_path = images_dir / candidate
        if not candidate_path.exists() or candidate_path.read_bytes() == image_bytes:
            return candidate
        n += 1


def add_yolo_annotation(
    component_name: str, filename: str, image_bytes: bytes, boxes: list[YoloBox]
) -> None:
    """Save one annotated image into the pool. `boxes` empty -> negative example.

    This is the convergence point for all three annotation paths (see module
    docstring above) — everything upstream of this call is presentation
    concern; everything downstream just sees images + label files.
    """
    paths = for_component(component_name)
    filename = _resolve_annotation_filename(paths.yolo_images_dir, filename, image_bytes)
    image_io.save_upload(image_bytes, paths.yolo_images_dir, filename)
    label_path = paths.yolo_labels_dir / f"{Path(filename).stem}.txt"
    yolo_annotation.write_yolo_label(label_path, boxes)


def add_yolo_labeled_pair(component_name: str, filename: str, image_bytes: bytes, label_text: str) -> None:
    """Annotation path 1: image + a ready-made YOLO-format .txt label, used as-is."""
    boxes: list[YoloBox] = []
    for line in label_text.splitlines():
        line = line.strip()
        if not line:
            continue
        cls_str, cx_str, cy_str, w_str, h_str = line.split()[:5]
        boxes.append((int(cls_str), float(cx_str), float(cy_str), float(w_str), float(h_str)))
    add_yolo_annotation(component_name, filename, image_bytes, boxes)


def convert_mask_to_boxes(mask_bytes: bytes, class_id: int = 0) -> list[YoloBox]:
    """Annotation path 2: run mask-to-bbox conversion. Caller (UI) reviews before saving."""
    mask_image = image_io.to_pil(mask_bytes).convert("L")
    return yolo_annotation.mask_to_yolo_boxes(np.array(mask_image), class_id=class_id)


def yolo_annotation_summary(component_name: str) -> dict:
    """Counts for onboarding validation/display: how many images, how many have boxes."""
    paths = for_component(component_name)
    if not paths.yolo_images_dir.exists():
        return {"image_count": 0, "positive_count": 0, "negative_count": 0}
    image_paths = [p for p in paths.yolo_images_dir.iterdir() if p.is_file()]
    positive = sum(
        1
        for p in image_paths
        if len(yolo_annotation.read_yolo_label(paths.yolo_labels_dir / f"{p.stem}.txt")) > 0
    )
    return {
        "image_count": len(image_paths),
        "positive_count": positive,
        "negative_count": len(image_paths) - positive,
    }


# --- Verified-label feedback loop --------------------------------------------
#
# The bridge between core/inspections/store.py's verified_* layer (a human's
# judgment about one inspection) and the YOLO annotation pool above (what
# training actually reads). Two directions, kept as a matched pair so the
# verified_label JSON shape can't drift between where it's written (the
# Inspection Station's flag/verify actions) and where it's read
# (incorporate_verified_corrections() below) — both sides import these
# two functions rather than each inventing their own shape.

_CORRECTION_OVERSAMPLE = 2  # copies added per corrected error (verified_incorrect)
_CONFIRMED_OVERSAMPLE = 1  # copies added per confirmed-correct example (verified_correct)


def build_verified_label(verdict: str, boxes: list[YoloBox], class_names: list[str]) -> dict:
    """The one place a YOLO verified_label's shape is defined: {"verdict":
    ..., "defect_classes": [...], "boxes": [{"class": ..., "box": [cx, cy,
    w, h]}, ...]}. `boxes` empty means a negative example (approved, or a
    corrected false positive) — the caller passes verdict="approved" and
    an empty list for that case, verdict="failed" with real boxes
    otherwise. Box class ids outside `class_names` are dropped rather than
    raising — the canvas widget that produces `boxes` already constrains
    choices to the current class list, so this is just defense in depth,
    not an expected path.
    """
    valid_boxes = [b for b in boxes if b[0] < len(class_names)]
    return {
        "verdict": verdict,
        "defect_classes": sorted({class_names[cls_id] for cls_id, *_rest in valid_boxes}),
        "boxes": [
            {"class": class_names[cls_id], "box": [cx, cy, w, h]} for cls_id, cx, cy, w, h in valid_boxes
        ],
    }


def verified_label_boxes_to_yolo(label: dict, class_names: list[str]) -> list[YoloBox]:
    """Inverse of build_verified_label()'s "boxes" shape, for materializing
    a verified_label back into the annotation pool. A box whose class name
    no longer exists in `class_names` (e.g. the class list changed since
    this label was saved) is skipped rather than raising — see
    incorporate_verified_corrections()'s own docstring for why a
    single stale entry shouldn't abort a whole consumption batch.
    """
    boxes: list[YoloBox] = []
    for entry in label.get("boxes") or []:
        cls_name = entry.get("class")
        if cls_name not in class_names:
            continue
        cx, cy, w, h = entry["box"]
        boxes.append((class_names.index(cls_name), cx, cy, w, h))
    return boxes


_UNSUPERVISED_MODEL_TYPES = ("autoencoder", "patchcore", "isolation_forest")


@dataclass(frozen=True)
class IncorporationResult:
    """Outcome of one incorporate_verified_corrections() call."""

    incorporated: int
    images_added: int
    skipped_unusable: int
    per_bucket_counts: dict[str, int] = field(default_factory=dict)


def list_pending_verified_corrections(component_name: str):
    """Every verified-but-not-yet-incorporated inspection for this
    component — the pool the Onboard page's manual_review selection UI
    picks from. A filter applied to list_verified_for_training()'s
    result (the sole query path — see its own docstring), not a
    parallel query.
    """
    return [
        r for r in inspection_store.list_verified_for_training(component_name)
        if r.verified_incorporation_status == "pending"
    ]


def incorporate_verified_corrections(
    component_name: str, inspection_ids: list[int], *, registry: ComponentRegistry | None = None
) -> IncorporationResult:
    """Copy the given human-verified inspections into this component's
    actual training/ data — where they become permanent, get picked up
    by every future training run automatically, and (per
    core/inspections/retention.py's cleanup_archived_inspections())
    become safe to clean up on the normal retention schedule even though
    the original inspection record's own lifecycle (acknowledged ->
    archived -> deleted) has nothing to do with it anymore — the training
    example now lives on independently in training/, not the inspection
    row.

    `inspection_ids` is an explicit, caller-chosen list — this function
    has no opinion on POLICY (see Component.verified_correction_policy:
    off/manual_review/automatic, in config/settings.py). The Onboard
    page's Train section decides which ids to pass: every pending one
    for 'automatic', an operator-picked subset for 'manual_review' (the
    real quality gate — a bad annotation gets caught by a human before
    it ever reaches training/), none at all for 'off'. This function is
    purely the mechanism, reusable regardless of which policy chose the
    list.

    Each id is re-validated directly against the database (via
    core.inspections.store.get(), never trusted from the caller as-is):
    must still be genuinely verified (a non-null label) and still
    'pending' — a stale, already-incorporated, or otherwise-invalid id
    is silently skipped rather than erroring the whole batch.

    Routing depends on the verified label's verdict/classes/boxes AND
    the component's model_type, following the exact directory/format
    each trainer already reads (confirmed by reading every trainer's own
    file-discovery code) — no special-case logic needed at training time:
    - YOLO: routed via _incorporate_for_yolo() — a 'failed' label with
      boxes becomes a positive annotated example (box + class, YOLO
      format) in training/yolo/; an 'approved' label becomes a negative
      example (no box).
    - classifier (binary, supervised on both approved AND failed
      images): routed via _incorporate_for_binary_folder() — 'approved'
      into training/approved/, 'failed' into training/failed/, both
      directly trainable.
    - autoencoder/patchcore/isolation_forest (unsupervised, fit ONLY on
      training/approved/ — every one of these trainers reads
      training/failed/ purely for post-hoc threshold/AUROC evaluation,
      never passes it to .fit()): also routed via
      _incorporate_for_binary_folder(), but a 'failed' label has nothing
      to be fit on there — these methods have no defect-learning step at
      all — so it's skipped rather than dumped into training/failed/ as
      if it were trainable defect data; logged clearly instead (still
      meaningful evidence for a human threshold review, just not
      automatically actionable the training-data way).

    Corrections are traceably named
    (`correction_<inspection id>_copy<n>.<ext>`) directly inside the
    existing flat directory structure — never a subfolder: confirmed
    that autoencoder/classifier/isolation_forest all scan their
    directories non-recursively (a subfolder would be silently invisible
    to them), while only PatchCore's anomalib backend happens to recurse.
    A flat, prefixed filename is the one convention guaranteed to be
    picked up identically by every model_type's trainer, and stays
    trivially greppable/removable later if a specific correction turns
    out to have been wrong.

    Weighting ("corrected errors weigh heaviest"): a verified_incorrect
    correction is added _CORRECTION_OVERSAMPLE times, a verified_correct
    confirmation _CONFIRMED_OVERSAMPLE time — literal file duplication
    (distinct filenames) is the only practical way to weight an
    example's influence given none of these training loops expose a
    per-sample-weight hook.
    """
    registry = registry or ComponentRegistry()
    component = registry.get(component_name)
    if component is None:
        raise KeyError(f"No component named {component_name!r}")

    records: list[InspectionRecord] = []
    for inspection_id in inspection_ids:
        record = inspection_store.get(inspection_id)
        if record is None:
            logger.warning("component=%s inspection id=%s no longer exists — skipping", component_name, inspection_id)
            continue
        if record.verified_status not in ("verified_correct", "verified_incorrect") or not record.verified_label:
            logger.warning(
                "component=%s inspection id=%s is not actually verified — skipping incorporation",
                component_name, inspection_id,
            )
            continue
        if record.verified_incorporation_status != "pending":
            continue  # already incorporated earlier — never double-add
        records.append(record)

    paths = for_component(component_name)
    if component.model_type == "yolo":
        return _incorporate_for_yolo(component, records, paths)
    return _incorporate_for_binary_folder(component, records, paths)


def _incorporate_for_yolo(
    component: Component, records: list[InspectionRecord], paths: ComponentPaths
) -> IncorporationResult:
    class_names = get_yolo_classes(component.name)
    consumed_ids: list[int] = []
    images_added = 0
    skipped_unusable = 0
    per_bucket_counts: dict[str, int] = {}

    for record in records:
        label = record.verified_label or {}
        boxes = verified_label_boxes_to_yolo(label, class_names)

        if label.get("verdict") == "failed" and not boxes:
            logger.warning(
                "component=%s inspection id=%s verified_correct/incorrect with verdict=failed but no "
                "usable boxes — cannot materialize as YOLO training data, leaving 'pending'",
                component.name, record.id,
            )
            skipped_unusable += 1
            continue

        image_abs = paths.root / record.image_path if record.image_path else None
        if image_abs is None or not image_abs.exists():
            logger.warning(
                "component=%s inspection id=%s has no image on disk (%s) — skipping, leaving 'pending'",
                component.name, record.id, record.image_path,
            )
            continue

        image_bytes = image_abs.read_bytes()
        copies = _CORRECTION_OVERSAMPLE if record.verified_status == "verified_incorrect" else _CONFIRMED_OVERSAMPLE
        for i in range(copies):
            filename = f"correction_{record.id}_copy{i}{image_abs.suffix}"
            add_yolo_annotation(component.name, filename, image_bytes, boxes)
            images_added += 1
            if boxes:
                for cls_id, *_rest in boxes:
                    class_name = class_names[cls_id]
                    per_bucket_counts[class_name] = per_bucket_counts.get(class_name, 0) + 1
            else:
                per_bucket_counts["(negative)"] = per_bucket_counts.get("(negative)", 0) + 1

        consumed_ids.append(record.id)
        logger.info(
            "component=%s inspection id=%s (%s) incorporated into training/yolo/: %d copy/copies, %d box(es)",
            component.name, record.id, record.verified_status, copies, len(boxes),
        )

    inspection_store.mark_incorporated(consumed_ids)
    return IncorporationResult(
        incorporated=len(consumed_ids),
        images_added=images_added,
        skipped_unusable=skipped_unusable,
        per_bucket_counts=per_bucket_counts,
    )


def _incorporate_for_binary_folder(
    component: Component, records: list[InspectionRecord], paths: ComponentPaths
) -> IncorporationResult:
    consumed_ids: list[int] = []
    images_added = 0
    skipped_unusable = 0
    per_bucket_counts: dict[str, int] = {}

    for record in records:
        label = record.verified_label or {}
        verdict = label.get("verdict")

        if verdict == "approved":
            target_dir, bucket = paths.training_approved_dir, "approved"
        elif verdict == "failed" and component.model_type in _UNSUPERVISED_MODEL_TYPES:
            logger.info(
                "component=%s inspection id=%s is a verified failed/defect correction, but %s is "
                "unsupervised and never trains on failed/defect images (training_failed_dir is only "
                "used for post-hoc threshold/AUROC evaluation, never passed to .fit()) — this can't "
                "become training data the way a YOLO box or classifier label would. Leaving "
                "'pending'; may be worth reviewing this component's anomaly threshold instead.",
                component.name, record.id, component.model_type,
            )
            skipped_unusable += 1
            continue
        elif verdict == "failed":
            target_dir, bucket = paths.training_failed_dir, "failed"
        else:
            logger.warning(
                "component=%s inspection id=%s has an unrecognized verified_label verdict %r — skipping",
                component.name, record.id, verdict,
            )
            skipped_unusable += 1
            continue

        image_abs = paths.root / record.image_path if record.image_path else None
        if image_abs is None or not image_abs.exists():
            logger.warning(
                "component=%s inspection id=%s has no image on disk (%s) — skipping, leaving 'pending'",
                component.name, record.id, record.image_path,
            )
            continue

        image_bytes = image_abs.read_bytes()
        copies = _CORRECTION_OVERSAMPLE if record.verified_status == "verified_incorrect" else _CONFIRMED_OVERSAMPLE
        for i in range(copies):
            filename = f"correction_{record.id}_copy{i}{image_abs.suffix}"
            image_io.save_upload(image_bytes, target_dir, filename)
            images_added += 1
            per_bucket_counts[bucket] = per_bucket_counts.get(bucket, 0) + 1

        consumed_ids.append(record.id)
        logger.info(
            "component=%s inspection id=%s (%s) incorporated into training/%s/: %d copy/copies",
            component.name, record.id, record.verified_status, bucket, copies,
        )

    inspection_store.mark_incorporated(consumed_ids)
    return IncorporationResult(
        incorporated=len(consumed_ids),
        images_added=images_added,
        skipped_unusable=skipped_unusable,
        per_bucket_counts=per_bucket_counts,
    )
