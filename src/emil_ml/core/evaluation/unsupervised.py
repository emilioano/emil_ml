"""The shared evaluation procedure for every unsupervised anomaly method
(autoencoder, PatchCore, Isolation Forest) — see core/evaluation/__init__.py
for why this one procedure is genuinely shared while loss-curve plotting
is not.

These methods train only on approved images, so there is no training-time
accuracy to report. What they DO have, if labeled failed examples are
available (never trained on — only ever used for calibration/validation,
see each trainer's own module docstring), is a real, labeled test set:
approved images the model has seen, and failed images it hasn't. That's
exactly the "test set with both classes" this module evaluates against —
threshold-based precision/recall/F1, an ROC and/or PR curve over
thresholds, and a histogram of the two classes' score distributions.

Without labeled failed examples, only the histogram (approved-only) is
produced, and `notes` says so explicitly — a threshold-based class metric
is meaningless without knowing what a failed example's score even looks
like.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from emil_ml.core.evaluation import plots


def evaluate_anomaly_scores(
    approved_scores: list[float],
    failed_scores: list[float] | None,
    threshold: float | None,
    output_dir: Path,
    *,
    title_prefix: str = "",
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Returns (metrics, plot_files, notes) — the three pieces every
    unsupervised trainer's evaluate() folds into its own EvaluationResult
    (see core/base.py). `title_prefix` (e.g. "PatchCore ") only affects
    plot titles, purely cosmetic.
    """
    plot_files: list[str] = []
    metrics: dict[str, Any] = {}
    notes: list[str] = []

    plots.plot_score_histogram(
        approved_scores, failed_scores or [], threshold,
        output_dir / "score_histogram.png",
        title=f"{title_prefix}anomaly score: approved vs failed".strip(),
    )
    plot_files.append("score_histogram.png")

    if not failed_scores:
        notes.append(
            "No labeled failed examples were available for this component — threshold-based "
            "precision/recall/F1 and ROC/PR curves require a labeled test set with both classes. "
            "Only the approved-only score distribution was generated."
        )
        return metrics, plot_files, notes

    from sklearn.metrics import auc, precision_recall_curve, roc_curve

    y_true = np.concatenate([np.zeros(len(approved_scores)), np.ones(len(failed_scores))])
    y_score = np.concatenate([np.asarray(approved_scores, dtype=float), np.asarray(failed_scores, dtype=float)])

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = float(auc(fpr, tpr))
    plots.plot_roc_curve(fpr, tpr, roc_auc, output_dir / "roc_curve.png", title=f"{title_prefix}ROC curve".strip())
    plot_files.append("roc_curve.png")

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plots.plot_pr_curve(precision, recall, output_dir / "pr_curve.png", title=f"{title_prefix}precision-recall curve".strip())
    plot_files.append("pr_curve.png")

    if threshold is not None:
        y_pred = (y_score > threshold).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        threshold_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        threshold_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        threshold_f1 = (
            2 * threshold_precision * threshold_recall / (threshold_precision + threshold_recall)
            if (threshold_precision + threshold_recall) > 0
            else 0.0
        )
        metrics["confusion_matrix"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
        metrics["threshold_precision"] = threshold_precision
        metrics["threshold_recall"] = threshold_recall
        metrics["threshold_f1"] = threshold_f1

    metrics["roc_auc"] = roc_auc
    return metrics, plot_files, notes
