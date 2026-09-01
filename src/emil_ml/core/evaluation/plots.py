"""Shared matplotlib plotting helpers — see core/evaluation/__init__.py.

Forces the non-interactive "Agg" backend before importing pyplot: this
always runs inside a Streamlit server process or the watcher, never with
a display attached, and setting the backend after pyplot has already
been imported elsewhere in the process is unreliable — doing it here, at
this module's own top level, before pyplot is touched, is what makes it
safe regardless of import order.

Every function here takes plain Python/numpy values and a destination
Path, draws exactly one figure, saves it, and closes it (matplotlib
figures are not garbage-collected automatically — leaving them open
across many training runs in the same long-lived Streamlit process would
leak memory).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_DPI = 120


def plot_confusion_matrix(
    cm: np.ndarray, class_names: list[str], path: Path, *, title: str = "Confusion matrix"
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    thresh = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(int(cm[i, j])), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)


def plot_curves(
    series: dict[str, list[float]],
    path: Path,
    *,
    title: str,
    xlabel: str = "Epoch",
    ylabel: str = "Value",
    phase_boundary: int | None = None,
) -> None:
    """One or more named series (e.g. {"loss": [...], "val_loss": [...]}),
    each plotted against its own 1-indexed position. `phase_boundary`
    (optional) draws a vertical marker after that many epochs — used by
    the classifier's evaluate() to mark where fine-tuning took over from
    head-only training within one continuous curve.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, values in series.items():
        if values:
            ax.plot(range(1, len(values) + 1), values, label=label)
    if phase_boundary is not None and phase_boundary > 0:
        ax.axvline(phase_boundary + 0.5, color="gray", linestyle="--", linewidth=1, label="fine-tune starts")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)


def plot_score_histogram(
    approved_scores: list[float],
    failed_scores: list[float],
    threshold: float | None,
    path: Path,
    *,
    title: str = "Anomaly score distribution",
) -> None:
    """Always safe to call, even with one or both lists empty (an empty
    list just contributes nothing to the plot — see each unsupervised
    trainer's evaluate(), which always has approved scores but may not
    have failed ones)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = 30
    if approved_scores:
        ax.hist(approved_scores, bins=bins, alpha=0.6, label=f"approved (n={len(approved_scores)})", color="tab:green")
    if failed_scores:
        ax.hist(failed_scores, bins=bins, alpha=0.6, label=f"failed (n={len(failed_scores)})", color="tab:red")
    if threshold is not None:
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"threshold={threshold:.4g}")
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, path: Path, *, title: str = "ROC curve") -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)


def plot_pr_curve(precision: np.ndarray, recall: np.ndarray, path: Path, *, title: str = "Precision-recall curve") -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-0.02, 1.02)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
