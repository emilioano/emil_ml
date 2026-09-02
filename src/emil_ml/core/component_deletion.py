"""Permanent component destruction — the second half of the delete
lifecycle, after ComponentRegistry.soft_delete() (config/registry.py).

Deliberately its own top-level module, not folded into config/registry.py
or any single core/ subpackage: this is the highest-blast-radius code in
the whole project, reaching into every layer a component owns data in
(filesystem, ChromaDB, the inspections/machine_readings tables, and
finally the registry row itself) to erase all of it in one deliberate
operation. core/inspections/retention.py earned its own module for the
exact same reason at the inspection level; this is that same posture one
level up, at the component level.

Not wired into any scheduler or automatic trigger for the actual
deletion — permanently_delete_component() is only ever called from an
explicit, confirmed UI action (the Onboard page's trash view) or
cleanup_expired_soft_deleted_components(), itself only run via an
explicit "Run cleanup now" button, never automatically. Same "no
destructive action without a deliberate click" posture as every other
consequential action in this project.

Two-phase by design: a component must already be soft-deleted
(lifecycle_status='deleted', via ComponentRegistry.soft_delete()) before
permanently_delete_component() will touch it at all — there is no path
from "active" or "inactive" straight to permanent destruction. That
first phase is a pure, zero-risk registry flag flip (see registry.py);
everything genuinely destructive lives here, gated behind it.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from emil_ml.config.registry import ComponentRegistry
from emil_ml.config.settings import DEFAULT_COMPONENT_DELETION_RETENTION_DAYS
from emil_ml.core.cascade import policy_store
from emil_ml.core.cascade import stream_store as cascade_stream_store
from emil_ml.core.inspections import store as inspection_store
from emil_ml.core.reporting.knowledge import indexer
from emil_ml.core.reporting.machine_context.source import SqliteMachineContextSource
from emil_ml.core.training_runs import store as training_runs_store
from emil_ml.training.onboard import list_model_backups
from emil_ml.utils.paths import for_component

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_DELETED_AT_FMT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class DeletionImpactSummary:
    """What's actually at stake if a soft-deleted component is
    permanently destroyed — the confirmation UI's entire reason for
    existing. `verified_correction_count` is called out on its own
    (rather than folded into inspection_count) because it's the one
    number that represents irreplaceable human judgment, not just
    re-creatable machine output.
    """

    component_name: str
    display_name: str
    inspection_count: int
    verified_correction_count: int
    has_active_model: bool
    model_backup_count: int
    training_file_count: int
    training_run_count: int
    knowledge_document_count: int
    chromadb_chunk_count: int
    machine_reading_count: int
    cascade_stream_result_count: int
    reaction_policy_count: int


@dataclass(frozen=True)
class PermanentDeletionResult:
    """Outcome of one permanently_delete_component() call — a report,
    not just a bool, so a caller can show exactly what got cleaned up
    and surface anything that didn't (without treating a partial failure
    as silent success)."""

    component_name: str
    filesystem_removed: bool
    chromadb_chunks_removed: int
    inspections_removed: int
    machine_readings_removed: int
    training_runs_removed: int
    cascade_stream_rows_removed: int
    reaction_policies_removed: int
    registry_row_removed: bool
    already_complete: bool = False  # True if a prior run had already finished everything
    errors: list[str] = field(default_factory=list)


def summarize_deletion_impact(component_name: str, *, registry: ComponentRegistry | None = None) -> DeletionImpactSummary:
    """Read-only — computes what WOULD be lost, touches nothing. Used by
    the trash view's "delete permanently" confirmation dialog so an
    operator sees real numbers (especially verified corrections, the
    irreplaceable ones) before confirming, not just a bare "are you
    sure?".
    """
    registry = registry or ComponentRegistry()
    component = registry.get(component_name)
    if component is None:
        raise KeyError(f"No component named {component_name!r}")

    paths = for_component(component_name)
    inspections = inspection_store.list_all(component_name=component_name, limit=None)
    verified_count = len(inspection_store.list_verified_for_training(component_name))

    training_file_count = 0
    if paths.training_dir.exists():
        training_file_count = sum(
            1 for p in paths.training_dir.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
        )

    knowledge_document_count = 0
    if paths.knowledge_dir.exists():
        knowledge_document_count = sum(1 for p in paths.knowledge_dir.iterdir() if p.is_file())

    return DeletionImpactSummary(
        component_name=component_name,
        display_name=component.display_name,
        inspection_count=len(inspections),
        verified_correction_count=verified_count,
        has_active_model=component.model_path is not None,
        model_backup_count=len(list_model_backups(component_name)),
        training_file_count=training_file_count,
        training_run_count=len(training_runs_store.list_for_component(component_name)),
        knowledge_document_count=knowledge_document_count,
        chromadb_chunk_count=indexer.count_component_chunks(component_name),
        machine_reading_count=SqliteMachineContextSource().count_readings(component_name),
        cascade_stream_result_count=cascade_stream_store.count_results_for_component(component_name),
        reaction_policy_count=len(policy_store.list_policies(component_name)),
    )


def permanently_delete_component(
    component_name: str, *, registry: ComponentRegistry | None = None
) -> PermanentDeletionResult:
    """Irreversibly erase every trace of this component: its entire
    filesystem tree (training/, models/ including backups, input/,
    analyzed/, archive/, knowledge/, corrections), every ChromaDB chunk
    tagged with its component_type, every row in inspections and
    machine_readings that references it, and finally the registry row
    itself.

    Requires the component to already be soft-deleted
    (lifecycle_status='deleted' — see ComponentRegistry.soft_delete());
    raises otherwise. This is the one hard precondition standing between
    "active or inactive" and "gone forever" — there is no direct path
    that skips it.

    Resumable by construction: each of the four cleanup steps below is
    independently idempotent (removing an already-gone directory,
    deleting from an already-empty ChromaDB filter, deleting 0 remaining
    DB rows are all safe no-ops), and the registry row is removed LAST,
    deliberately — as long as it still exists, a caller can re-run this
    function after an interruption (a crashed process, a killed script)
    and it will pick up exactly where it left off, re-attempting only
    whatever didn't finish. If the registry row is already gone when
    this is called, that means a prior run already completed every step
    — this returns immediately with `already_complete=True` rather than
    raising or ambiguously trying to redo work that already happened.

    One failing step is logged into `errors` and does not abort the
    rest — a filesystem permission error, for instance, shouldn't
    prevent the ChromaDB/DB cleanup from still happening.

    THE GUARANTEE THIS EXISTS TO UPHOLD: after this returns with no
    errors, a new component created later with the same slug inherits
    NOTHING from this one — no leftover files, no stale ChromaDB chunks
    (exactly the cross-component leak family this project has already
    hit once, from misplaced files rather than incomplete deletion, but
    the failure mode looks identical from a retrieval query's
    perspective), no orphaned inspection history.
    """
    registry = registry or ComponentRegistry()
    component = registry.get(component_name)

    if component is None:
        logger.info("component=%s already fully deleted (no registry row found) — nothing to do", component_name)
        return PermanentDeletionResult(
            component_name=component_name,
            filesystem_removed=True,
            chromadb_chunks_removed=0,
            inspections_removed=0,
            machine_readings_removed=0,
            training_runs_removed=0,
            cascade_stream_rows_removed=0,
            reaction_policies_removed=0,
            registry_row_removed=True,
            already_complete=True,
        )

    if component.lifecycle_status != "deleted":
        raise ValueError(
            f"Component {component_name!r} must be soft-deleted first (lifecycle_status='deleted'); "
            f"currently {component.lifecycle_status!r}. Call registry.soft_delete() before this."
        )

    errors: list[str] = []

    filesystem_removed = False
    try:
        paths = for_component(component_name)
        if paths.root.exists():
            shutil.rmtree(paths.root)
        filesystem_removed = True
    except Exception as exc:  # noqa: BLE001 - one failing step must not abort the rest of the cleanup
        logger.exception("component=%s permanent delete: filesystem cleanup failed", component_name)
        errors.append(f"filesystem: {exc}")

    chromadb_chunks_removed = 0
    try:
        chromadb_chunks_removed = indexer.delete_component_chunks(component_name)
    except Exception as exc:  # noqa: BLE001
        logger.exception("component=%s permanent delete: ChromaDB cleanup failed", component_name)
        errors.append(f"chromadb: {exc}")

    inspections_removed = 0
    try:
        inspections_removed = inspection_store.delete_all_for_component(component_name)
    except Exception as exc:  # noqa: BLE001
        logger.exception("component=%s permanent delete: inspections cleanup failed", component_name)
        errors.append(f"inspections: {exc}")

    machine_readings_removed = 0
    try:
        machine_readings_removed = SqliteMachineContextSource().delete_readings(component_name)
    except Exception as exc:  # noqa: BLE001
        logger.exception("component=%s permanent delete: machine_readings cleanup failed", component_name)
        errors.append(f"machine_readings: {exc}")

    training_runs_removed = 0
    try:
        training_runs_removed = training_runs_store.delete_all_for_component(component_name)
    except Exception as exc:  # noqa: BLE001
        logger.exception("component=%s permanent delete: training_runs cleanup failed", component_name)
        errors.append(f"training_runs: {exc}")

    cascade_stream_rows_removed = 0
    try:
        cascade_stream_rows_removed = cascade_stream_store.delete_all_for_component(component_name)
    except Exception as exc:  # noqa: BLE001
        logger.exception("component=%s permanent delete: cascade_stream cleanup failed", component_name)
        errors.append(f"cascade_stream: {exc}")

    reaction_policies_removed = 0
    try:
        reaction_policies_removed = policy_store.delete_all_for_component(component_name)
    except Exception as exc:  # noqa: BLE001
        logger.exception("component=%s permanent delete: reaction_policies cleanup failed", component_name)
        errors.append(f"reaction_policies: {exc}")

    # The registry row — last, deliberately (see docstring: this is what
    # makes an interrupted-and-resumed deletion safe).
    registry_row_removed = False
    try:
        registry.delete(component_name)
        registry_row_removed = True
    except Exception as exc:  # noqa: BLE001
        logger.exception("component=%s permanent delete: registry row removal failed", component_name)
        errors.append(f"registry: {exc}")

    logger.info(
        "component=%s permanent deletion %s: filesystem=%s chromadb_chunks=%d inspections=%d "
        "machine_readings=%d training_runs=%d cascade_stream_rows=%d reaction_policies=%d registry=%s errors=%d",
        component_name, "complete" if not errors else "completed with errors",
        filesystem_removed, chromadb_chunks_removed, inspections_removed, machine_readings_removed,
        training_runs_removed, cascade_stream_rows_removed, reaction_policies_removed, registry_row_removed, len(errors),
    )
    return PermanentDeletionResult(
        component_name=component_name,
        filesystem_removed=filesystem_removed,
        chromadb_chunks_removed=chromadb_chunks_removed,
        inspections_removed=inspections_removed,
        machine_readings_removed=machine_readings_removed,
        training_runs_removed=training_runs_removed,
        cascade_stream_rows_removed=cascade_stream_rows_removed,
        reaction_policies_removed=reaction_policies_removed,
        registry_row_removed=registry_row_removed,
        errors=errors,
    )


def cleanup_expired_soft_deleted_components(
    *, registry: ComponentRegistry | None = None, now: datetime | None = None
) -> dict[str, PermanentDeletionResult]:
    """Permanently delete every soft-deleted component whose deleted_at
    is older than DEFAULT_COMPONENT_DELETION_RETENTION_DAYS — the trash
    view's "Run cleanup now" button. Not run automatically (see module
    docstring); an operator can also permanently delete a specific
    component immediately regardless of this window via
    permanently_delete_component() directly, for when they're already
    certain.

    `now` is an injectable override (defaults to the real current time)
    purely so this function's age logic can be tested deterministically
    without waiting real days.
    """
    registry = registry or ComponentRegistry()
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=DEFAULT_COMPONENT_DELETION_RETENTION_DAYS)

    results: dict[str, PermanentDeletionResult] = {}
    for component in registry.list_deleted():
        if not component.deleted_at:
            continue
        try:
            deleted_at = datetime.strptime(component.deleted_at, _DELETED_AT_FMT).replace(tzinfo=timezone.utc)
        except ValueError:
            logger.warning(
                "component=%s unparseable deleted_at %r — skipping this cleanup pass",
                component.name, component.deleted_at,
            )
            continue
        if deleted_at > cutoff:
            continue  # not old enough yet
        results[component.name] = permanently_delete_component(component.name, registry=registry)
    return results
