"""Retention cleanup: the third and final inspection lifecycle step,
archived -> deleted (see lifecycle.py's own module docstring for the
first two: new -> acknowledged -> archived).

Deliberately its own module, not folded into lifecycle.py or store.py:
this is the one place in the whole project that actually deletes an
inspection's files AND its database row — every other action anywhere
else in core/inspections only ever moves a record between states (see
store.py's module docstring). That makes this the highest-blast-radius
code in the inspections package, and it earns a module of its own rather
than blending in next to functions that never delete anything.

Not wired into any scheduler or automatic trigger — this is a callable
capability the Onboard page exposes as an explicit, manually-triggered
button ("Run retention cleanup now"), the same "no destructive action
without a deliberate click" posture every other consequential action in
this project already follows (train, restore a model backup, bulk
archive, ...).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from emil_ml.config.registry import ComponentRegistry
from emil_ml.core.inspections import store
from emil_ml.utils.paths import for_component

logger = logging.getLogger(__name__)

_ARCHIVED_AT_FMT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class RetentionResult:
    """Outcome of one cleanup_archived_inspections() call — a report, not
    just a count, so a caller can show WHY the archive isn't shrinking as
    much as expected (protected records) rather than that being a silent
    mystery."""

    deleted: int
    protected_pending_verified: int
    errors: list[tuple[int, str]] = field(default_factory=list)


def cleanup_archived_inspections(
    component_name: str,
    *,
    registry: ComponentRegistry | None = None,
    now: datetime | None = None,
) -> RetentionResult:
    """Permanently delete this component's archived inspections older
    than its own `inspection_retention_days` setting — both the image/
    report files on disk and the database row (see store.delete(), the
    only function that actually removes a row).

    `inspection_retention_days <= 0` means "keep archived inspections
    indefinitely" (retention disabled), not "delete immediately" — the
    safer reading for a setting a component owner might leave at its
    minimum (0) without meaning to nuke every archived record on the
    next cleanup run.

    THE PROTECTION THIS FUNCTION EXISTS TO ENFORCE: an inspection with a
    human-verified label (verified_status in ('verified_correct',
    'verified_incorrect')) whose verified_incorporation_status is still
    'pending' is NEVER deleted here, no matter how old it is. That's real,
    not-yet-used training data — a correction that gets archived and then
    deleted before a retrain ever consumes it (see training/onboard.py's
    incorporate_verified_corrections(), which is what eventually
    flips it to 'incorporated') is gone forever: the image, the label,
    everything. That would silently defeat the entire point of the
    correction feedback loop, so this check comes before the age check
    even applies — no amount of retention_days makes a pending
    verification eligible. Once it's 'incorporated', the value has
    already been realized and normal retention applies exactly as it
    would to any other archived record.

    Every record skipped for this reason is logged at WARNING (so
    protected records accumulating is visible, not a silent mystery —
    see also pending_verified_counts() below, which the Inspection
    Station surfaces for the same reason) and counted in the returned
    result's `protected_pending_verified`.

    A record that fails to delete (e.g. a file already gone, a
    permissions error) is logged and counted in `errors`; it does not
    abort the rest of the sweep — one bad record shouldn't block cleanup
    of everything else due for deletion.

    `now` is an injectable override (defaults to the real current time)
    purely so this function's age logic can be tested deterministically
    without waiting real days.
    """
    registry = registry or ComponentRegistry()
    component = registry.get(component_name)
    if component is None:
        raise KeyError(f"No component named {component_name!r}")

    retention_days = component.inspection_retention_days
    if retention_days <= 0:
        return RetentionResult(deleted=0, protected_pending_verified=0)

    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=retention_days)
    paths = for_component(component_name)

    archived_records = store.list_all(component_name=component_name, status="archived", limit=None)

    deleted = 0
    protected = 0
    errors: list[tuple[int, str]] = []

    for record in archived_records:
        if not record.archived_at:
            continue
        try:
            archived_at = datetime.strptime(record.archived_at, _ARCHIVED_AT_FMT).replace(tzinfo=timezone.utc)
        except ValueError as exc:
            errors.append((record.id, f"unparseable archived_at {record.archived_at!r}: {exc}"))
            continue
        if archived_at > cutoff:
            continue  # not old enough yet

        if (
            record.verified_status in ("verified_correct", "verified_incorrect")
            and record.verified_incorporation_status == "pending"
        ):
            protected += 1
            logger.warning(
                "component=%s retention SKIPPED inspection id=%s (archived %s, older than the "
                "%d-day retention window) — has a human-verified label not yet incorporated into "
                "training; it will become eligible for deletion once a retrain consumes it "
                "(see training/onboard.py's incorporate_verified_corrections())",
                component_name, record.id, record.archived_at, retention_days,
            )
            continue

        try:
            for rel_path in (record.image_path, record.report_path):
                if rel_path:
                    (paths.root / rel_path).unlink(missing_ok=True)
            store.delete(record.id)
            deleted += 1
        except Exception as exc:  # noqa: BLE001 - one bad record must not abort the rest of the sweep
            logger.exception("component=%s retention: failed to delete inspection id=%s", component_name, record.id)
            errors.append((record.id, str(exc)))

    logger.info(
        "component=%s retention cleanup complete: deleted=%d protected_pending_verified=%d errors=%d",
        component_name, deleted, protected, len(errors),
    )
    return RetentionResult(deleted=deleted, protected_pending_verified=protected, errors=errors)


def cleanup_all_components(
    *, registry: ComponentRegistry | None = None, now: datetime | None = None
) -> dict[str, RetentionResult]:
    """Convenience wrapper for the Onboard page's single "Run retention
    cleanup now" button — runs cleanup_archived_inspections() once per
    registered component, keyed by component name. Not a different code
    path: purely a loop over the single-component function above, so
    every guarantee (verified-and-pending protection, per-record error
    isolation) applies identically per component.
    """
    registry = registry or ComponentRegistry()
    return {
        component.name: cleanup_archived_inspections(component.name, registry=registry, now=now)
        for component in registry.list_all()
    }


def pending_verified_counts(registry: ComponentRegistry | None = None) -> dict[str, int]:
    """Component name -> count of verified inspections not yet
    incorporated into training, across every component that has any —
    for the Inspection Station to show "N pending verified correction(s)
    waiting for a retrain" instead of letting retention-protected records
    silently accumulate with no visible explanation.

    Built entirely on list_verified_for_training() — the sole query path
    for verified data (see its own docstring) — aggregated per component
    for display; not a parallel or competing query.
    """
    registry = registry or ComponentRegistry()
    counts: dict[str, int] = {}
    for component in registry.list_all():
        pending = [
            r for r in store.list_verified_for_training(component.name)
            if r.verified_incorporation_status == "pending"
        ]
        if pending:
            counts[component.name] = len(pending)
    return counts
