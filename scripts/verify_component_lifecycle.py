"""Verifies deactivate/reactivate, soft-delete/restore, and permanent
deletion on a disposable throwaway component with a real, rich footprint:
a knowledge document (indexed into ChromaDB), training images, an
inspection (with a verified correction), and a machine reading — so
permanent deletion has something real to clean up in every layer.

Run with: python scripts/verify_component_lifecycle.py
"""

from __future__ import annotations

import io
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from PIL import Image as PILImage

from emil_ml.config.logging_config import configure_logging
from emil_ml.config.registry import ComponentRegistry
from emil_ml.core import component_deletion
from emil_ml.core.inspections import store
from emil_ml.core.reporting.knowledge import indexer
from emil_ml.core.reporting.machine_context.source import SqliteMachineContextSource
from emil_ml.core.training_runs import store as training_runs_store
from emil_ml.training import onboard
from emil_ml.utils.paths import for_component
from emil_ml.watcher.service import WatcherService

COMPONENT_DISPLAY_NAME = "Lifecycle Test Component"


def _make_image_bytes() -> bytes:
    rng = np.random.default_rng(0)
    arr = np.clip(np.full((32, 32, 3), 120, dtype=np.int16) + rng.integers(-10, 10, (32, 32, 3)), 0, 255).astype("uint8")
    buf = io.BytesIO()
    PILImage.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _setup_component(registry: ComponentRegistry) -> str:
    component = onboard.create_component(COMPONENT_DISPLAY_NAME, model_type="classifier", registry=registry)
    name = component.name
    paths = for_component(name)

    # Training images.
    onboard.add_training_images(name, approved=[("a1.png", _make_image_bytes()), ("a2.png", _make_image_bytes())])

    # Knowledge document, indexed into ChromaDB.
    paths.knowledge_dir.mkdir(parents=True, exist_ok=True)
    (paths.knowledge_dir / "manual.md").write_text(
        "---\ndoc_type: manual\nsource: Test Manual\n---\n# Section\nSome test content for indexing.\n",
        encoding="utf-8",
    )
    registry.update_settings(name, reporting_enabled=True)
    indexer.index_component_type(name)

    # A real inspection with a verified correction.
    image_bytes = _make_image_bytes()
    paths.analyzed_approved_dir.mkdir(parents=True, exist_ok=True)
    image_path = paths.analyzed_approved_dir / "inspected.png"
    image_path.write_bytes(image_bytes)
    record = store.create(
        name, verdict="approved", score=0.1, threshold=0.5,
        image_path=image_path.relative_to(paths.root).as_posix(), run_by="lifecycle-test",
    )
    store.verify(record.id, status="verified_correct", label={"verdict": "approved", "defect_classes": [], "boxes": []}, by="qa-reviewer")

    # A machine reading.
    SqliteMachineContextSource().insert_reading(name, {"temperature": 50.0})

    # A recorded training run (as train_component() would leave behind).
    training_runs_store.create(
        name, display_name=COMPONENT_DISPLAY_NAME, modality="vision", model_type="classifier",
        status="success", threshold=0.5, metrics={"accuracy": 0.9},
    )

    return name


def main() -> None:
    configure_logging()
    registry = ComponentRegistry()
    all_pass = True
    name = _setup_component(registry)
    paths = for_component(name)

    try:
        # === 1: deactivate — hidden from active use, preserved, reversible ===
        print("=== 1: deactivate() excludes from list_active(), preserves everything, reversible ===")
        registry.deactivate(name)
        component = registry.get(name)
        ok1a = component.lifecycle_status == "inactive"
        ok1b = name not in {c.name for c in registry.list_active()}
        ok1c = name in {c.name for c in registry.list_all()}  # still in general management listing
        ok1d = paths.root.exists() and indexer.count_component_chunks(name) > 0
        ok1 = ok1a and ok1b and ok1c and ok1d
        print(f"  lifecycle_status={component.lifecycle_status} excluded from list_active: {ok1b}")
        print(f"  still in list_all: {ok1c}, filesystem+chromadb intact: {ok1d}")
        print(f"-> {'PASS' if ok1 else 'FAIL'}")
        all_pass &= ok1

        # Watcher respects it dynamically.
        watcher_registry = registry
        watcher = WatcherService(registry=watcher_registry)
        watcher._sync_watches()
        watched_names = set(watcher._watched_dirs.values())
        ok1e = name not in watched_names
        print(f"  watcher watching this component while inactive: {name in watched_names} (expected False)")
        print(f"-> {'PASS' if ok1e else 'FAIL'}: watcher does not watch a deactivated component.")
        all_pass &= ok1e
        print()

        # === 2: reactivate — back to normal ===
        print("=== 2: reactivate() ===")
        registry.reactivate(name)
        component = registry.get(name)
        ok2 = component.lifecycle_status == "active" and name in {c.name for c in registry.list_active()}
        print(f"  lifecycle_status={component.lifecycle_status}")
        print(f"-> {'PASS' if ok2 else 'FAIL'}")
        all_pass &= ok2

        watcher._sync_watches()
        watched_names = set(watcher._watched_dirs.values())
        ok2b = name in watched_names
        print(f"  watcher watching this component after reactivation: {ok2b}")
        print(f"-> {'PASS' if ok2b else 'FAIL'}: watcher picks up reactivation dynamically.")
        all_pass &= ok2b
        print()

        # === 3: soft delete — hidden from management, in trash, preserved, reversible ===
        print("=== 3: soft_delete() moves to trash, hides from list_all(), preserves everything ===")
        registry.soft_delete(name)
        component = registry.get(name)
        ok3a = component.lifecycle_status == "deleted" and component.deleted_at is not None
        ok3b = name not in {c.name for c in registry.list_all()}  # hidden from default listing
        ok3c = name in {c.name for c in registry.list_deleted()}  # visible in trash
        ok3d = paths.root.exists() and indexer.count_component_chunks(name) > 0 and len(store.list_all(component_name=name, limit=None)) > 0
        ok3 = ok3a and ok3b and ok3c and ok3d
        print(f"  lifecycle_status={component.lifecycle_status} deleted_at={component.deleted_at}")
        print(f"  hidden from list_all: {ok3b}, visible in list_deleted: {ok3c}, data intact: {ok3d}")
        print(f"-> {'PASS' if ok3 else 'FAIL'}")
        all_pass &= ok3
        print()

        # === 4: restore — fully undone ===
        print("=== 4: restore() fully undoes soft_delete() ===")
        registry.restore(name)
        component = registry.get(name)
        ok4 = component.lifecycle_status == "active" and component.deleted_at is None and name in {c.name for c in registry.list_all()}
        print(f"  lifecycle_status={component.lifecycle_status} deleted_at={component.deleted_at}")
        print(f"-> {'PASS' if ok4 else 'FAIL'}")
        all_pass &= ok4
        print()

        # === 5: impact summary shows real numbers, including verified corrections ===
        print("=== 5: summarize_deletion_impact() reflects real data, verified corrections highlighted ===")
        registry.soft_delete(name)  # required precondition for the summary's normal use context (trash view)
        impact = component_deletion.summarize_deletion_impact(name, registry=registry)
        print(f"  {impact}")
        ok5 = (
            impact.inspection_count == 1
            and impact.verified_correction_count == 1
            and impact.knowledge_document_count == 1
            and impact.chromadb_chunk_count > 0
            and impact.machine_reading_count == 1
            and impact.training_file_count == 2
            and impact.training_run_count == 1
        )
        print(f"-> {'PASS' if ok5 else 'FAIL'}")
        all_pass &= ok5
        print()

        # === 6: precondition enforcement — can't permanently delete a non-deleted component ===
        print("=== 6: permanently_delete_component() refuses a non-'deleted' component ===")
        registry.restore(name)
        ok6 = False
        try:
            component_deletion.permanently_delete_component(name, registry=registry)
        except ValueError as exc:
            ok6 = True
            print(f"  raised ValueError as expected: {exc}")
        print(f"-> {'PASS' if ok6 else 'FAIL'}")
        all_pass &= ok6
        print()

        # === 7: resumability — partial completion (chromadb already cleaned) resumes cleanly ===
        print("=== 7: a partially-completed deletion resumes cleanly on re-run ===")
        registry.soft_delete(name)
        pre_chunks = indexer.count_component_chunks(name)
        indexer.delete_component_chunks(name)  # simulate step 2 already having completed in a prior, interrupted run
        assert indexer.count_component_chunks(name) == 0
        result = component_deletion.permanently_delete_component(name, registry=registry)
        print(f"  result: {result}")
        ok7 = (
            result.chromadb_chunks_removed == 0  # nothing left to remove there — already done
            and result.filesystem_removed
            and result.inspections_removed == 1
            and result.machine_readings_removed == 1
            and result.training_runs_removed == 1
            and result.registry_row_removed
            and not result.errors
        )
        print(f"-> {'PASS' if ok7 else 'FAIL'}: pre-cleaned ChromaDB step didn't break the rest of the run.")
        all_pass &= ok7
        print()

        # === 8: re-running permanently_delete_component() again is a clean no-op ===
        print("=== 8: calling permanently_delete_component() again (already fully gone) is a safe no-op ===")
        result2 = component_deletion.permanently_delete_component(name, registry=registry)
        ok8 = result2.already_complete and not result2.errors
        print(f"  result: {result2}")
        print(f"-> {'PASS' if ok8 else 'FAIL'}")
        all_pass &= ok8
        print()

        # === 9: complete cleanup — filesystem, chromadb, inspections, machine_readings, registry ===
        print("=== 9: everything is actually gone ===")
        ok9 = (
            not paths.root.exists()
            and indexer.count_component_chunks(name) == 0
            and len(store.list_all(component_name=name, limit=None)) == 0
            and SqliteMachineContextSource().count_readings(name) == 0
            and len(training_runs_store.list_for_component(name)) == 0
            and registry.get(name) is None
        )
        print(f"  filesystem gone: {not paths.root.exists()}")
        print(f"  chromadb chunks: {indexer.count_component_chunks(name)}")
        print(f"  inspections: {len(store.list_all(component_name=name, limit=None))}")
        print(f"  machine readings: {SqliteMachineContextSource().count_readings(name)}")
        print(f"  training runs: {len(training_runs_store.list_for_component(name))}")
        print(f"  registry row: {registry.get(name)}")
        print(f"-> {'PASS' if ok9 else 'FAIL'}")
        all_pass &= ok9
        print()

        # === 10: a new component with the SAME slug inherits NOTHING ===
        print("=== 10: a new component with the same display name/slug is completely clean ===")
        new_name = _setup_component_fresh_check(registry)
        all_pass &= new_name is not None
        print()

        print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED — see above'}")
    finally:
        # Best-effort final cleanup regardless of pass/fail above.
        leftover = registry.get(name)
        if leftover is not None:
            if leftover.lifecycle_status != "deleted":
                registry.soft_delete(name)
            component_deletion.permanently_delete_component(name, registry=registry)


def _setup_component_fresh_check(registry: ComponentRegistry) -> str | None:
    """Create a brand-new component with the exact same display name (and
    therefore the same slug) as the one just permanently deleted, and
    confirm it starts completely empty — no leftover chunks, inspections,
    or files inherited from its predecessor."""
    component = onboard.create_component(COMPONENT_DISPLAY_NAME, model_type="classifier", registry=registry)
    name = component.name
    paths = for_component(name)
    try:
        chunks = indexer.count_component_chunks(name)
        inspections = len(store.list_all(component_name=name, limit=None))
        readings = SqliteMachineContextSource().count_readings(name)
        training_runs = len(training_runs_store.list_for_component(name))
        knowledge_files = list(paths.knowledge_dir.iterdir()) if paths.knowledge_dir.exists() else []
        training_files = list(paths.training_approved_dir.iterdir()) if paths.training_approved_dir.exists() else []
        print(f"  new component slug: {name}")
        print(f"  inherited chunks={chunks} inspections={inspections} readings={readings} "
              f"training_runs={training_runs} knowledge_files={len(knowledge_files)} training_files={len(training_files)}")
        ok = (
            chunks == 0 and inspections == 0 and readings == 0 and training_runs == 0
            and not knowledge_files and not training_files
        )
        print(f"-> {'PASS' if ok else 'FAIL'}: new same-slug component inherits nothing from the deleted one.")
        return name if ok else None
    finally:
        registry.soft_delete(name)
        component_deletion.permanently_delete_component(name, registry=registry)


if __name__ == "__main__":
    main()
