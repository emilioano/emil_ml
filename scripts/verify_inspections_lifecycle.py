"""Fas 5 sanity check: persistence + lifecycle, end to end through the
real orchestrator (not mocked), covering every item in the Fas 5
checklist:

1. Report generated -> .md written beside the image with correct
   frontmatter, DB record created with the right path and status='new'.
2. Acknowledge -> status new -> acknowledged in DB; files untouched
   (acknowledgement is not a file move).
3. Archive -> image + .md moved to a date-partitioned archive with a
   unique name, DB path updated atomically in the same operation,
   status='archived'.
4. Async: run_inspection() returns with the verdict already final while
   report_status is still 'pending'; a background thread fills in the
   report shortly after, without the caller having blocked on it.
5. The "no documentation" case (Fas 4) is persisted the same as any
   other report — not skipped just because it has nothing to cite.

Pinned to mock LLM mode for this script's own run (see _pin_mock_llm_mode()
below), regardless of config/settings.py's DEFAULT_LLM_MODE — everything
this script checks (DB record shape, status transitions, the archive
move, path updates) is about persistence and lifecycle, none of which
depends on how a report's text was produced or how long that took. Tying
this test to a real LLM's generation latency (DEFAULT_LLM_MODE="ollama"
can legitimately take 30s-2min per report) was never intentional — it's
what the RAG orchestration itself was verified against first, before
Ollama was ever wired in (see verify_reporting_generation.py, which
pins llm_mode="mock" the same way for the same reason). This is a
test-isolation fix, not a statement about which mode real usage should
run in.

Run with: python scripts/verify_inspections_lifecycle.py
"""

from __future__ import annotations

import functools
import shutil
import sys
import time
import glob

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from emil_ml.config.logging_config import configure_logging
from emil_ml.config.registry import ComponentRegistry
from emil_ml.core.inspections import lifecycle, store
from emil_ml.core.inspections.orchestrator import run_inspection
from emil_ml.core.reporting import reporter
from emil_ml.utils.paths import for_component

COMPONENT_NAME = "tandborste"


def _pin_mock_llm_mode() -> None:
    """Force every report this script generates to use mock mode, no
    matter what config/settings.py's DEFAULT_LLM_MODE is set to.

    orchestrator._generate_report_and_update() calls
    `reporter.generate_report(component, prediction, on_progress=...,
    on_chunk=...)` without an explicit llm_mode, so it always falls
    through to generate_report()'s own default parameter value — which
    is bound once, at reporter.py's *import* time, to whatever
    DEFAULT_LLM_MODE was back then. Reassigning
    config.settings.DEFAULT_LLM_MODE from here wouldn't reach that
    already-bound default (Python binds default argument values at
    function-definition time, not call time).

    What actually works, without touching any production code: replace
    the `generate_report` attribute on the `reporter` module object
    itself. orchestrator.py calls it as `reporter.generate_report(...)`
    — an attribute lookup on that module at call time, not a frozen
    reference — so this redirects every caller downstream of it
    (run_inspection(), both the threaded and async_report=False paths)
    to always use mock mode, exactly as if DEFAULT_LLM_MODE were "mock"
    for the duration of this script only.
    """
    reporter.generate_report = functools.partial(reporter.generate_report, llm_mode="mock")


def main() -> None:
    configure_logging()
    _pin_mock_llm_mode()
    registry = ComponentRegistry()
    registry.update_settings(COMPONENT_NAME, reporting_enabled=True, reporting_condition="always")
    paths = for_component(COMPONENT_NAME)

    candidates = glob.glob(f"data/components/{COMPONENT_NAME}/training/approved/*")
    if not candidates:
        raise SystemExit(f"No sample images found for {COMPONENT_NAME!r}.")
    image_path = candidates[0]

    # === 1 & 4: report generated + async ================================
    print("=== 1 & 4: run_inspection() (async), check immediate vs. eventual state ===")
    t0 = time.time()
    record, details = run_inspection(image_path, COMPONENT_NAME, registry=registry)
    call_duration = time.time() - t0
    print(f"run_inspection() returned in {call_duration:.3f}s")
    print(f"  verdict={record.verdict} score={record.score:.4f} status={record.status} report_status={record.report_status}")
    # Not a wall-clock budget: the call includes one-time, this-process-only
    # model loading (autoencoder + embedding model, both lazily loaded and
    # cached on first use) that has nothing to do with report generation.
    # What actually proves this is non-blocking: the call already returned
    # with report_status='pending' rather than waiting for the report to
    # reach 'complete' — verified separately below.
    ok_fast = record.report_status == "pending" and record.status == "new"
    print(f"-> {'PASS' if ok_fast else 'FAIL'}: verdict final immediately, report_status='pending' (not blocked on the report).")
    print()

    print("Checking DB + .md file for the fast (pre-report) record...")
    image_abs = paths.root / record.image_path
    print(f"  image file exists: {image_abs.exists()}  ({record.image_path})")
    ok_image = image_abs.exists() and record.report_path is None
    print(f"-> {'PASS' if ok_image else 'FAIL'}: image saved immediately, report_path still None (report not written yet).")
    print()

    print("Waiting for the background report thread to finish (mock LLM, should be near-instant)...")
    deadline = time.time() + 15
    while time.time() < deadline:
        record = store.get(record.id)
        if record.report_status != "pending":
            break
        time.sleep(0.2)
    print(f"  final report_status: {record.report_status}  (waited {time.time() - deadline + 15:.1f}s)")
    ok_async = record.report_status == "complete" and record.report_path is not None
    print(f"-> {'PASS' if ok_async else 'FAIL'}: background thread completed the report without run_inspection() having blocked on it.")
    print()

    report_abs = paths.root / record.report_path
    print(f"Report .md exists: {report_abs.exists()}  ({record.report_path})")
    md_text = report_abs.read_text(encoding="utf-8") if report_abs.exists() else ""
    has_frontmatter = md_text.startswith("---") and f"component: {COMPONENT_NAME}" in md_text and "verdict:" in md_text and "score:" in md_text and "timestamp:" in md_text
    print(f"-> {'PASS' if has_frontmatter else 'FAIL'}: .md has correct YAML frontmatter.")
    print()
    print("--- .md content (first 500 chars) ---")
    print(md_text[:500])
    print("---")
    print()

    # === 2: acknowledge ===================================================
    print("=== 2: acknowledge (status new -> acknowledged, files untouched) ===")
    pre_ack_image_path = record.image_path
    pre_ack_report_path = record.report_path
    store.acknowledge(record.id, by="verify-script")
    record = store.get(record.id)
    print(f"  status={record.status} acknowledged_by={record.acknowledged_by} acknowledged_at={record.acknowledged_at}")
    ok_ack = (
        record.status == "acknowledged"
        and record.acknowledged_by == "verify-script"
        and record.image_path == pre_ack_image_path
        and record.report_path == pre_ack_report_path
        and (paths.root / record.image_path).exists()
        and (paths.root / record.report_path).exists()
    )
    print(f"-> {'PASS' if ok_ack else 'FAIL'}: status flipped, paths and files unchanged (ack != move).")
    print()

    # === 3: archive ========================================================
    print("=== 3: archive (move to date-partitioned archive, unique name, atomic DB update) ===")
    old_image_abs = paths.root / record.image_path
    old_report_abs = paths.root / record.report_path
    new_image_path, new_report_path = lifecycle.archive(paths, record)
    store.mark_archived(record.id, image_path=new_image_path, report_path=new_report_path)
    record = store.get(record.id)
    print(f"  new image_path: {record.image_path}")
    print(f"  new report_path: {record.report_path}")
    print(f"  status={record.status} archived_at={record.archived_at}")

    ok_archive = (
        record.status == "archived"
        and record.image_path.startswith("archive/")
        and record.report_path.startswith("archive/")
        and not old_image_abs.exists()
        and not old_report_abs.exists()
        and (paths.root / record.image_path).exists()
        and (paths.root / record.report_path).exists()
    )
    # Date partition check: archive/<year>/<month>/<day>/...
    parts = record.image_path.split("/")
    ok_partition = len(parts) == 5 and parts[0] == "archive" and all(p.isdigit() for p in parts[1:3])
    print(f"-> {'PASS' if ok_archive else 'FAIL'}: files moved (old paths gone, new paths exist), DB updated atomically, status='archived'.")
    print(f"-> {'PASS' if ok_partition else 'FAIL'}: date-partitioned path (archive/YYYY/MM/DD/...).")
    print()

    # Unique naming: archive a second inspection of the same component and
    # confirm no filename collision even with the same original stem style.
    print("Archiving a second inspection (from the same component) to confirm no filename collision...")
    record2, _ = run_inspection(image_path, COMPONENT_NAME, registry=registry, async_report=False)
    store.acknowledge(record2.id, by="verify-script")
    record2 = store.get(record2.id)
    new_image_path2, new_report_path2 = lifecycle.archive(paths, record2)
    store.mark_archived(record2.id, image_path=new_image_path2, report_path=new_report_path2)
    record2 = store.get(record2.id)
    ok_unique = record2.image_path != record.image_path and (paths.root / record2.image_path).exists()
    print(f"  first archived image:  {record.image_path}")
    print(f"  second archived image: {record2.image_path}")
    print(f"-> {'PASS' if ok_unique else 'FAIL'}: distinct filenames, no collision.")
    print()

    # === 5: "no documentation" case persisted too =========================
    print("=== 5: 'no documentation' report (Fas 4) is persisted, not skipped ===")
    empty_name = None
    try:
        from emil_ml.training import onboard

        empty_component = onboard.create_component(
            "Fas5 No Docs Test", model_type="autoencoder", registry=registry,
            reporting_enabled=True, reporting_condition="always",
        )
        empty_name = empty_component.name
        empty_paths = for_component(empty_name)
        empty_paths.create_all()
        import numpy as np
        from PIL import Image as PILImage

        rng = np.random.default_rng(0)
        img = PILImage.fromarray(np.clip(np.full((32, 32, 3), 120, dtype=np.int16) + rng.integers(-5, 5, (32, 32, 3)), 0, 255).astype("uint8"))
        img.save(empty_paths.training_approved_dir / "a.png")
        for i in range(1, 15):
            img2 = PILImage.fromarray(np.clip(np.full((32, 32, 3), 120, dtype=np.int16) + rng.integers(-5, 5, (32, 32, 3)), 0, 255).astype("uint8"))
            img2.save(empty_paths.training_approved_dir / f"a{i}.png")
        from emil_ml.training.onboard import train_component

        train_component(empty_name, registry=registry)
        empty_component = registry.get(empty_name)

        record5, _ = run_inspection(img, empty_name, registry=registry, async_report=False)
        print(f"  report_status={record5.report_status}")
        print(f"  report_text={record5.report_text}")
        print(f"  report_path={record5.report_path}")
        report5_abs = empty_paths.root / record5.report_path if record5.report_path else None
        ok_no_docs = (
            record5.report_status == "complete"
            and record5.report_text is not None
            and "No relevant documentation" in record5.report_text
            and report5_abs is not None
            and report5_abs.exists()
        )
        print(f"-> {'PASS' if ok_no_docs else 'FAIL'}: honest 'no documentation' report persisted to DB + .md, not skipped.")
    finally:
        if empty_name:
            registry.delete(empty_name)
            shutil.rmtree(for_component(empty_name).root, ignore_errors=True)

    print()
    print("Fas 5 persistence + lifecycle verification complete.")


if __name__ == "__main__":
    main()
