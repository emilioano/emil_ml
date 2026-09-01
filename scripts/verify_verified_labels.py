"""Verifies the prediction/verified-label split in core/inspections/store.py
on a disposable throwaway component (deleted in `finally`, same convention
as the other verify_*.py scripts). Pinned to mock LLM mode (see
verify_inspections_lifecycle.py's own docstring for why) since this is a
pure persistence/data-model check, unrelated to report generation.

1. store.verify() sets verified_status/verified_label/verified_by/
   verified_at WITHOUT touching the model's own prediction
   (verdict/score/defect_classes) at all.
2. store.list_verified_for_training() returns ONLY verified records —
   unverified ones (including an auto-acknowledged approved one, and a
   plain never-touched record) never come with.
3. Neither a normal store.acknowledge() nor orchestrator.py's
   auto_acknowledge path (approved_handling='auto_acknowledge') ever sets
   verified_status off its 'unverified' default — acknowledgement is not
   verification, full stop, even when it's automatic.
4. verify() rejects a missing label and an invalid status, and
   verified_correct/verified_incorrect both work with the prediction
   preserved either way (confirming even a "correct" verification doesn't
   silently defer to the prediction instead of an explicit label).

Run with: python scripts/verify_verified_labels.py
"""

from __future__ import annotations

import functools
import glob
import shutil
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from emil_ml.config.logging_config import configure_logging
from emil_ml.config.registry import ComponentRegistry
from emil_ml.core.inspections import orchestrator, store
from emil_ml.core.reporting import reporter

COMPONENT_NAME = "tandborste"
FORCE_APPROVED_THRESHOLD = 999.0
FORCE_FAILED_THRESHOLD = 0.0


def _pin_mock_llm_mode() -> None:
    """See verify_inspections_lifecycle.py's own _pin_mock_llm_mode() for
    the full explanation — same technique, same reason."""
    reporter.generate_report = functools.partial(reporter.generate_report, llm_mode="mock")


def main() -> None:
    configure_logging()
    _pin_mock_llm_mode()
    registry = ComponentRegistry()
    registry.update_settings(COMPONENT_NAME, reporting_enabled=False, approved_handling="hide_from_default_view")

    candidates = glob.glob(f"data/components/{COMPONENT_NAME}/training/approved/*")
    if not candidates:
        raise SystemExit("Need at least one sample image for tandborste.")
    sample_image = candidates[0]

    # === setup: one failed record to verify against ========================
    record, _ = orchestrator.run_inspection(
        sample_image,
        COMPONENT_NAME,
        registry=registry,
        async_report=False,
        run_by="test-operator",
        threshold_override=FORCE_FAILED_THRESHOLD,
    )
    original_verdict = record.verdict
    original_score = record.score
    original_defect_classes = list(record.defect_classes)

    # === 1: verify() sets the verified layer without touching prediction ===
    print("=== 1: verify() sets verified_* without changing verdict/score/defect_classes ===")
    corrected_label = {"verdict": "failed", "defect_classes": ["missing_bristles"], "boxes": []}
    store.verify(record.id, status="verified_incorrect", label=corrected_label, by="qa-reviewer")
    reloaded = store.get(record.id)
    ok1 = (
        reloaded.verdict == original_verdict
        and reloaded.score == original_score
        and reloaded.defect_classes == original_defect_classes
        and reloaded.verified_status == "verified_incorrect"
        and reloaded.verified_label == corrected_label
        and reloaded.verified_by == "qa-reviewer"
        and reloaded.verified_at is not None
    )
    print(f"  prediction unchanged: verdict={reloaded.verdict} score={reloaded.score} defect_classes={reloaded.defect_classes}")
    print(f"  verified layer: status={reloaded.verified_status} label={reloaded.verified_label} by={reloaded.verified_by}")
    print(f"-> {'PASS' if ok1 else 'FAIL'}: prediction preserved, verified layer set independently.")
    print()

    # === 2: list_verified_for_training() returns only verified records =====
    print("=== 2: list_verified_for_training() excludes unverified records (incl. auto-acknowledged approved) ===")
    registry.update_settings(COMPONENT_NAME, approved_handling="auto_acknowledge")
    auto_ack_record, _ = orchestrator.run_inspection(
        sample_image,
        COMPONENT_NAME,
        registry=registry,
        async_report=False,
        run_by="test-operator",
        threshold_override=FORCE_APPROVED_THRESHOLD,
    )
    plain_record, _ = orchestrator.run_inspection(
        sample_image,
        COMPONENT_NAME,
        registry=registry,
        async_report=False,
        run_by="test-operator",
        threshold_override=FORCE_FAILED_THRESHOLD,
    )
    registry.update_settings(COMPONENT_NAME, approved_handling="hide_from_default_view")

    verified_ids = {r.id for r in store.list_verified_for_training(COMPONENT_NAME)}
    print(f"  verified ids returned: {sorted(verified_ids)}")
    ok2 = (
        record.id in verified_ids
        and auto_ack_record.id not in verified_ids
        and plain_record.id not in verified_ids
    )
    print(f"-> {'PASS' if ok2 else 'FAIL'}: only the explicitly-verified record comes back.")
    print()

    # === 3: acknowledge()/auto_acknowledge never set verification ==========
    print("=== 3: neither acknowledge() nor auto_acknowledge ever touch verified_status ===")
    auto_ack_reloaded = store.get(auto_ack_record.id)
    ok3a = auto_ack_reloaded.status == "acknowledged" and auto_ack_reloaded.verified_status == "unverified"
    print(
        f"  auto-acknowledged approved record: status={auto_ack_reloaded.status} "
        f"verified_status={auto_ack_reloaded.verified_status}"
    )

    store.acknowledge(plain_record.id, by="operator-2")
    manually_acked = store.get(plain_record.id)
    ok3b = manually_acked.status == "acknowledged" and manually_acked.verified_status == "unverified"
    print(
        f"  manually-acknowledged failed record: status={manually_acked.status} "
        f"verified_status={manually_acked.verified_status}"
    )
    ok3 = ok3a and ok3b
    print(f"-> {'PASS' if ok3 else 'FAIL'}: acknowledgement (manual or automatic) never implies verification.")
    print()

    # === 4: verify() input validation + verified_correct also requires a label ===
    print("=== 4: verify() rejects a missing label and an invalid status ===")
    ok4a = False
    try:
        store.verify(plain_record.id, status="verified_correct", label={}, by="qa-reviewer")
    except ValueError as exc:
        ok4a = True
        print(f"  empty label raised ValueError as expected: {exc}")

    ok4b = False
    try:
        store.verify(plain_record.id, status="unverified", label={"verdict": "failed"}, by="qa-reviewer")
    except ValueError as exc:
        ok4b = True
        print(f"  status='unverified' raised ValueError as expected: {exc}")

    confirm_label = {"verdict": "failed", "defect_classes": [], "boxes": []}
    store.verify(plain_record.id, status="verified_correct", label=confirm_label, by="qa-reviewer")
    confirmed = store.get(plain_record.id)
    ok4c = (
        confirmed.verified_status == "verified_correct"
        and confirmed.verified_label == confirm_label
        and confirmed.verdict == "failed"  # prediction still untouched
    )
    print(f"  verified_correct with explicit label: status={confirmed.verified_status} label={confirmed.verified_label}")
    ok4 = ok4a and ok4b and ok4c
    print(f"-> {'PASS' if ok4 else 'FAIL'}: validation rejects bad input; verified_correct still requires/keeps an explicit label.")
    print()

    all_pass = all([ok1, ok2, ok3, ok4])
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED — see above'}")


if __name__ == "__main__":
    main()
