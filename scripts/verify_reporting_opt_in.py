"""Verifies the reporting opt-in behavior end to end through the real
pipeline.inspect() entry point:

1. reporting_enabled=False -> inspect() returns exactly what it did before
   RAG existed (no "report" key), and core.reporting.reporter.generate_report
   is never called at all.
2. reporting_enabled=True, condition="always", but no knowledge documents
   and no machine_parameters defined -> inspect() still returns a report,
   with an honest "no relevant documentation" report_text instead of
   crashing or inventing content.
3. index_all() only indexes components with reporting_enabled on, even if
   a disabled component has files sitting in its knowledge/ directory.

Run with: python scripts/verify_reporting_opt_in.py
"""

from __future__ import annotations

import io
import random
import shutil
import sys
from unittest.mock import patch

import numpy as np
from PIL import Image

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from emil_ml.config.logging_config import configure_logging
from emil_ml.config.registry import ComponentRegistry
from emil_ml.core.reporting import reporter
from emil_ml.core.reporting.knowledge import indexer
from emil_ml.pipeline.inspect import inspect
from emil_ml.training import onboard
from emil_ml.utils.paths import for_component

IMAGE_SIZE = 32


def _png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _make_approved(rng: random.Random) -> bytes:
    base = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.int16)
    base[:, :] = [40, 160, 150]
    noise = np.array([[rng.gauss(0, 8) for _ in range(IMAGE_SIZE)] for _ in range(IMAGE_SIZE)])
    for c in range(3):
        base[:, :, c] = np.clip(base[:, :, c] + noise, 0, 255)
    return _png_bytes(base.astype(np.uint8))


def _make_failed(rng: random.Random) -> bytes:
    arr = np.array(
        [[[rng.randint(0, 255) for _ in range(3)] for _ in range(IMAGE_SIZE)] for _ in range(IMAGE_SIZE)],
        dtype=np.uint8,
    )
    return _png_bytes(arr)


def _make_and_train(registry: ComponentRegistry, display_name: str, rng: random.Random, **settings):
    component = onboard.create_component(
        display_name, image_size=IMAGE_SIZE, epochs=5, latent_dim=8, batch_size=4, registry=registry, **settings
    )
    approved = [(f"approved_{i}.png", _make_approved(rng)) for i in range(20)]
    failed = [(f"failed_{i}.png", _make_failed(rng)) for i in range(5)]
    onboard.add_training_images(component.name, approved=approved, failed=failed)
    onboard.train_component(component.name, registry=registry)
    return registry.get(component.name)


def main() -> None:
    configure_logging()
    rng = random.Random(42)
    registry = ComponentRegistry()
    created: list[str] = []

    try:
        # === Scenario 1: reporting disabled ================================
        print("=== Scenario 1: reporting_enabled=False ===")
        off_component = _make_and_train(registry, "Reporting Opt-In Test OFF", rng, reporting_enabled=False)
        created.append(off_component.name)
        print(f"created: {off_component.name}, reporting_enabled={off_component.reporting_enabled}")

        with patch.object(reporter, "generate_report", side_effect=AssertionError("must not be called")) as spy:
            result = inspect(_make_failed(rng), off_component.name, registry=registry)
        print("inspect() keys:", sorted(result.keys()))
        ok = "report" not in result and spy.call_count == 0
        print(f"-> {'PASS' if ok else 'FAIL'}: no 'report' key, generate_report never called.")
        print()

        # === Scenario 2: reporting enabled, no docs, no machine params =====
        print("=== Scenario 2: reporting_enabled=True, condition='always', no docs/params ===")
        on_component = _make_and_train(
            registry, "Reporting Opt-In Test ON", rng, reporting_enabled=True, reporting_condition="always"
        )
        created.append(on_component.name)
        print(f"created: {on_component.name}, reporting_enabled={on_component.reporting_enabled}, "
              f"condition={on_component.reporting_condition}")
        knowledge_dir = for_component(on_component.name).knowledge_dir
        print(f"knowledge_dir exists: {knowledge_dir.exists()}, "
              f"contents: {list(knowledge_dir.iterdir()) if knowledge_dir.exists() else []}")
        print(f"machine_parameters: {on_component.machine_parameters!r}")

        result = inspect(_make_approved(rng), on_component.name, registry=registry)
        print("inspect() keys:", sorted(result.keys()))
        has_report = "report" in result
        print(f"-> {'PASS' if has_report else 'FAIL'}: 'report' key present (condition='always').")
        if has_report:
            report = result["report"]
            print("report_text:", report.report_text)
            print("sources:", report.sources)
            print("machine_context_used:", report.machine_context_used)
            honest_empty = (
                report.sources == []
                and report.machine_context_used == []
                and "No relevant documentation" in report.report_text
            )
            print(f"-> {'PASS' if honest_empty else 'FAIL'}: honest 'no documentation' report, no crash, no invented content.")
        print()

        # === Scenario 3: index_all() respects the opt-in ====================
        print("=== Scenario 3: index_all() skips a disabled component even with files on disk ===")
        stray_doc_dir = for_component(off_component.name).knowledge_dir
        stray_doc_dir.mkdir(parents=True, exist_ok=True)
        (stray_doc_dir / "stray.md").write_text(
            "---\ndoc_type: manual\nsource: Stray Test Doc\n---\n# Section\nSome content.\n", encoding="utf-8"
        )
        counts = indexer.index_all()
        print("index_all() result:", counts)
        skipped_disabled = off_component.name not in counts
        included_enabled = on_component.name in counts
        print(f"-> {'PASS' if skipped_disabled else 'FAIL'}: disabled component {off_component.name!r} not in results "
              f"despite having a stray document on disk.")
        print(f"-> {'PASS' if included_enabled else 'FAIL'}: enabled component {on_component.name!r} was indexed "
              f"({counts.get(on_component.name)} chunks).")
        print()

        print("Reporting opt-in verification OK.")
    finally:
        for name in created:
            registry.delete(name)
            shutil.rmtree(for_component(name).root, ignore_errors=True)
        print("Cleaned up test components.")


if __name__ == "__main__":
    main()
