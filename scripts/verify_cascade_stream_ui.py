"""Drives app/pages/5_cascade_stream.py end-to-end via Streamlit's AppTest —
component picker, Kafka/sampling settings save, video-file upload +
processing, and the live results feed — exactly the widget interactions a
real operator would make, no direct backend calls for the parts under test
(setup/teardown of the test component itself uses the backend directly,
same convention scripts/verify_onboard_cascade_ui.py already established).

Run with: python scripts/verify_cascade_stream_ui.py
"""

from __future__ import annotations

import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage import data
from streamlit.testing.v1 import AppTest

from emil_ml.config.logging_config import configure_logging
from emil_ml.config.registry import ComponentRegistry
from emil_ml.core.cascade import stream_store
from emil_ml.training import onboard

COMPONENT_DISPLAY_NAME = "Cascade Stream UI Test Component"
COMPONENT_NAME = "cascade-stream-ui-test-component"
VIDEO_PATH = Path("scratch_cascade_stream_ui_test_video.mp4")

ALL_PASS = True


def _check(label: str, condition: bool, detail: str = "") -> None:
    global ALL_PASS
    status = "PASS" if condition else "FAIL"
    print(f"  {status}: {label}" + (f" — {detail}" if detail else ""))
    ALL_PASS = ALL_PASS and condition


def _build_test_video() -> bytes:
    astronaut_bgr = cv2.cvtColor(np.asarray(Image.fromarray(data.astronaut()).convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = astronaut_bgr.shape[:2]
    writer = cv2.VideoWriter(str(VIDEO_PATH), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    for _ in range(15):
        writer.write(astronaut_bgr)
    writer.release()
    data_bytes = VIDEO_PATH.read_bytes()
    VIDEO_PATH.unlink()
    return data_bytes


def main() -> None:
    configure_logging()
    registry = ComponentRegistry()

    existing = registry.get(COMPONENT_NAME)
    if existing is not None:
        registry.delete(existing.name)
    stream_store.delete_all_for_component(COMPONENT_NAME)

    try:
        print("=== setup: real ready coco_detector component ===")
        component = onboard.create_component(COMPONENT_DISPLAY_NAME, model_type="coco_detector", registry=registry)
        onboard.train_component(component.name, registry=registry)
        print(f"  component={component.name}")
        print()

        print("=== 1: page loads with no exception, component picker shows it ===")
        at = AppTest.from_file("app/pages/5_cascade_stream.py")
        at.run(timeout=180)
        _check("no exception on load", not at.exception, detail=str(at.exception))
        select = at.selectbox(key="cascade_stream_component_select")
        _check("component appears in the picker", COMPONENT_DISPLAY_NAME in select.options, detail=str(select.options))
        select.select(COMPONENT_DISPLAY_NAME)
        at.run(timeout=60)
        _check("still no exception after selecting the component", not at.exception, detail=str(at.exception))
        print()

        print("=== 2: save Kafka + sampling settings ===")
        at.text_input(key=f"cascade_stream_kafka_bootstrap_{component.name}").set_value("localhost:9092")
        at.text_input(key=f"cascade_stream_kafka_topic_{component.name}").set_value("test-topic")
        at.number_input(key=f"cascade_stream_sample_rate_{component.name}").set_value(0.5)
        at.button(key=f"save_cascade_stream_settings_{component.name}").click()
        at.run(timeout=60)
        saved_component = registry.get(component.name)
        _check(
            "Kafka bootstrap servers saved",
            saved_component.cascade_stream_kafka_bootstrap_servers == "localhost:9092",
            detail=saved_component.cascade_stream_kafka_bootstrap_servers,
        )
        _check("Kafka topic saved", saved_component.cascade_stream_kafka_topic == "test-topic")
        _check("sample rate saved", saved_component.cascade_stream_sample_rate_seconds == 0.5)
        page_text = " ".join(m.value for m in at.markdown) + " " + " ".join(c.value for c in at.code)
        _check(
            "terminal-command block shows the right invocation",
            f"python -m emil_ml.cascade_stream --component {component.name}" in page_text,
        )
        print()

        print("=== 3: upload and process a video end-to-end through real widget interaction ===")
        video_bytes = _build_test_video()
        at.file_uploader(key=f"cascade_stream_video_upload_{component.name}").set_value(
            ("test_video.mp4", video_bytes, "video/mp4")
        )
        at.run(timeout=60)
        at.button(key=f"cascade_stream_process_video_{component.name}").click()
        at.run(timeout=120)
        _check("no exception while processing the video", not at.exception, detail=str(at.exception))
        success_text = " ".join(s.value for s in at.success)
        _check("success message shows frames processed", "frame(s) processed" in success_text, detail=success_text)
        print()

        print("=== 4: results feed shows real cascade output afterward ===")
        results = stream_store.list_recent_results(component.name, limit=10)
        _check("at least one result row was persisted", len(results) > 0, detail=str(len(results)))
        at.run(timeout=60)
        feed_text = " ".join(c.value for c in at.caption) + " ".join(m.value for m in at.markdown)
        _check("results feed renders at least one object label", "human" in feed_text or "category:" in feed_text)
        print()

        print("=== 5: thumbnails have boxes drawn (detections carry real boxes, files are valid images) ===")
        any_boxed = any(any(obj.get("box") is not None for obj in r.objects) for r in results)
        _check("at least one detection has a box (drawable)", any_boxed)
        for r in results:
            if r.thumbnail_path:
                with Image.open(r.thumbnail_path) as im:
                    im.verify()
        _check("every thumbnail file is a valid, openable image", True)
        print()

        print("=== 6: upload and process a still image end-to-end through real widget interaction ===")
        image_bytes = io.BytesIO()
        Image.fromarray(data.astronaut()).convert("RGB").save(image_bytes, format="PNG")
        at.file_uploader(key=f"cascade_stream_image_upload_{component.name}").set_value(
            ("test_image.png", image_bytes.getvalue(), "image/png")
        )
        at.run(timeout=60)
        at.button(key=f"cascade_stream_process_image_{component.name}").click()
        at.run(timeout=60)
        _check("no exception while processing the still image", not at.exception, detail=str(at.exception))
        success_text = " ".join(s.value for s in at.success)
        _check("success message shows object(s) detected", "object(s) detected" in success_text, detail=success_text)
        image_results = [r for r in stream_store.list_recent_results(component.name, limit=20) if r.source == "image"]
        _check("an 'image'-sourced result row was persisted", len(image_results) == 1, detail=str(len(image_results)))
        print()

        print("=== 7: 'Clear results & images' removes every result row + thumbnail file, keeps run history ===")
        results_before = stream_store.list_recent_results(component.name, limit=50)
        thumbnail_paths = [Path(r.thumbnail_path) for r in results_before if r.thumbnail_path]
        _check("there are results + thumbnails to clean up", len(thumbnail_paths) > 0, detail=str(len(thumbnail_paths)))
        runs_before = stream_store.list_runs(component.name, limit=50)

        at.button(key=f"cascade_stream_clean_{component.name}").click()
        at.run(timeout=60)
        _check("no exception while cleaning up", not at.exception, detail=str(at.exception))
        _check("no results remain", stream_store.list_recent_results(component.name) == [])
        _check(
            "every thumbnail file was actually deleted from disk",
            all(not p.exists() for p in thumbnail_paths),
            detail=str([str(p) for p in thumbnail_paths if p.exists()]),
        )
        runs_after = stream_store.list_runs(component.name, limit=50)
        _check(
            "run history is untouched by the cleanup (only images/results are cleared)",
            len(runs_after) == len(runs_before),
            detail=f"before={len(runs_before)} after={len(runs_after)}",
        )
        print()

        print(f"Overall: {'ALL PASS' if ALL_PASS else 'SOME FAILED — see above'}")
    finally:
        component_after = registry.get(COMPONENT_NAME)
        if component_after is not None:
            registry.delete(component_after.name)
        stream_store.delete_all_for_component(COMPONENT_NAME)
        VIDEO_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
