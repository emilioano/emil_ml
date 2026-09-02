"""Verifies the video-file half of the cascade live-stream feature
end-to-end, through real code, with no Kafka broker needed:
VideoFileFrameSource -> stream_processor.should_sample()/process_frame() ->
stream_store — the exact loop app/pages/5_cascade_stream.py's "Video file"
section runs, and the same should_sample()/process_frame() the standalone
Kafka consumer (emil_ml.cascade_stream.service) calls too.

Uses skimage's bundled astronaut() photo (a real, detectable "person") to
build a short synthetic test video via cv2.VideoWriter — same test-image
sourcing convention scripts/verify_cascade_full.py already established for
this cascade.

The Kafka side's construction-and-graceful-failure-with-no-broker behavior
is verified separately (see the session's own manual check of
KafkaFrameSource against an unreachable broker) — an actual live Kafka
produce->consume flow cannot be verified in this environment at all; see
this project's cascade-stream implementation plan for that explicit
limitation.

Run with: python scripts/verify_cascade_stream_video.py
"""

from __future__ import annotations

import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage import data

from emil_ml.config.logging_config import configure_logging
from emil_ml.config.registry import ComponentRegistry
from emil_ml.core.cascade import stream_processor, stream_store
from emil_ml.core.cascade.frame_sources import VideoFileFrameSource
from emil_ml.training import onboard

COMPONENT_DISPLAY_NAME = "Cascade Stream Video Test Component"
VIDEO_PATH = Path("scratch_cascade_stream_test_video.mp4")
FPS = 10.0
FRAME_COUNT = 30  # ~2.9s of video at 10fps (positions 0.0 .. 2.9)
SAMPLE_RATE_SECONDS = 1.0  # expect samples at position 0.0, 1.0, 2.0 -> 3 processed frames

ALL_PASS = True


def _check(label: str, condition: bool, detail: str = "") -> None:
    global ALL_PASS
    status = "PASS" if condition else "FAIL"
    print(f"  {status}: {label}" + (f" — {detail}" if detail else ""))
    ALL_PASS = ALL_PASS and condition


def _build_test_video() -> None:
    astronaut_bgr = cv2.cvtColor(np.asarray(Image.fromarray(data.astronaut()).convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = astronaut_bgr.shape[:2]
    writer = cv2.VideoWriter(str(VIDEO_PATH), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for _ in range(FRAME_COUNT):
        writer.write(astronaut_bgr)
    writer.release()


def main() -> None:
    configure_logging()
    registry = ComponentRegistry()

    existing = registry.get(COMPONENT_DISPLAY_NAME.lower().replace(" ", "-"))
    if existing is not None:
        registry.delete(existing.name)
    stream_store.delete_all_for_component(COMPONENT_DISPLAY_NAME.lower().replace(" ", "-"))

    try:
        print("=== 1: create + train a real coco_detector component ===")
        component = onboard.create_component(COMPONENT_DISPLAY_NAME, model_type="coco_detector", registry=registry)
        onboard.train_component(component.name, registry=registry)
        component = registry.get(component.name)
        _check("component is ready", component.status == "ready", detail=component.status)
        print()

        print("=== 2: build a short synthetic test video (astronaut.jpg repeated) ===")
        _build_test_video()
        _check("video file was written", VIDEO_PATH.exists())
        print()

        print("=== 3: run it through VideoFileFrameSource -> should_sample() -> process_frame() -> stream_store ===")
        source = VideoFileFrameSource(VIDEO_PATH)
        reported_count = source.frame_count()
        print(f"  frame_count() reports: {reported_count}")
        _check("frame_count() matches the video actually written", reported_count == FRAME_COUNT, detail=str(reported_count))

        run = stream_store.start_run(component.name, source="video", source_detail=VIDEO_PATH.name)
        last_processed_position: float | None = None
        frames_seen = 0
        frames_processed = 0
        for frame in source.frames():
            frames_seen += 1
            if stream_processor.should_sample(last_processed_position, SAMPLE_RATE_SECONDS, frame.position_seconds):
                stream_processor.process_frame(frame, component, run_id=run.id, source="video", registry=registry)
                last_processed_position = frame.position_seconds
                frames_processed += 1
        stream_store.heartbeat(run.id, frames_seen=frames_seen, frames_processed=frames_processed)
        stream_store.finish_run(run.id, status="completed")
        source.close()

        print(f"  frames_seen={frames_seen} frames_processed={frames_processed}")
        _check("every frame was seen", frames_seen == FRAME_COUNT, detail=str(frames_seen))
        _check(
            "throttling actually reduced processed count vs seen count (real sampling, not every frame)",
            frames_processed == 3,
            detail=f"expected 3 (positions 0.0/1.0/2.0), got {frames_processed}",
        )
        print()

        print("=== 4: the run row reflects the same counts, marked completed ===")
        finished_run = stream_store.get_run(run.id)
        _check("run status is 'completed'", finished_run.status == "completed", detail=finished_run.status)
        _check("run.frames_seen matches", finished_run.frames_seen == frames_seen)
        _check("run.frames_processed matches", finished_run.frames_processed == frames_processed)
        _check("run.finished_at is set", finished_run.finished_at is not None)
        print()

        print("=== 5: cascade_stream_results rows exist, one per processed frame, with real detections ===")
        results = stream_store.list_recent_results(component.name, limit=10)
        _check("one result row per processed frame", len(results) == frames_processed, detail=str(len(results)))
        any_detected_person = False
        for r in results:
            _check(f"result {r.id}: source='video'", r.source == "video")
            _check(f"result {r.id}: frame_ref is set", bool(r.frame_ref), detail=r.frame_ref)
            _check(f"result {r.id}: thumbnail was saved", bool(r.thumbnail_path) and Path(r.thumbnail_path).exists())
            if any(obj.get("category") == "human" for obj in r.objects):
                any_detected_person = True
        _check(
            "at least one processed frame detected a real 'human' (astronaut.jpg is a real detectable person)",
            any_detected_person,
        )
        print()

        print("=== 6: count_results_for_component() matches, and delete_all_for_component() cleans up everything ===")
        count_before = stream_store.count_results_for_component(component.name)
        _check("count_results_for_component matches list length", count_before == len(results), detail=str(count_before))
        removed = stream_store.delete_all_for_component(component.name)
        _check(
            "delete_all_for_component removed every run + result row",
            removed == len(results) + 1,
            detail=f"removed={removed}, expected={len(results) + 1}",
        )
        _check("no active run remains", stream_store.get_active_run(component.name) is None)
        _check("no results remain", stream_store.list_recent_results(component.name) == [])
        print()

        print(f"Overall: {'ALL PASS' if ALL_PASS else 'SOME FAILED — see above'}")
    finally:
        component_after = registry.get(COMPONENT_DISPLAY_NAME.lower().replace(" ", "-"))
        if component_after is not None:
            registry.delete(component_after.name)
        stream_store.delete_all_for_component(COMPONENT_DISPLAY_NAME.lower().replace(" ", "-"))
        VIDEO_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
