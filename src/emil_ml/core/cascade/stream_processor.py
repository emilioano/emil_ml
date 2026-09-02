"""Shared per-frame logic for continuous cascade operation — the one thing
both the standalone Kafka consumer (emil_ml.cascade_stream.service) and the
Cascade Stream page's video-file section call, identically, so cascade
dispatch is never duplicated between them. Direct analog to
core/inspections/orchestrator.py's role: called the same way by the folder
watcher and the Streamlit UI.

Deliberately split into two pieces with one job each:
- `should_sample()`: pure decision, no I/O — "is this frame worth running?"
- `process_frame()`: only ever called once the caller has already decided
  yes — runs the cascade, saves a thumbnail, persists the result.

The caller's own loop owns: iterating the frame source, calling
should_sample() per frame, incrementing frames_seen (every frame) vs
frames_processed (only ones actually run), and calling
core/cascade/stream_store.py's heartbeat() periodically. Keeping that outside
process_frame() mirrors the clean split emil_ml/watcher/service.py already
draws between _wait_until_stable() and _process_file().
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from emil_ml.config.registry import Component, ComponentRegistry
from emil_ml.core.cascade import pipeline as cascade_pipeline
from emil_ml.core.cascade import stream_store
from emil_ml.core.cascade.frame_sources import Frame
from emil_ml.core.cascade.policy import ReactionPolicy
from emil_ml.core.detection.yolo import annotation as yolo_annotation
from emil_ml.utils import image_io
from emil_ml.utils.paths import for_component

# Every processed stream frame is thumbnailed unconditionally (unlike the
# identity-scoped, opt-in CASCADE_SAVED_FRAMES_DIR) and there is no
# retention/cleanup job for a component's cascade_stream_frames_dir yet (see
# utils/paths.py's own comment on that property) — downscaling keeps a
# long-running consumer's disk growth far slower, at no real cost to the
# results feed, which only ever displays these at a small fixed width
# anyway.
_THUMBNAIL_MAX_DIMENSION = 640


def should_sample(last_processed_position: float | None, sample_rate_seconds: float, current_position: float) -> bool:
    """Whether a frame at `current_position` (a Frame's own
    `position_seconds` — see frame_sources.py) is worth running through the
    cascade, given the last-processed frame's position (None if no frame
    has been processed yet — the very first frame is always sampled).

    Compares Frame.position_seconds values, never wall-clock
    `datetime.now()` — a live Kafka feed needs real elapsed time between
    arrivals, but an uploaded video decoding faster than real-time needs to
    be throttled against the video's OWN timeline (frame_index / fps), not
    how fast the file happens to decode, or the same setting would mean a
    different thing for each source. Each frame source fills in
    position_seconds with the right meaning for itself; this function only
    ever compares those.
    """
    if last_processed_position is None:
        return True
    return (current_position - last_processed_position) >= sample_rate_seconds


def process_frame(
    frame: Frame,
    component: Component,
    *,
    run_id: int,
    source: str,
    registry: ComponentRegistry | None = None,
    on_action: Callable[[str, ReactionPolicy], None] | None = None,
) -> stream_store.CascadeStreamResult:
    """Run the cascade on one already-sampled frame and persist the
    result. Callers must call should_sample() themselves first — this
    function always processes what it's given."""
    cascade_result = cascade_pipeline.run_cascade(frame.raw, component.name, registry=registry, on_action=on_action)

    component_paths = for_component(component.name)
    directory = component_paths.cascade_stream_frames_dir
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    thumbnail_path = directory / f"{timestamp}.png"

    # Boxes are drawn BEFORE downscaling — their coordinates come from
    # run_cascade() against the full-size frame, so drawing them after
    # thumbnail() would misplace every box relative to the shrunk image.
    thumbnail = image_io.to_pil(frame.raw).convert("RGB")
    drawable = [
        {"class": obj.label, "confidence": obj.confidence, "box": list(obj.box)}
        for obj in cascade_result.objects
        if obj.box is not None
    ]
    if drawable:
        thumbnail = yolo_annotation.render_boxes_on_image(thumbnail, drawable)
    thumbnail.thumbnail((_THUMBNAIL_MAX_DIMENSION, _THUMBNAIL_MAX_DIMENSION))
    thumbnail.save(thumbnail_path)

    # File a copy under analyzed/<identity>/ for every identity the cascade
    # actually recognized in this frame (never for "unknown" — that's a
    # specialist ran but found no match, not a recognition) — same
    # analyzed/<category>/ convention every other model_type already uses
    # (core/inspections/lifecycle.py's save_analyzed_image(), analyzed/
    # approved and analyzed/failed), just with the category being an
    # identity instead of a verdict. One frame with several recognized
    # people files one copy into each of their folders.
    for obj in cascade_result.objects:
        sr = obj.specialist_result
        if sr is None or not sr.matched:
            continue
        identity_dir = component_paths.analyzed_identity_dir(sr.identity_key)
        identity_dir.mkdir(parents=True, exist_ok=True)
        thumbnail.save(identity_dir / f"{sr.identity_key}_{timestamp}.png")

    return stream_store.record_result(
        run_id,
        component_name=component.name,
        source=source,
        frame_ref=frame.frame_ref,
        objects=cascade_result.objects,
        thumbnail_path=str(thumbnail_path),
    )
