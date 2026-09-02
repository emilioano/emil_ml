"""Frame sources for continuous cascade operation: a live Kafka topic
(KafkaFrameSource) or an uploaded video file (VideoFileFrameSource). Both
yield the same `Frame` shape, so core/cascade/stream_processor.py's
per-frame logic never needs to know which one it's talking to — only where
the frame comes from differs (see app/pages/2_onboard.py's own foreshadowing
comment on run_cascade()).

Each source's heavy client library (confluent_kafka, cv2) is imported
LAZILY, inside its class's `__init__`, never at module top level — same
convention core/anomaly/patchcore, core/reporting/knowledge, and
core/cascade/specialists/face already follow (see their own modules' and
pyproject.toml's `patchcore`/`rag`/`cascade` extras' comments): importing
this module (which core/cascade/stream_processor.py and
app/pages/5_cascade_stream.py both do unconditionally) must never require
the `kafka` or `video` extra installed unless that specific source is
actually constructed.

Deliberately duck-typed (`frames() -> Iterator[Frame]`, `close()`), not a
formal ABC/Protocol — the caller always knows statically which concrete
source it's using, unlike core/modality/base.py's BaseModalityHandler,
which genuinely needs registry-driven dispatch.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Frame:
    """One frame from any source, in the shape
    core/cascade/stream_processor.py needs — never constructed directly by
    callers outside this module."""

    raw: Any  # bytes (Kafka message value) or an RGB ndarray (decoded video frame) — both valid run_cascade() raw_input, see utils/image_io.py's to_pil()
    frame_ref: str  # "partition:offset" (kafka) or "frame842@28.07s" (video) — persisted verbatim into cascade_stream_results.frame_ref
    position_seconds: float  # meaning is source-specific (real elapsed time for Kafka, video-timeline position for a file) — see stream_processor.should_sample()'s own docstring for why this must never be wall-clock datetime.now()
    received_at: datetime  # UTC wall-clock time this Frame was constructed — display/telemetry only, never used for throttling


class KafkaFrameSource:
    """Consumes raw image-byte messages from one Kafka topic, one frame per
    message — see cascade_stream/service.py, the only caller."""

    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        *,
        group_id: str,
        poll_timeout: float,
    ) -> None:
        from confluent_kafka import Consumer

        self._topic = topic
        self._poll_timeout = poll_timeout
        self._start_monotonic = time.monotonic()
        self._consumer = Consumer(
            {
                "bootstrap.servers": bootstrap_servers,
                "group.id": group_id,
                "auto.offset.reset": "latest",
                # Auto-commit is deliberately left on (the client's own
                # default) rather than manually managed: the sampling
                # throttle already discards the overwhelming majority of
                # frames, so reprocessing a handful after a crash/restart
                # is a non-issue, not a correctness gap worth the extra
                # complexity of manual offset commits.
            }
        )
        self._consumer.subscribe([topic])

    def frames(self, *, should_stop: Callable[[], bool] | None = None) -> Iterator[Frame]:
        """Blocks in Consumer.poll() loops, one poll_timeout at a time,
        until the topic is idle-forever or `should_stop()` returns True —
        checked on EVERY poll cycle, not just when a message actually
        arrives, since an idle topic would otherwise never give the caller
        a chance to stop (poll() returning None doesn't end this
        generator on its own). Never raises for "no message yet" or
        "broker unreachable"; both just mean the next poll() returns None
        and the loop tries again, so a down broker degrades to quiet
        retries, not a crash."""
        while should_stop is None or not should_stop():
            msg = self._consumer.poll(self._poll_timeout)
            if msg is None:
                continue
            if msg.error() is not None:
                logger.warning("Kafka consumer error on topic %s: %s", self._topic, msg.error())
                continue
            yield Frame(
                raw=msg.value(),
                frame_ref=f"{msg.partition()}:{msg.offset()}",
                position_seconds=time.monotonic() - self._start_monotonic,
                received_at=datetime.now(timezone.utc),
            )

    def close(self) -> None:
        self._consumer.close()


class VideoFileFrameSource:
    """Decodes every frame of a local video file, in order, via OpenCV —
    see app/pages/5_cascade_stream.py, the only caller (the uploaded file
    is written to a temp path first; this needs a real path, not in-memory
    bytes)."""

    def __init__(self, path: Path) -> None:
        import cv2

        self._cv2 = cv2
        self._capture = cv2.VideoCapture(str(path))
        fps = self._capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            logger.warning("Video %s reports an invalid FPS (%r); frame positions will all read as 0.0s", path, fps)
        self._fps = fps or 0.0

    def frames(self) -> Iterator[Frame]:
        cv2 = self._cv2
        frame_index = 0
        while True:
            ok, frame_bgr = self._capture.read()
            if not ok:
                return
            # cv2 decodes BGR; utils/image_io.py's to_pil() does a bare
            # Image.fromarray() on an ndarray with no channel reordering,
            # so every downstream use (face embeddings, saved thumbnails,
            # the annotated preview) would be silently color-swapped
            # without this conversion.
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            position_seconds = (frame_index / self._fps) if self._fps else 0.0
            yield Frame(
                raw=frame_rgb,
                frame_ref=f"frame{frame_index}@{position_seconds:.2f}s",
                position_seconds=position_seconds,
                received_at=datetime.now(timezone.utc),
            )
            frame_index += 1

    def frame_count(self) -> int:
        """Best-effort total frame count for a progress bar — 0 if the
        container doesn't report one reliably; callers should fall back to
        an indeterminate spinner in that case, not divide by zero."""
        count = int(self._capture.get(self._cv2.CAP_PROP_FRAME_COUNT))
        return count if count > 0 else 0

    def close(self) -> None:
        self._capture.release()
