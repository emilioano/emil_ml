"""CascadeStreamService — consumes a Kafka topic and runs the object/face
cascade on a throttled sample of its frames, forever, until stopped.

A standalone, long-lived process (see this package's own `__init__.py`
docstring for why it must never be started from inside a Streamlit
session) — the direct analog of `emil_ml.watcher.service.WatcherService`,
but simpler: Kafka's own `Consumer.poll()` loop already IS the discovery
mechanism (a message arrives or it doesn't), so there's no watchdog+poll
dual mechanism and no separate worker thread/queue to decouple slow work
from fast discovery the way the watcher needs — cascade dispatch is
already throttled to roughly `cascade_stream_sample_rate_seconds` by
should_sample(), so one straightforward loop is enough.

Calls core/cascade/pipeline.py's run_cascade() (via
core/cascade/stream_processor.py), never core/inspections/orchestrator.py's
run_inspection() — coco_detector components have no approved/failed
verdict (see core/registry_factory.py's is_cascade_only(), which start()
below enforces before ever opening a Kafka connection).
"""

from __future__ import annotations

import logging
import threading
import time

from emil_ml.config.registry import Component, ComponentRegistry
from emil_ml.config.settings import CASCADE_STREAM_HEARTBEAT_INTERVAL_SECONDS, CASCADE_STREAM_KAFKA_POLL_TIMEOUT_SECONDS
from emil_ml.core import registry_factory
from emil_ml.core.cascade import stream_processor, stream_store
from emil_ml.core.cascade.frame_sources import KafkaFrameSource

logger = logging.getLogger(__name__)


class CascadeStreamService:
    def __init__(
        self,
        component_name: str,
        *,
        registry: ComponentRegistry | None = None,
        heartbeat_interval: float = CASCADE_STREAM_HEARTBEAT_INTERVAL_SECONDS,
        poll_timeout: float = CASCADE_STREAM_KAFKA_POLL_TIMEOUT_SECONDS,
    ) -> None:
        self.component_name = component_name
        self.registry = registry or ComponentRegistry()
        self.heartbeat_interval = heartbeat_interval
        self.poll_timeout = poll_timeout
        self._stop_event = threading.Event()

    def _load_component(self) -> Component:
        """Fails fast with a clear message rather than silently no-op'ing
        — a misconfigured or not-yet-ready component should never look
        like a quietly-idle stream."""
        component = self.registry.get(self.component_name)
        if component is None:
            raise KeyError(f"No component named {self.component_name!r}")
        if not registry_factory.is_cascade_only(component.model_type):
            raise ValueError(
                f"Component {self.component_name!r} (model_type={component.model_type!r}) isn't a "
                "cascade-only component — the cascade stream only runs coco_detector components."
            )
        if component.status != "ready":
            raise ValueError(f"Component {self.component_name!r} is not ready (status={component.status!r})")
        if not component.cascade_stream_kafka_bootstrap_servers or not component.cascade_stream_kafka_topic:
            raise ValueError(
                f"Component {self.component_name!r} has no Kafka bootstrap servers/topic configured — "
                "set them on the Cascade Stream page first."
            )
        return component

    def stop(self) -> None:
        self._stop_event.set()

    def run_forever(self) -> None:
        """Runs until stop() is called (e.g. from a SIGINT/SIGTERM
        handler, see __main__.py) or an unhandled exception occurs.
        Always leaves the run row in a terminal state ('stopped' on a
        clean stop, 'crashed' with the error recorded otherwise) — the
        Cascade Stream page relies on this to distinguish the two."""
        component = self._load_component()
        run = stream_store.start_run(
            component.name, source="kafka", source_detail=component.cascade_stream_kafka_topic
        )

        # source is constructed INSIDE the try block, not before it — a
        # construction failure (missing confluent_kafka, a malformed
        # bootstrap.servers string, ...) must still mark the run 'crashed'
        # with a real error message, not leave it stuck at 'running'
        # forever with no explanation once its heartbeat goes stale.
        source: KafkaFrameSource | None = None
        last_processed_position: float | None = None
        frames_seen = 0
        frames_processed = 0
        status = "stopped"
        error: str | None = None
        try:
            source = KafkaFrameSource(
                component.cascade_stream_kafka_bootstrap_servers,
                component.cascade_stream_kafka_topic,
                group_id=f"emil-cascade-stream-{component.name}",
                poll_timeout=self.poll_timeout,
            )
            logger.info(
                "Cascade stream started: component=%s topic=%s sample_rate=%ss",
                component.name,
                component.cascade_stream_kafka_topic,
                component.cascade_stream_sample_rate_seconds,
            )

            last_heartbeat_monotonic = time.monotonic()
            for frame in source.frames(should_stop=self._stop_event.is_set):
                frames_seen += 1
                if stream_processor.should_sample(
                    last_processed_position, component.cascade_stream_sample_rate_seconds, frame.position_seconds
                ):
                    stream_processor.process_frame(
                        frame, component, run_id=run.id, source="kafka", registry=self.registry
                    )
                    last_processed_position = frame.position_seconds
                    frames_processed += 1

                if time.monotonic() - last_heartbeat_monotonic >= self.heartbeat_interval:
                    stream_store.heartbeat(run.id, frames_seen=frames_seen, frames_processed=frames_processed)
                    last_heartbeat_monotonic = time.monotonic()
        except Exception as exc:
            status = "crashed"
            error = str(exc)
            logger.exception("Cascade stream crashed: component=%s", component.name)
            raise
        finally:
            stream_store.heartbeat(run.id, frames_seen=frames_seen, frames_processed=frames_processed)
            stream_store.finish_run(run.id, status=status, error=error)
            if source is not None:
                source.close()
            logger.info(
                "Cascade stream stopped: component=%s frames_seen=%d frames_processed=%d",
                component.name,
                frames_seen,
                frames_processed,
            )
