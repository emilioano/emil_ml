"""Standalone folder watcher — a production entry point, not a pipeline.

Watches every registered component's input/ directory and, when a fully
written file appears, calls the exact same entry point the Streamlit UI
does (core.inspections.orchestrator.run_inspection(), which itself calls
pipeline.inspect() — see that module's own docstring: "Used identically
by the Streamlit UI and the folder watcher, so verdicts are guaranteed
consistent regardless of caller.") This module contains no detection or
RAG logic of its own; its whole job is notice a file, hand it to
run_inspection(), react to the outcome.

Runs as its own long-lived process (see __main__.py — `python -m
emil_ml.watcher`), independent of Streamlit: a Streamlit session is
short-lived and gets restarted by the server on every code change or
crash, which would kill a watcher started inside it. In practice this
module IS the system's backend — the same independent entry point a
future non-Streamlit frontend would talk to, alongside the UI.

Registry-driven, one process for every component (not a thread per
component) — component.name determines nothing about which input/ a
file was dropped in; which input/ directory it appeared in is what
determines which component it belongs to.

Two discovery mechanisms, not one:
- watchdog filesystem events (immediate, but unreliable enough on their
  own — especially over a network share a production camera might write
  to, exactly the deployment this is built for — that they can't be the
  only mechanism).
- a periodic full rescan of every input/ dir (WATCHER_POLL_INTERVAL_SECONDS,
  see _poll_once()) as the safety net that catches whatever an event
  missed. Also how a newly onboarded component's input/ starts being
  watched without restarting the process — see _sync_watches().

Both discovery paths feed the same queue, deduplicated by an in-flight
set (keyed by path) so a file whose event AND poll both notice it is
only ever processed once.

A single worker thread drains that queue sequentially — not because
concurrency would be wrong, but because detection itself is fast and
serializing it avoids concurrent-inference contention on whatever
backend (TF/Torch) a component's predictor uses. The slow part — report
generation — already runs in its own background thread inside
run_inspection() (async_report=True, its default): a report taking
qwen3:8b a couple of minutes never blocks the next file in the queue.

Stability check (see _wait_until_stable()) is the single most important
robustness detail here: a camera can write a partial frame, and acting
on a file mid-write produces a silent or cryptic downstream error
instead of a clean "wait, then process." A file is only handed to
run_inspection() once its size has stopped changing across several
consecutive checks.

A file that can't be processed (corrupt image, a not-ready component,
any unexpected error) is moved to that component's error/ directory with
a log line — never retried forever from input/, never silently dropped.
A file that's processed successfully is simply removed from input/:
run_inspection() -> lifecycle.save_analyzed_image() already wrote the
durable, collision-safe copy into analyzed/{approved|failed}/ (that
function is deliberately timestamp-named rather than keeping the
original filename for exactly this reason — see its own docstring), so
copying the original there too would just create a duplicate image.
"""

from __future__ import annotations

import logging
import queue
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import ObservedWatch

from emil_ml.config.registry import ComponentRegistry
from emil_ml.config.settings import (
    WATCHER_POLL_INTERVAL_SECONDS,
    WATCHER_STABILITY_CHECK_INTERVAL_SECONDS,
    WATCHER_STABILITY_REQUIRED_CHECKS,
)
from emil_ml.core.inspections import orchestrator
from emil_ml.utils.paths import for_component

logger = logging.getLogger(__name__)

# Same set image_io.py/the Streamlit uploaders already accept — kept as
# its own small constant here rather than importing image_io's private
# _IMAGE_EXTENSIONS, since this module cares about it for a different
# reason (filtering which filesystem events/poll results are candidates
# at all, before any image-loading is attempted).
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_TIMESTAMP_FMT = "%Y%m%d_%H%M%S_%f"


def _is_candidate_file(path: Path) -> bool:
    return path.suffix.lower() in _IMAGE_EXTENSIONS and path.is_file()


def _wait_until_stable(
    path: Path,
    *,
    check_interval: float = WATCHER_STABILITY_CHECK_INTERVAL_SECONDS,
    required_checks: int = WATCHER_STABILITY_REQUIRED_CHECKS,
) -> bool:
    """Poll `path`'s size until it hasn't changed for `required_checks`
    consecutive checks, `check_interval` apart.

    Returns False if the file disappears before stabilizing (a rename
    race, or it was already handled some other way) — vanishing mid-wait
    is a normal outcome to shrug off, not an error to raise.
    """
    stable_count = 0
    last_size = -1
    while stable_count < required_checks:
        try:
            size = path.stat().st_size
        except OSError:
            return False
        stable_count = stable_count + 1 if size == last_size else 0
        last_size = size
        if stable_count < required_checks:
            time.sleep(check_interval)
    return True


class WatcherService:
    """Owns the watchdog Observer, the discovery queue, and the worker
    thread. `run_forever()` blocks until `stop()` is called (e.g. from a
    signal handler in __main__.py)."""

    def __init__(
        self,
        *,
        registry: ComponentRegistry | None = None,
        poll_interval: float = WATCHER_POLL_INTERVAL_SECONDS,
        stability_check_interval: float = WATCHER_STABILITY_CHECK_INTERVAL_SECONDS,
        stability_required_checks: int = WATCHER_STABILITY_REQUIRED_CHECKS,
    ) -> None:
        self.registry = registry or ComponentRegistry()
        self.poll_interval = poll_interval
        self.stability_check_interval = stability_check_interval
        self.stability_required_checks = stability_required_checks

        self._queue: queue.Queue[tuple[str, Path]] = queue.Queue()
        self._in_flight: set[Path] = set()
        self._in_flight_lock = threading.Lock()
        self._watched_dirs: dict[str, str] = {}  # str(input_dir) -> component_name
        self._scheduled_watches: dict[str, ObservedWatch] = {}  # str(input_dir) -> watchdog handle, for unschedule()

        self._observer = Observer()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._poll_thread: threading.Thread | None = None

    # --- discovery: watchdog events --------------------------------------

    def _enqueue(self, component_name: str, path: Path, *, via: str) -> None:
        with self._in_flight_lock:
            if path in self._in_flight:
                return
            self._in_flight.add(path)
        logger.debug("component=%s discovered %s via %s", component_name, path.name, via)
        self._queue.put((component_name, path))

    def _handle_event_path(self, raw_path: str) -> None:
        path = Path(raw_path)
        if not _is_candidate_file(path):
            return
        component_name = self._watched_dirs.get(str(path.parent))
        if component_name is None:
            return  # an event for a directory we're not (or no longer) watching
        self._enqueue(component_name, path, via="event")

    def _sync_watches(self) -> None:
        """Add a watchdog schedule for every currently-ACTIVE registered
        component's input/ that isn't already being watched, and remove
        the schedule for any component that's no longer active
        (deactivated, soft-deleted, or otherwise gone) since the last
        sync — so deactivating a component takes effect on an already-
        running watcher within one poll interval, not just at next
        restart.

        Called at startup and on every poll tick — the latter is also
        what lets a newly onboarded (or reactivated) component start
        being watched without restarting the process, at the cost of it
        taking up to one poll interval to notice, not zero.

        Uses list_active() (lifecycle_status='active'), not list_all() —
        an inactive or soft-deleted component's input/ is never watched,
        matching every other active-use iteration point (RAG's
        index_all(), the Inspect page's component picker).
        """
        active_components = self.registry.list_active()
        active_input_dirs: set[str] = set()
        for component in active_components:
            paths = for_component(component.name)
            paths.input_dir.mkdir(parents=True, exist_ok=True)
            key = str(paths.input_dir)
            active_input_dirs.add(key)
            if key in self._watched_dirs:
                continue
            self._watched_dirs[key] = component.name
            self._scheduled_watches[key] = self._observer.schedule(_Handler(self), key, recursive=False)
            logger.info("component=%s watching %s", component.name, key)

        for key in list(self._watched_dirs):
            if key in active_input_dirs:
                continue
            component_name = self._watched_dirs.pop(key)
            watch = self._scheduled_watches.pop(key, None)
            if watch is not None:
                self._observer.unschedule(watch)
            logger.info("component=%s no longer active — stopped watching %s", component_name, key)

    # --- discovery: periodic poll safety net -----------------------------

    def _poll_once(self) -> None:
        self._sync_watches()
        logger.debug("poll scan: checking %d watched input/ dir(s)", len(self._watched_dirs))
        for input_dir_str, component_name in list(self._watched_dirs.items()):
            input_dir = Path(input_dir_str)
            if not input_dir.exists():
                continue
            for path in input_dir.iterdir():
                if _is_candidate_file(path):
                    self._enqueue(component_name, path, via="poll")

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._poll_once()
            except Exception:  # noqa: BLE001 - the poll loop must never die
                logger.exception("Error during periodic input/ rescan")
            self._stop_event.wait(self.poll_interval)

    # --- processing --------------------------------------------------------

    def _move_to_error(self, component_name: str, path: Path, reason: str) -> None:
        error_dir = for_component(component_name).error_dir
        error_dir.mkdir(parents=True, exist_ok=True)
        # Timestamp-prefixed for the same reason lifecycle.py's analyzed/
        # filenames are: a repeatedly-numbered camera frame ("017.png"
        # every cycle) must not silently overwrite a previous failure
        # still sitting here waiting to be looked at.
        timestamp = datetime.now(timezone.utc).strftime(_TIMESTAMP_FMT)
        dest = error_dir / f"{timestamp}_{path.name}"
        try:
            shutil.move(str(path), str(dest))
        except OSError as move_exc:
            # The error-routing mechanism itself failing is a genuine
            # ERROR, not graceful degradation — unlike the normal
            # "couldn't process this file" case below, this means the
            # file is now stuck in neither input/ nor error/.
            logger.error(
                "component=%s could not move %s to error/ (%s) — original failure: %s",
                component_name, path, move_exc, reason,
            )
            return
        # WARNING, not ERROR: this is the system degrading gracefully
        # exactly as designed (see module docstring) — a bad file is
        # routed aside and the queue keeps moving. It still needs to be
        # visible, not silent, which is what WARNING is for here.
        logger.warning("component=%s moved %s to error/ (%s): %s", component_name, path.name, dest, reason)

    def _process_file(self, component_name: str, path: Path) -> None:
        logger.debug("component=%s waiting for %s to finish writing", component_name, path.name)
        if not _wait_until_stable(
            path,
            check_interval=self.stability_check_interval,
            required_checks=self.stability_required_checks,
        ):
            logger.debug("component=%s %s vanished before it finished writing, skipping", component_name, path.name)
            return
        if not path.exists():
            return

        logger.debug("component=%s %s stable, starting inspection", component_name, path.name)
        try:
            record, _details = orchestrator.run_inspection(
                path, component_name, registry=self.registry, run_by="watcher"
            )
        except Exception as exc:  # noqa: BLE001 - any failure routes to error/, never crashes the watcher
            self._move_to_error(component_name, path, str(exc))
            return

        # run_inspection() already persisted the durable, collision-safe
        # copy into analyzed/{approved|failed}/ and created the DB
        # record (see module docstring) — this file's content already
        # lives on safely, so all that's left is removing the original
        # from input/ so it's never picked up again.
        try:
            path.unlink()
        except OSError:
            pass  # already gone somehow; not fatal either way
        logger.info(
            "component=%s processed %s: verdict=%s score=%.6f moved_to=%s (inspection id=%s)",
            component_name,
            path.name,
            record.verdict,
            record.score,
            record.image_path,
            record.id,
        )

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                component_name, path = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                self._process_file(component_name, path)
            except Exception:  # noqa: BLE001 - the worker loop must never die
                logger.exception("component=%s unexpected error processing %s", component_name, path)
            finally:
                with self._in_flight_lock:
                    self._in_flight.discard(path)
                self._queue.task_done()

    # --- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        self._sync_watches()
        # Initial scan: pick up anything already sitting in input/ from
        # before this process started — a previous run that crashed, or
        # files that arrived while the watcher was down — not just new
        # events from this point on.
        self._poll_once()

        self._observer.start()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="watcher-worker")
        self._worker_thread.start()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True, name="watcher-poll")
        self._poll_thread.start()
        logger.info("Watcher started, watching %d component(s)", len(self._watched_dirs))

    def stop(self) -> None:
        self._stop_event.set()
        self._observer.stop()
        self._observer.join(timeout=5)
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5)
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=5)
        logger.info("Watcher stopped.")

    def run_forever(self) -> None:
        self.start()
        try:
            while not self._stop_event.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


class _Handler(FileSystemEventHandler):
    """Forwards watchdog events to a WatcherService.

    on_created covers a normal write-in-place. on_moved covers the
    common "write to a temp name, then atomically rename into place"
    pattern some cameras/tools use — by the time that rename fires the
    file is already fully written, though _wait_until_stable() still
    runs regardless (cheap and harmless when it's already stable).
    """

    def __init__(self, service: WatcherService) -> None:
        self._service = service

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._service._handle_event_path(event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._service._handle_event_path(event.dest_path)
