"""Serializes report generation against Ollama across a whole process.

Detection (pipeline.inspect(), orchestrated by orchestrator.run_inspection())
stays fully synchronous and unblocked — this module only affects the ASYNC
report-generation step orchestrator.py kicks off after detection returns.

Report generation calls Ollama twice per report — a query embedding
(nomic-embed-text, in core/reporting/knowledge/indexer.py's embed(), used
by retriever.py at query time) and the actual LLM generation (qwen3:8b, in
llm.py) — and a single local GPU can only run one qwen3:8b generation at a
time with reasonable throughput. Running several concurrently doesn't
speed anything up; it makes every one of them slower, to the point of
blowing straight through Ollama's own request timeout. Confirmed via
`ollama ps` this was never a memory problem — both models comfortably fit
in VRAM together (5.6 GB + 323 MB of 8 GB) — purely a concurrency one:
four qwen3:8b generations starting within 0.3s of each other, all timing
out waiting behind each other for the same GPU.

The fix: a single persistent worker thread per process (started lazily,
the first time a report is queued) drains a FIFO queue of report jobs one
at a time — the exact same "single sequential worker" principle
watcher/service.py already uses for inspections (see its _worker_loop),
just extended one layer further, to reports. orchestrator.run_inspection()
still returns immediately with report_status='pending' the instant
detection finishes; only the actual Ollama calls a report needs are now
serialized against every OTHER report's Ollama calls in this process, not
against detection, which never touches this queue at all.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable

logger = logging.getLogger(__name__)

_ReportJob = Callable[[], None]

_queue: "queue.Queue[_ReportJob]" = queue.Queue()
_worker_thread: threading.Thread | None = None
_worker_lock = threading.Lock()


def _worker_loop() -> None:
    logger.info("Report worker started — report generations now run one at a time against Ollama")
    while True:
        job = _queue.get()
        try:
            job()
        except Exception:  # noqa: BLE001 - the worker loop must never die; one bad job can't take down every future report
            logger.exception("Report worker: a queued report job raised unexpectedly")
        finally:
            _queue.task_done()


def _ensure_worker_started() -> None:
    global _worker_thread
    with _worker_lock:
        if _worker_thread is None or not _worker_thread.is_alive():
            _worker_thread = threading.Thread(target=_worker_loop, daemon=True, name="report-worker")
            _worker_thread.start()


def enqueue(job: _ReportJob) -> None:
    """Queue one report-generation job onto the single, process-wide,
    serialized report worker — never spawns a new thread per call, so
    concurrent report requests (e.g. the watcher processing a burst of
    files) never hit Ollama at the same time. The worker thread starts
    lazily on first use and lives for the rest of the process; nothing
    needs to stop it explicitly (daemon thread, same as the watcher's
    own worker/poll threads).
    """
    _ensure_worker_started()
    _queue.put(job)
