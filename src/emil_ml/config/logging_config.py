"""Central logging configuration for EMIL Lab — one call, at each entry
point's own startup, sets up every `emil_ml.*` logger's destination and
format for that process. Modules never configure logging themselves;
they just get their own logger via `logging.getLogger(__name__)` and log
to it — where that ends up (console, file, both) is decided here, once,
not scattered across every module that happens to log something.

One central, chronologically-ordered log per day, not one per component:
most of what matters most to see is system-level, not component-level
(the watcher's own start/stop/poll, an Ollama/ChromaDB failure, a
background thread crashing) — and the hardest bugs often span components
(e.g. a cross-component retrieval leak where a transistor document got
retrieved for a toothbrush query: that's only visible in one unified
log, never in per-component files, and only ever obvious at a glance,
not by recognizing an incident number). A single file per day also
preserves event ordering between components, which per-component logs
would lose. Component-scoped filtering is still possible without
splitting the file — see retriever.py/orchestrator.py for the
`component=<name>` convention every component-related log line follows
(a stable `grep component=tandborste` pattern), while system-level lines
(watcher start/stop, Ollama connection errors, general infrastructure)
carry no such field, since forcing one onto them would misrepresent them
as being about a specific component when they aren't.

configure_logging() is idempotent — safe to call more than once in the
same process (e.g. once per Streamlit page script, since each one is a
separate script Streamlit re-runs) without attaching duplicate handlers;
a second call only updates the level.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from emil_ml.config.settings import (
    DEFAULT_LOG_LEVEL,
    LOG_DIR,
    LOG_FILE_DATE_FORMAT,
    LOG_FILE_PREFIX,
    LOG_FILE_SUFFIX,
)

_LOG_FORMAT = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"

# The one logger every emil_ml.* module's logger is a child of (Python's
# logging hierarchy is dot-separated, so "emil_ml.watcher.service" etc.
# all route through this) — handlers go here, not on the true root
# logger, so this configuration only ever affects our own app's loggers,
# never a third-party library's.
_APP_LOGGER_NAME = "emil_ml"
_configured = False


class _DailyFileHandler(logging.Handler):
    """Writes to `LOG_DIR/log<YYYYMMDD>.txt`, opening a new file the
    moment a record is emitted after the date has changed.

    Not logging.handlers.TimedRotatingFileHandler: that writes the
    *active* file under one fixed name and only renames it to a dated
    name once rotation happens — so a currently-being-written file
    wouldn't actually carry today's date until the day was already over.
    Here the active file is always named for the day it's actually being
    written on, and a long-lived process (the watcher, in particular, or
    a Streamlit server left running overnight) rolls over on its own at
    midnight without needing a restart.
    """

    def __init__(self, log_dir: Path, *, encoding: str = "utf-8") -> None:
        super().__init__()
        self._log_dir = log_dir
        self._encoding = encoding
        self._current_date: str | None = None
        self._stream = None

    def _path_for(self, date_str: str) -> Path:
        return self._log_dir / f"{LOG_FILE_PREFIX}{date_str}{LOG_FILE_SUFFIX}"

    def _ensure_stream(self) -> None:
        today = datetime.now().strftime(LOG_FILE_DATE_FORMAT)
        if today == self._current_date:
            return
        if self._stream is not None:
            self._stream.close()
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._stream = open(self._path_for(today), "a", encoding=self._encoding)
        self._current_date = today

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._ensure_stream()
            self._stream.write(self.format(record) + "\n")
            self._stream.flush()
        except Exception:  # noqa: BLE001 - logging.Handler's own contract: report, don't raise
            self.handleError(record)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
        super().close()


def configure_logging(level: str | int | None = None) -> None:
    """Set up console + daily-file handlers on the `emil_ml` logger tree.

    Call once, at process startup, from each entry point (the Streamlit
    app, `python -m emil_ml.watcher`, and the verify_*.py scripts) —
    never from inside a module that just wants to log something (see
    this module's own docstring). `level` defaults to
    config/settings.py's DEFAULT_LOG_LEVEL (itself overridable via the
    EMIL_LOG_LEVEL env var); the watcher passes its own --log-level flag
    through explicitly instead, since it has a CLI to carry one.
    """
    global _configured

    resolved_level = level if level is not None else DEFAULT_LOG_LEVEL
    if isinstance(resolved_level, str):
        resolved_level = getattr(logging, resolved_level.upper(), logging.INFO)

    logger = logging.getLogger(_APP_LOGGER_NAME)
    logger.setLevel(resolved_level)

    if _configured:
        return
    _configured = True

    formatter = logging.Formatter(_LOG_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = _DailyFileHandler(LOG_DIR)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Don't also hand every record up to the true root logger — this
    # logger's own two handlers (console + file) are the complete,
    # intended destination; propagating further risks double-printing if
    # something else (a library, a test runner) has its own root handler.
    logger.propagate = False
