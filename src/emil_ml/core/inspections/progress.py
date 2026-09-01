"""In-memory, per-process live progress for report generation.

Report generation runs in a background thread (orchestrator.py) and can
take up to a couple of minutes — this module is what lets the UI show
what's actually happening while it waits (stage messages, plus the LLM's
"thinking" and response text streaming in token by token) instead of a
static spinner.

Deliberately NOT persisted anywhere — no new DB column, no file. It's
transient, per-attempt scratch state that's meaningless once generation
finishes (the final report_text/report_prompt/report_thinking already
ARE persisted in store.py, for after-the-fact inspection — see the
"verbose" expander in app/pages/1_inspect.py and 3_history.py). This is
only for watching it happen live.

A dict keyed by inspection_id, guarded by a lock since the background
thread writes and the Streamlit UI thread reads concurrently — fine for
this project's single-process deployment; a multi-process one would need
a real message bus instead of a module-level dict.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

_lock = threading.Lock()


@dataclass
class ProgressState:
    stages: list[str] = field(default_factory=list)
    thinking: str = ""
    response: str = ""


_state: dict[int, ProgressState] = {}


def start(inspection_id: int) -> None:
    with _lock:
        _state[inspection_id] = ProgressState()


def add_stage(inspection_id: int, message: str) -> None:
    with _lock:
        state = _state.get(inspection_id)
        if state is not None:
            state.stages.append(message)


def set_chunk(inspection_id: int, *, thinking: str | None = None, response: str | None = None) -> None:
    """Overwrite the live thinking/response text with the latest accumulated
    value (llm.py passes the full text-so-far on every call, not a delta,
    so the UI never has to reassemble fragments itself)."""
    with _lock:
        state = _state.get(inspection_id)
        if state is None:
            return
        if thinking is not None:
            state.thinking = thinking
        if response is not None:
            state.response = response


def get(inspection_id: int) -> ProgressState | None:
    """A snapshot copy, never the live object — the background thread may
    still be mutating it between this call and whenever the UI reads it."""
    with _lock:
        state = _state.get(inspection_id)
        if state is None:
            return None
        return ProgressState(stages=list(state.stages), thinking=state.thinking, response=state.response)


def clear(inspection_id: int) -> None:
    with _lock:
        _state.pop(inspection_id, None)
