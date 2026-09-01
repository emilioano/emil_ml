"""Swappable text-generation backend — one function, same signature and
return shape regardless of which mode is active, so reporter.py never
knows or cares which backend actually produced the text.

Three modes (config/settings.py's DEFAULT_LLM_MODE picks the default):
    mock    Fixed, deterministic response, no network call. Default
            during development — isolates "is the orchestration right"
            (retrieve -> analyze -> prompt -> generate -> ReportResult)
            from "is the generated text good", the same way every earlier
            phase in this project was verified without an LLM in the loop
            first.
    ollama  Calls a local Ollama instance (OLLAMA_GENERATE_URL), default
            model qwen3:8b. Streamed (stream=true) — not because the final
            LLMResult needs it (it's still assembled into one full
            text/thinking pair, same as before), but so a caller can watch
            it arrive live via `on_chunk` (see app/pages/1_inspect.py's
            auto-refreshing report section). Confirmed directly against
            this model's real streaming shape: each NDJSON line carries an
            incremental "thinking" token during the reasoning phase, then
            incremental "response" tokens once it starts answering — never
            both accumulated at once, so `on_chunk(kind, text_so_far)` is
            called with whichever one is actually growing. Connection/
            timeout failures degrade to an LLMResult with `error` set and
            an honest text saying generation failed, rather than raising
            and crashing the pipeline — same "never crash, degrade
            honestly" principle reporter.py already applies to missing
            documentation. Default timeout is generous (240s) and, with
            streaming, applies between chunks rather than to the whole
            call — qwen3:8b is a "thinking" model that can spend a long,
            variable stretch of time reasoning before it emits the actual
            response (confirmed directly: 71s on one run, >120s on
            another for a similar-sized prompt), but as long as SOME
            token arrives at least that often, the call doesn't time out.
    cloud   Stub for a higher-quality cloud API. Not implemented — raises
            clearly if selected, rather than silently falling back to
            another mode.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Callable

import requests

from emil_ml.config.settings import (
    DEFAULT_LLM_MODE,
    DEFAULT_RAG_LLM_MODEL,
    OLLAMA_GENERATE_URL,
    OLLAMA_MAX_RETRIES,
    OLLAMA_RETRY_BACKOFF_SECONDS,
    VALID_LLM_MODES,
)

logger = logging.getLogger(__name__)

_MOCK_RESPONSE_TEMPLATE = (
    "[MOCK LLM RESPONSE — no model was called]\n\n"
    "This is a fixed, deterministic placeholder used to verify the reporting pipeline's "
    "orchestration without a real LLM in the loop, citing [1] as an example. Inspect "
    "ReportResult.prompt_used to see exactly what would have been sent to a real model.\n\n"
    "## Sources\n[1] Mock source"
    # The [1] appears BOTH inline in the body above AND in the Sources
    # line, deliberately: reporter.py's _cited_chunks() only counts a
    # citation as real if it appears in the body, before the model's own
    # "## Sources" heading (see its own docstring — a real model can
    # otherwise pad Sources with numbers it never actually cited, and an
    # earlier version of this mock text accidentally exercised exactly
    # that loophole: its only "[1]" was inside Sources itself, so once
    # _cited_chunks() stopped counting that, mock mode would always
    # produce an empty source list — misrepresenting "orchestration
    # found and cited documentation" as "found none" in every mock-mode
    # verification, unrelated to what's actually being tested there.
)


@dataclass(frozen=True)
class LLMResult:
    """What every mode returns, regardless of backend."""

    text: str
    model: str
    error: str | None = None  # set when generation failed but didn't crash the caller
    thinking: str | None = None  # a "thinking" model's chain-of-thought, if it returned one


def generate(
    prompt: str,
    *,
    mode: str = DEFAULT_LLM_MODE,
    model: str | None = None,
    timeout: float = 240.0,
    on_chunk: Callable[[str, str], None] | None = None,
) -> LLMResult:
    """Generate text from `prompt` via the given backend mode.

    `on_chunk(kind, text_so_far)` — kind is "thinking" or "response" —
    is called as tokens stream in (ollama mode only; mock returns
    instantly and cloud isn't implemented, so neither has anything
    incremental to report). Purely a progress-observation hook: the
    returned LLMResult is unaffected either way.
    """
    if mode == "mock":
        return _generate_mock(model=model)
    if mode == "ollama":
        resolved_model = model or DEFAULT_RAG_LLM_MODEL
        logger.debug("LLM generate: mode=ollama model=%s prompt_len=%d", resolved_model, len(prompt))
        return _generate_ollama(prompt, model=resolved_model, timeout=timeout, on_chunk=on_chunk)
    if mode == "cloud":
        return _generate_cloud(prompt, model=model)
    raise ValueError(f"Unknown LLM mode {mode!r}; must be one of {VALID_LLM_MODES}")


def _generate_mock(*, model: str | None) -> LLMResult:
    return LLMResult(text=_MOCK_RESPONSE_TEMPLATE, model=model or "mock")


def _generate_ollama(
    prompt: str, *, model: str, timeout: float, on_chunk: Callable[[str, str], None] | None
) -> LLMResult:
    # Retries on failure (see OLLAMA_MAX_RETRIES in config/settings.py):
    # same transient cold-model-load 500 that indexer.embed() rides out —
    # the first request after Ollama's idle timeout unloads a model can
    # trip its GPU-discovery watchdog, while the very next request succeeds.
    # thinking_parts/response_parts are reset on every attempt — a retry
    # re-streams from scratch, so a previous attempt's partial tokens
    # (already surfaced via on_chunk, if it got that far) aren't carried
    # forward into the new one.
    last_exc: requests.exceptions.RequestException | None = None
    thinking_parts: list[str] = []
    response_parts: list[str] = []
    succeeded = False
    for attempt in range(OLLAMA_MAX_RETRIES):
        thinking_parts = []
        response_parts = []
        try:
            with requests.post(
                OLLAMA_GENERATE_URL,
                json={"model": model, "prompt": prompt, "stream": True},
                timeout=timeout,
                stream=True,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    # Never both non-empty on the same line — confirmed
                    # directly against qwen3:8b's real streaming output:
                    # "thinking" carries tokens during the reasoning phase,
                    # then "response" takes over once it starts answering.
                    thinking_delta = chunk.get("thinking")
                    if thinking_delta:
                        thinking_parts.append(thinking_delta)
                        if on_chunk:
                            on_chunk("thinking", "".join(thinking_parts))
                    response_delta = chunk.get("response")
                    if response_delta:
                        response_parts.append(response_delta)
                        if on_chunk:
                            on_chunk("response", "".join(response_parts))
            last_exc = None
            succeeded = True
            break
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt + 1 < OLLAMA_MAX_RETRIES:
                logger.warning(
                    "Ollama generate request failed (attempt %d/%d, model=%s): %s — retrying",
                    attempt + 1, OLLAMA_MAX_RETRIES, model, exc,
                )
                time.sleep(OLLAMA_RETRY_BACKOFF_SECONDS * (attempt + 1))

    if not succeeded:
        # Degrade honestly rather than raising — a report generation
        # failure shouldn't take down the inspection pipeline that
        # triggered it (pipeline.inspect() has already produced a valid
        # verdict by the time this runs).
        logger.warning("Ollama generate failed after %d attempt(s) (model=%s): %s", OLLAMA_MAX_RETRIES, model, last_exc)
        return LLMResult(
            text=f"Report generation failed: could not reach Ollama at {OLLAMA_GENERATE_URL} ({last_exc}).",
            model=model,
            error=str(last_exc),
        )

    # qwen3 (and other "thinking" models) separate chain-of-thought into
    # its own stream — only the assembled "response" text is the actual
    # answer, so it's what the report body is built from. The thinking
    # trace isn't part of the report, but is kept on LLMResult (and
    # threaded through to ReportResult.thinking_used) purely for
    # inspection/debugging — see the "verbose" expander in
    # app/pages/1_inspect.py and 3_history.py.
    text = "".join(response_parts).strip()
    thinking = "".join(thinking_parts).strip() or None
    if not text:
        return LLMResult(
            text="Report generation failed: Ollama returned an empty response.",
            model=model,
            error="empty response",
            thinking=thinking,
        )
    return LLMResult(text=text, model=model, thinking=thinking)


def _generate_cloud(prompt: str, *, model: str | None) -> LLMResult:
    raise NotImplementedError(
        "The 'cloud' LLM mode is a stub — no cloud API is wired up yet. Use 'mock' or 'ollama'."
    )
