"""Executes a reaction policy: identity-agnostic lookup -> perform its
configured actions. See policy.py's module docstring for the design
principle this follows (machine_context/analyzer.py's parameter-agnostic
iteration, one layer over).

`execute_policy()` is the only function in the cascade that performs a
real side effect (logging, saving a frame to disk) — pipeline.py itself
never does. Actions are intentionally minimal, real, and headless (no
Streamlit/UI dependency here): "display" and "alert" are distinguishable
log entries a UI could subscribe to via `on_action`, not literal
rendering — there is no dedicated cascade UI in this build (see
core/cascade/__init__.py). `on_action`, when given, is called once per
action actually performed, mirroring the `on_progress(done, total)`
callback pattern already used for long-running bulk operations elsewhere
in this project (see core/inspections/lifecycle.py) — a generic
observation hook, not a UI dependency baked into this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from emil_ml.config.settings import CASCADE_SAVED_FRAMES_DIR
from emil_ml.core.cascade import policy_store
from emil_ml.core.cascade.policy import (
    ACTION_ALERT,
    ACTION_DISPLAY,
    ACTION_LOG,
    ACTION_SAVE_FRAME,
    DEFAULT_PRIORITY,
    ReactionPolicy,
)

logger = logging.getLogger(__name__)

# Used when no policy row exists yet for a (component_name, specialist,
# identity_key) — e.g. a person was just added to the known-individuals
# database but nobody has configured this component's reaction to them
# yet. Degrading to "log it" rather than raising means the cascade never
# breaks just because a policy hasn't been set up — see module docstring.
_FALLBACK_POLICY_LABEL = "unconfigured"
_FALLBACK_POLICY_MESSAGE_TEMPLATE = "No reaction policy configured for {specialist}:{identity_key} — logged only."


@dataclass(frozen=True)
class PolicyExecutionResult:
    policy: ReactionPolicy
    executed_actions: tuple[str, ...]
    saved_frame_path: Path | None
    log_message: str


def _default_policy(component_name: str, specialist: str, identity_key: str) -> ReactionPolicy:
    return ReactionPolicy(
        component_name=component_name,
        specialist=specialist,
        identity_key=identity_key,
        label=_FALLBACK_POLICY_LABEL,
        message=_FALLBACK_POLICY_MESSAGE_TEMPLATE.format(specialist=specialist, identity_key=identity_key),
        actions=(ACTION_LOG,),
        priority=DEFAULT_PRIORITY,
    )


def execute_policy(
    component_name: str,
    specialist: str,
    identity_key: str,
    *,
    image: Any = None,
    on_action: Callable[[str, ReactionPolicy], None] | None = None,
) -> PolicyExecutionResult:
    """Look up the policy for (component_name, specialist, identity_key)
    and perform its configured actions in order. Falls back to
    `_default_policy()` (log only) if this component hasn't configured a
    reaction for this identity yet — reaction policies are per-component
    (see policy.py's module docstring: the same person can warrant a
    different reaction from a different component), so a policy
    configured on one component never applies to another.

    `image` is only used by the "save_frame" action; omit it (or pass
    None) for a caller that never configures that action — no error, the
    action is simply skipped with a debug log (see below).
    """
    policy = policy_store.get_policy(component_name, specialist, identity_key) or _default_policy(
        component_name, specialist, identity_key
    )

    executed: list[str] = []
    saved_frame_path: Path | None = None
    log_message = f"[{policy.priority}] {policy.label}: {policy.message}"

    for action in policy.actions:
        if action == ACTION_LOG:
            logger.info("cascade reaction (%s:%s) %s", specialist, identity_key, log_message)
        elif action == ACTION_DISPLAY:
            logger.info("cascade reaction (%s:%s) DISPLAY: %s", specialist, identity_key, log_message)
        elif action == ACTION_ALERT:
            logger.warning("cascade reaction (%s:%s) ALERT: %s", specialist, identity_key, log_message)
        elif action == ACTION_SAVE_FRAME:
            if image is None:
                logger.debug(
                    "cascade reaction (%s:%s) save_frame skipped — no image provided", specialist, identity_key
                )
            else:
                saved_frame_path = _save_frame(specialist, identity_key, image)
        else:  # pragma: no cover - policy_store.upsert_policy() already validates against VALID_ACTIONS
            logger.warning("cascade reaction (%s:%s) unknown action %r skipped", specialist, identity_key, action)
            continue

        executed.append(action)
        if on_action is not None:
            on_action(action, policy)

    return PolicyExecutionResult(
        policy=policy, executed_actions=tuple(executed), saved_frame_path=saved_frame_path, log_message=log_message
    )


def _save_frame(specialist: str, identity_key: str, image: Any) -> Path:
    directory = CASCADE_SAVED_FRAMES_DIR / specialist / identity_key
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / f"{timestamp}.png"
    image.save(path)
    return path
