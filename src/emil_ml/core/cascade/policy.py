"""Reaction policy: structured, per-component-per-identity configuration
for what happens once the cascade has identified something — the
cascade's analog to core/reporting/machine_context/parameters.py's
MachineParameterDef. Policy is DATA (one row per (component_name,
specialist, identity_key), see policy_store.py); policy_executor.py's
execute_policy() is identity-agnostic, exactly like machine_context/
analyzer.py never references a specific parameter's name — it only
iterates whatever a component's own parameter definitions say. Here,
execute_policy() never references a specific person (or, for a future
specialist, a specific car model) by name; all of that knowledge lives in
the policy table.

Scoped per component, not shared across every cascade component the way
the known-individuals identity registry is (see
core/cascade/specialists/face/store.py's module docstring on why THAT is
deliberately global): recognizing Alice is the same fact everywhere, but
how a component should REACT to recognizing her is a property of that
component's own context (e.g. a front-door camera welcomes her, a
restricted-area camera alerts security) — without this, every cascade
component configured with the same specialist would necessarily behave
identically, which defeats having more than one.

"unknown" is a first-class `identity_key` with its own policy row — every
specialist's SpecialistResult.identity_key is "unknown" for a non-match
(see core/cascade/base.py), so the lookup in policy_store.py is uniform
for known and unknown identities alike; there is no separate
"unknown-handling" code path.
"""

from __future__ import annotations

from dataclasses import dataclass

# A small, fixed action vocabulary — deliberately not a free-text field,
# so policy_executor.py can implement each action exactly once and a typo
# in a policy row is caught (see policy_store.py's validation) rather than
# silently doing nothing.
ACTION_LOG = "log"
ACTION_DISPLAY = "display"
ACTION_ALERT = "alert"
ACTION_SAVE_FRAME = "save_frame"
VALID_ACTIONS = (ACTION_LOG, ACTION_DISPLAY, ACTION_ALERT, ACTION_SAVE_FRAME)

VALID_PRIORITIES = ("low", "normal", "high")
DEFAULT_PRIORITY = "normal"


@dataclass(frozen=True)
class ReactionPolicy:
    """One (component, identity)'s configured reaction — a row from policy_store.py."""

    component_name: str  # which cascade component this reaction applies to
    specialist: str  # e.g. "face" — namespaces identity_key, see module docstring
    identity_key: str
    label: str  # e.g. "approved person", "welcome guest", "unknown"
    message: str
    actions: tuple[str, ...]  # subset of VALID_ACTIONS
    priority: str = DEFAULT_PRIORITY
