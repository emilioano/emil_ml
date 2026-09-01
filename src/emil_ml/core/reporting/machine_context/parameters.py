"""Per-component machine-parameter definitions: which parameters a
component's machine context tracks, their normal ranges, and how a
deviation in each direction is phrased as a searchable state.

Stored as JSON in the component registry (Component.machine_parameters),
not fixed table columns — a toothbrush line cares about temperature and
vibration; an optical inspection cares about brightness and exposure.
Which parameters exist, and how their anomalies are worded, is data, not
schema. This module has no dependencies on source.py or analyzer.py
(deliberately: both of those depend on this one, not on each other,
keeping the dependency graph one-directional) — see analyzer.py, which is
parameter-agnostic: it never references a parameter by name, only by
iterating whatever a component's own definitions say.

The vocabulary chosen for above_state/below_state is what ties this to
the knowledge base: retriever.py finds documentation by similarity search
over these exact words, so a state term only pays off if it shares
language with how the relevant document is worded (see
data/components/tandborste/knowledge/machine_parameters.md for the
toothbrush line's own vocabulary).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MachineParameterDef:
    """One parameter a component's machine context tracks."""

    name: str  # matches a MachineReading.values key
    unit: str
    normal_min: float
    normal_max: float
    above_state: str  # searchable term when the reading is above normal_max
    below_state: str | None = None  # searchable term when below normal_min; None if that direction isn't meaningful


def parse_machine_parameters(raw_json: str) -> list[MachineParameterDef]:
    """Parse a component's Component.machine_parameters JSON into typed definitions.

    Empty/missing JSON parses to an empty list — a component with no
    parameter definitions simply has no machine context available, not
    an error (see analyzer.analyze(), which handles this the same way it
    handles a missing reading).
    """
    if not raw_json:
        return []
    return [MachineParameterDef(**item) for item in json.loads(raw_json)]


def serialize_machine_parameters(defs: list[MachineParameterDef]) -> str:
    """Inverse of parse_machine_parameters() — for writing via registry.update_settings()."""
    return json.dumps([asdict(d) for d in defs])
