"""Translates raw machine readings into diagnostically interesting anomalies.

The central point of this module: parameters are numbers, but numbers
embed poorly and aren't interesting on their own — what's interesting is
the *deviation* from what's normal for this specific component (see
parameters.py's per-component MachineParameterDef; a temperature that's
normal for one machine can be high for another). So this module does two
things with a raw reading:

1. Produces a structured `ParameterAnomaly` per out-of-range parameter —
   value, normal range, direction, and magnitude — precise enough for
   Fas 4's prompt to reason about exactly.
2. Translates each anomaly into a normalized, searchable "state" term
   (e.g. "over-temperature") using that parameter's own above_state/
   below_state wording — never a hardcoded mapping in this file. Adding a
   new parameter to a component (or a whole new component with an
   entirely different parameter set) requires zero changes here: this
   module never references a parameter by name, it only iterates
   whatever `parameters.parse_machine_parameters(component.machine_parameters)`
   returns.
"""

from __future__ import annotations

from dataclasses import dataclass

from emil_ml.config.registry import Component
from emil_ml.core.reporting.machine_context.parameters import parse_machine_parameters
from emil_ml.core.reporting.machine_context.source import MachineReading


@dataclass(frozen=True)
class ParameterAnomaly:
    """One out-of-range parameter, with enough structure for a prompt to reason about exactly."""

    parameter: str  # the MachineParameterDef.name / MachineReading.values key
    value: float
    normal_min: float
    normal_max: float
    direction: str  # "above" | "below"
    magnitude: float  # how far past the relevant bound, always >= 0
    unit: str
    state: str  # normalized searchable term, e.g. "over-temperature"

    def describe(self) -> str:
        bound = self.normal_max if self.direction == "above" else self.normal_min
        return (
            f"{self.parameter}={self.value:g}{self.unit} is {self.direction} the normal range "
            f"({self.normal_min:g}-{self.normal_max:g}{self.unit}), {self.magnitude:g}{self.unit} "
            f"past the {'max' if self.direction == 'above' else 'min'} ({bound:g}{self.unit})"
        )


@dataclass(frozen=True)
class MachineContext:
    """Everything downstream (retriever.py, prompt.py) needs from machine context for one inspection."""

    component_name: str
    timestamp: str | None
    anomalies: list[ParameterAnomaly]
    searchable_states: list[str]  # deduplicated, same order as anomalies
    reading: MachineReading | None  # full raw reading, kept for reference/debugging

    @property
    def has_anomalies(self) -> bool:
        return len(self.anomalies) > 0


def analyze(reading: MachineReading | None, component: Component) -> MachineContext:
    """Compare a reading against `component`'s own parameter definitions and extract anomalies.

    `reading=None` (no machine context available for this inspection)
    produces an empty MachineContext, the same shape as "reading present,
    nothing anomalous" — callers don't need to special-case "no data" vs.
    "data, all normal". Likewise, a component with no parameter
    definitions at all (empty machine_parameters) always produces zero
    anomalies, regardless of what a reading contains — there's nothing to
    compare it against.
    """
    if reading is None:
        return MachineContext(
            component_name=component.name, timestamp=None, anomalies=[], searchable_states=[], reading=None
        )

    anomalies: list[ParameterAnomaly] = []
    for param in parse_machine_parameters(component.machine_parameters):
        if param.name not in reading.values:
            continue  # this reading doesn't cover this parameter — nothing to compare, not an error
        value = reading.values[param.name]

        if value > param.normal_max:
            anomalies.append(
                ParameterAnomaly(
                    parameter=param.name,
                    value=value,
                    normal_min=param.normal_min,
                    normal_max=param.normal_max,
                    direction="above",
                    magnitude=value - param.normal_max,
                    unit=param.unit,
                    state=param.above_state,
                )
            )
        elif value < param.normal_min and param.below_state is not None:
            anomalies.append(
                ParameterAnomaly(
                    parameter=param.name,
                    value=value,
                    normal_min=param.normal_min,
                    normal_max=param.normal_max,
                    direction="below",
                    magnitude=param.normal_min - value,
                    unit=param.unit,
                    state=param.below_state,
                )
            )

    searchable_states = list(dict.fromkeys(a.state for a in anomalies))  # dedup, preserve order
    return MachineContext(
        component_name=component.name,
        timestamp=reading.timestamp,
        anomalies=anomalies,
        searchable_states=searchable_states,
        reading=reading,
    )
