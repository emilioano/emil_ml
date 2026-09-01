"""Fills machine_readings with synthetic, plausible history for one or more
components — general demo/history data, distinct from
scripts/verify_machine_context.py's controlled, deterministic scenarios.

Run with: python scripts/seed_machine_readings.py [component_name ...]
(defaults to every registered component if none are named)
"""

from __future__ import annotations

import sys

from emil_ml.config.registry import ComponentRegistry
from emil_ml.core.reporting.machine_context.source import SqliteMachineContextSource


def main() -> None:
    registry = ComponentRegistry()
    names = sys.argv[1:] or [c.name for c in registry.list_all()]

    source = SqliteMachineContextSource()
    for name in names:
        component = registry.get(name)
        if component is None:
            print(f"Skipping {name!r}: no such component.")
            continue
        readings = source.seed_synthetic_readings(name, component=component)
        print(f"{name}: seeded {len(readings)} synthetic reading(s), "
              f"{readings[0].timestamp} .. {readings[-1].timestamp}")


if __name__ == "__main__":
    main()
