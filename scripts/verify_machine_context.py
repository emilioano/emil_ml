"""Fas 3 sanity check: raw reading -> anomaly -> searchable state ->
retrieved documentation, end to end, with no LLM involved.

Deliberately exercises TWO components with completely different parameter
sets (tandborste: temperature/speed/pressure/vibration/hours_since_service;
optical-sensor: brightness/exposure) through the exact same analyzer.analyze()
call, to prove the analyzer is genuinely parameter-agnostic — it never
branches on which component or parameter it's looking at.

Uses insert_reading() directly (not the random seeder) for fully
controlled, deterministic scenarios.
Run with: python scripts/verify_machine_context.py
"""

from __future__ import annotations

import sys

# Windows terminals default to a non-UTF-8 codepage that mangles non-ASCII
# characters (°, em dashes, ...) on print() otherwise — the underlying
# stored text is unaffected either way, this only fixes how this script
# displays it.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from emil_ml.config.logging_config import configure_logging
from emil_ml.config.registry import ComponentRegistry
from emil_ml.core.reporting.knowledge import indexer, retriever
from emil_ml.core.reporting.machine_context import analyzer
from emil_ml.core.reporting.machine_context.parameters import MachineParameterDef, serialize_machine_parameters
from emil_ml.core.reporting.machine_context.source import SqliteMachineContextSource

TANDBORSTE_PARAMS = [
    MachineParameterDef("temperature", "°C", 60.0, 75.0, "over-temperature", "low temperature"),
    MachineParameterDef("speed", " units/min", 45.0, 55.0, "high speed", "low speed"),
    MachineParameterDef("pressure", " bar", 80.0, 120.0, "elevated pressure", "low pressure"),
    MachineParameterDef("vibration", " mm/s", 0.0, 4.5, "elevated vibration", None),
    MachineParameterDef("hours_since_service", "h", 0.0, 500.0, "overdue service", None),
]

OPTICAL_SENSOR_PARAMS = [
    MachineParameterDef("brightness", " lux", 300.0, 800.0, "excessive brightness", "insufficient brightness"),
    MachineParameterDef("exposure", " ms", 5.0, 20.0, "excessive exposure", "insufficient exposure"),
]


def _print_context(ctx) -> None:  # noqa: ANN001 - analyzer.MachineContext, kept loose for script brevity
    if not ctx.anomalies:
        print("  No anomalies. Searchable states: []")
        return
    print(f"  Searchable states: {ctx.searchable_states}")
    for a in ctx.anomalies:
        print(f"    - {a.describe()}  (state={a.state!r})")


def _print_retrieval(results) -> None:  # noqa: ANN001 - list[retriever.RetrievedChunk]
    if not results:
        print("    (no chunks retrieved)")
        return
    for r in results[:3]:
        print(f"    sim={r.similarity:.3f}  [{r.doc_type}] {r.source} / {r.section!r}")


def _setup_components(registry: ComponentRegistry) -> None:
    """Idempotent: configures parameter definitions for both demo components
    and (re-)indexes optical-sensor's knowledge doc, safe to re-run."""
    registry.update_settings("tandborste", machine_parameters=serialize_machine_parameters(TANDBORSTE_PARAMS))

    if registry.get("optical-sensor") is None:
        registry.create("Optical Sensor", modality="image", model_type="autoencoder")
    registry.update_settings(
        "optical-sensor", machine_parameters=serialize_machine_parameters(OPTICAL_SENSOR_PARAMS)
    )
    indexer.index_component_type("optical-sensor")


def _midpoint_reading(param_defs: list[MachineParameterDef]) -> dict[str, float]:
    return {p.name: (p.normal_min + p.normal_max) / 2 for p in param_defs}


def main() -> None:
    configure_logging()
    registry = ComponentRegistry()
    print("Setting up tandborste + optical-sensor parameter definitions and indexing...")
    _setup_components(registry)
    print()

    source = SqliteMachineContextSource()

    # =====================================================================
    # tandborste: temperature / speed / pressure / vibration / hours_since_service
    # =====================================================================
    tandborste = registry.get("tandborste")
    print(f"=== tandborste — parameters: {[p.name for p in TANDBORSTE_PARAMS]} ===")
    print()

    print("--- Scenario 1: normal parameters (all at range midpoint) ---")
    reading = source.insert_reading("tandborste", _midpoint_reading(TANDBORSTE_PARAMS))
    ctx = analyzer.analyze(reading, tandborste)
    _print_context(ctx)
    print(f"  -> {'PASS' if not ctx.has_anomalies else 'FAIL'}: no anomalies, empty state list.")
    print()

    print("--- Scenario 2: injected anomaly — temperature well above normal max ---")
    values = _midpoint_reading(TANDBORSTE_PARAMS)
    values["temperature"] = 88.0  # normal max is 75.0
    reading = source.insert_reading("tandborste", values)
    ctx = analyzer.analyze(reading, tandborste)
    _print_context(ctx)
    ok = ctx.searchable_states == ["over-temperature"]
    print(f"  -> {'PASS' if ok else 'FAIL'}: expected exactly ['over-temperature'].")
    print()

    print("  Retrieval using this state as machine context:")
    query = retriever.build_query_text("tandborste", machine_states=ctx.searchable_states)
    print(f"    query: {query!r}")
    results = retriever.retrieve("tandborste", query)
    _print_retrieval(results)
    found = bool(results) and results[0].section == "Over-temperature"
    print(f"  -> {'PASS' if found else 'FAIL'}: top result is the 'Over-temperature' section.")
    print()

    print("--- Scenario 3: multiple simultaneous anomalies (temperature + vibration + overdue service) ---")
    values = _midpoint_reading(TANDBORSTE_PARAMS)
    values["temperature"] = 88.0
    values["vibration"] = 7.5  # normal max is 4.5
    values["hours_since_service"] = 700.0  # normal max is 500.0
    reading = source.insert_reading("tandborste", values)
    ctx = analyzer.analyze(reading, tandborste)
    _print_context(ctx)
    expected = {"over-temperature", "elevated vibration", "overdue service"}
    ok = set(ctx.searchable_states) == expected and len(ctx.anomalies) == 3
    print(f"  -> {'PASS' if ok else 'FAIL'}: expected exactly {sorted(expected)}.")
    print()

    # =====================================================================
    # optical-sensor: brightness / exposure — a completely different
    # parameter set, run through the exact same analyzer.analyze().
    # =====================================================================
    optical = registry.get("optical-sensor")
    print(f"=== optical-sensor — parameters: {[p.name for p in OPTICAL_SENSOR_PARAMS]} ===")
    print("(same analyzer.analyze() call as above — zero code branching on component or parameter)")
    print()

    print("--- Scenario 4: normal parameters (all at range midpoint) ---")
    reading = source.insert_reading("optical-sensor", _midpoint_reading(OPTICAL_SENSOR_PARAMS))
    ctx = analyzer.analyze(reading, optical)
    _print_context(ctx)
    print(f"  -> {'PASS' if not ctx.has_anomalies else 'FAIL'}: no anomalies, empty state list.")
    print()

    print("--- Scenario 5: injected anomaly — brightness well below normal min ---")
    values = _midpoint_reading(OPTICAL_SENSOR_PARAMS)
    values["brightness"] = 150.0  # normal min is 300.0
    reading = source.insert_reading("optical-sensor", values)
    ctx = analyzer.analyze(reading, optical)
    _print_context(ctx)
    ok = ctx.searchable_states == ["insufficient brightness"]
    print(f"  -> {'PASS' if ok else 'FAIL'}: expected exactly ['insufficient brightness'].")
    print()

    print("  Retrieval using this state as machine context:")
    query = retriever.build_query_text("optical-sensor", machine_states=ctx.searchable_states)
    print(f"    query: {query!r}")
    results = retriever.retrieve("optical-sensor", query)
    _print_retrieval(results)
    found = bool(results) and results[0].section == "Insufficient brightness"
    print(f"  -> {'PASS' if found else 'FAIL'}: top result is the 'Insufficient brightness' section.")
    print()

    print("--- Scenario 6: multiple simultaneous anomalies (excessive exposure + insufficient brightness) ---")
    values = _midpoint_reading(OPTICAL_SENSOR_PARAMS)
    values["brightness"] = 150.0
    values["exposure"] = 35.0  # normal max is 20.0
    reading = source.insert_reading("optical-sensor", values)
    ctx = analyzer.analyze(reading, optical)
    _print_context(ctx)
    expected = {"insufficient brightness", "excessive exposure"}
    ok = set(ctx.searchable_states) == expected and len(ctx.anomalies) == 2
    print(f"  -> {'PASS' if ok else 'FAIL'}: expected exactly {sorted(expected)}.")
    print()

    print("Fas 3 machine context OK — same analyzer, two unrelated parameter sets, no special-casing.")


if __name__ == "__main__":
    main()
