"""Phase 1 sanity check: init the DB, create a dummy component, verify folders/CRUD.

Run with: python scripts/verify_foundation.py
"""

from __future__ import annotations

from emil_ml.config.logging_config import configure_logging
from emil_ml.config.registry import ComponentRegistry
from emil_ml.utils.paths import for_component


def main() -> None:
    configure_logging()
    registry = ComponentRegistry()

    name = "widget-42"
    existing = registry.get(name)
    if existing is not None:
        print(f"Found existing component {name!r}, deleting it first.")
        registry.delete(name)

    print("Creating component 'Widget 42'...")
    component = registry.create("Widget 42", image_size=128, epochs=5)
    print(f"  -> {component}")
    assert component.name == "widget-42"
    assert component.status == "created"

    print("Creating folder tree...")
    paths = for_component(component.name)
    paths.create_all()
    for d in paths.all_dirs():
        assert d.exists(), f"missing dir: {d}"
        print(f"  ok: {d}")

    print("Updating settings...")
    registry.update_settings(component.name, epochs=10)
    updated = registry.get(component.name)
    assert updated is not None and updated.epochs == 10
    print(f"  -> epochs now {updated.epochs}")

    print("Simulating a training result...")
    registry.update_training_result(
        component.name,
        anomaly_threshold=0.0123,
        model_path=str(paths.model_path),
        status="ready",
    )
    ready = registry.get(component.name)
    assert ready is not None and ready.status == "ready"
    assert ready.anomaly_threshold == 0.0123
    print(f"  -> status={ready.status}, threshold={ready.anomaly_threshold}")

    print("Listing ready components...")
    ready_list = registry.list_ready()
    assert any(c.name == component.name for c in ready_list)
    print(f"  -> {[c.name for c in ready_list]}")

    print("Listing all components...")
    all_list = registry.list_all()
    print(f"  -> {[c.name for c in all_list]}")

    print("\nPhase 1 foundation OK.")


if __name__ == "__main__":
    main()
