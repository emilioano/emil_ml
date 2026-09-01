"""Production-line machine parameters at inspection time.

`MachineContextSource` is the interface everything else in this package
talks to; `SqliteMachineContextSource` is a fictional POC implementation
backed by a table in the same emil.db every other per-component data
lives in. Swapping in a real source later (OPC-UA, a REST endpoint against
actual equipment, ...) means writing one new class against this same
interface — nothing in analyzer.py, retriever.py, or reporter.py needs to
change or even know the difference.

Readings are stored key-value — (component_name, timestamp,
parameter_name, value) rows — not fixed columns, so the same table holds
a toothbrush line's {temperature, vibration} and an optical inspection's
{brightness, exposure} without a schema change; which parameters exist is
defined per-component in the registry (see parameters.py), not here. One
"reading" for an inspection is just every row sharing that component's
most recent timestamp.

Parameters themselves aren't the interesting part — see analyzer.py for
why raw numbers get translated into anomalies before anything downstream
uses them.
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from emil_ml.config.database import connect, get_connection
from emil_ml.config.registry import Component
from emil_ml.config.settings import DB_PATH, MACHINE_READINGS_TABLE
from emil_ml.core.reporting.machine_context.parameters import parse_machine_parameters

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MachineReading:
    """Raw machine parameters for one component at one point in time.

    `values` maps parameter name -> reading, covering whatever parameters
    that component happens to have been recorded for at this timestamp —
    not a fixed set of fields.
    """

    component_name: str
    timestamp: str  # ISO 8601
    values: dict[str, float]


class MachineContextSource(ABC):
    """Fetches machine readings for a given inspection.

    Everything downstream (analyzer.py, reporter.py) depends on this
    interface, never on SQLite or any other storage detail directly.
    """

    @abstractmethod
    def get_readings(self, component_name: str, *, timestamp: str | None = None) -> MachineReading | None:
        """The reading at or before `timestamp` (default: the most recent known reading).

        Returns None if no reading exists for this component at all —
        analyzer.py treats that as "no machine context available", not
        an error.
        """
        ...


_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {MACHINE_READINGS_TABLE} (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    component_name   TEXT NOT NULL,
    timestamp        TEXT NOT NULL,
    parameter_name   TEXT NOT NULL,
    value            REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_{MACHINE_READINGS_TABLE}_component_timestamp
    ON {MACHINE_READINGS_TABLE} (component_name, timestamp);
"""


class SqliteMachineContextSource(MachineContextSource):
    """POC source: a SQLite table seeded with synthetic data, with a
    deliberate injection point (insert_reading()) for demoing how the
    same image can get a different report depending on machine context —
    just pass a parameter value outside the component's normal range.
    """

    def __init__(self, db_path: Path | str = DB_PATH) -> None:
        self.db_path = db_path
        with connect(self.db_path) as conn:
            self._ensure_schema(conn)

    @staticmethod
    def _ensure_schema(conn) -> None:  # noqa: ANN001 - sqlite3.Connection
        existing_columns = {
            row["name"] for row in conn.execute(f"PRAGMA table_info({MACHINE_READINGS_TABLE})").fetchall()
        }
        if existing_columns and "parameter_name" not in existing_columns:
            # Old wide-column shape (one column per parameter) from an
            # earlier design — incompatible, and this table only ever
            # holds disposable synthetic/demo data, so drop and recreate
            # rather than attempt a cell-by-cell migration.
            conn.execute(f"DROP TABLE IF EXISTS {MACHINE_READINGS_TABLE}")
        conn.executescript(_SCHEMA)

    def insert_reading(
        self,
        component_name: str,
        values: dict[str, float],
        *,
        timestamp: str | None = None,
    ) -> MachineReading:
        """Insert one reading (one row per parameter, all sharing one timestamp).

        The anomaly injection point: pass any parameter value outside
        this component's normal range for that parameter (see
        core/reporting/machine_context/parameters.py's MachineParameterDef,
        stored per-component in the registry) to simulate a machine-
        context anomaly at this inspection.
        """
        timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        with connect(self.db_path) as conn:
            conn.executemany(
                f"INSERT INTO {MACHINE_READINGS_TABLE} (component_name, timestamp, parameter_name, value) "
                "VALUES (?, ?, ?, ?)",
                [(component_name, timestamp, name, value) for name, value in values.items()],
            )
        return MachineReading(component_name=component_name, timestamp=timestamp, values=dict(values))

    def get_readings(self, component_name: str, *, timestamp: str | None = None) -> MachineReading | None:
        conn = get_connection(self.db_path)
        try:
            self._ensure_schema(conn)
            if timestamp is None:
                ts_row = conn.execute(
                    f"SELECT MAX(timestamp) AS ts FROM {MACHINE_READINGS_TABLE} WHERE component_name = ?",
                    (component_name,),
                ).fetchone()
            else:
                ts_row = conn.execute(
                    f"SELECT MAX(timestamp) AS ts FROM {MACHINE_READINGS_TABLE} "
                    "WHERE component_name = ? AND timestamp <= ?",
                    (component_name, timestamp),
                ).fetchone()
            if ts_row is None or ts_row["ts"] is None:
                logger.warning("component=%s no machine reading available", component_name)
                return None
            actual_timestamp = ts_row["ts"]

            rows = conn.execute(
                f"SELECT parameter_name, value FROM {MACHINE_READINGS_TABLE} "
                "WHERE component_name = ? AND timestamp = ?",
                (component_name, actual_timestamp),
            ).fetchall()
        finally:
            conn.close()

        values = {row["parameter_name"]: row["value"] for row in rows}
        logger.debug("component=%s picked reading at %s: %s", component_name, actual_timestamp, values)
        return MachineReading(component_name=component_name, timestamp=actual_timestamp, values=values)

    def count_readings(self, component_name: str) -> int:
        """How many reading rows (across every timestamp/parameter) this
        component has — a read-only count for core/component_deletion.py's
        deletion-impact summary."""
        conn = get_connection(self.db_path)
        try:
            self._ensure_schema(conn)
            row = conn.execute(
                f"SELECT COUNT(*) AS n FROM {MACHINE_READINGS_TABLE} WHERE component_name = ?",
                (component_name,),
            ).fetchone()
        finally:
            conn.close()
        return row["n"] if row else 0

    def delete_readings(self, component_name: str) -> int:
        """Permanently remove every machine reading for this component —
        used exclusively by core/component_deletion.py's
        permanently_delete_component(), never as a normal workflow
        action. Idempotent: deleting 0 remaining rows (e.g. a resumed,
        already-completed deletion) is a safe no-op. Returns how many
        rows were removed.
        """
        with connect(self.db_path) as conn:
            self._ensure_schema(conn)
            cursor = conn.execute(
                f"DELETE FROM {MACHINE_READINGS_TABLE} WHERE component_name = ?", (component_name,)
            )
            return cursor.rowcount

    def seed_synthetic_readings(
        self,
        component_name: str,
        *,
        component: Component,
        n_readings: int = 30,
        interval_hours: float = 8.0,
        rng: random.Random | None = None,
    ) -> list[MachineReading]:
        """Bulk-insert plausible, mostly-normal history for demo/testing purposes.

        Reads `component`'s own parameter definitions (parameters.py) —
        whatever parameters it defines, seeded, no fixed set assumed.
        Centers each at its normal-range midpoint with modest Gaussian
        noise (sigma = range width / 6, so ~99.7% of synthetic readings
        land inside the normal range — occasional natural outliers are
        intentional, real sensor noise looks like that too). For
        deterministic, guaranteed-in/out-of-range readings, use
        insert_reading() directly instead — that's what
        scripts/verify_machine_context.py does for its controlled
        scenarios.
        """
        rng = rng or random.Random()
        param_defs = parse_machine_parameters(component.machine_parameters)
        readings: list[MachineReading] = []
        now = datetime.now(timezone.utc)

        def _sample(min_val: float, max_val: float) -> float:
            mid = (min_val + max_val) / 2
            sigma = max((max_val - min_val) / 6, 1e-6)
            return max(rng.gauss(mid, sigma), 0.0)

        for i in range(n_readings):
            ts = (now - timedelta(hours=interval_hours * (n_readings - i))).isoformat()
            values = {p.name: _sample(p.normal_min, p.normal_max) for p in param_defs}
            readings.append(self.insert_reading(component_name, values, timestamp=ts))
        return readings
