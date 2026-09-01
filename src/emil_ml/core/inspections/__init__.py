"""Inspection persistence and lifecycle — the durable record of every
inspection, independent of whether reporting was even enabled for its
component.

store.py       The `inspections` table: the queryable source of truth
               (verdict, score, defect classes, report text + provenance,
               lifecycle status). Self-contained schema management, same
               pattern as core/reporting/machine_context/source.py's
               machine_readings table.
lifecycle.py   File-side operations: writing the human-readable .report.md
               next to an analyzed image, and moving files through
               input/ -> analyzed/{approved|failed}/ -> archive/ as an
               inspection is acknowledged and archived. The database is
               authoritative — a .report.md can always be regenerated
               from a stored InspectionRecord; the reverse never happens.
orchestrator.py Ties pipeline.inspect() together with the two modules
               above: runs detection (fast), persists the image + a
               'new' record immediately, and — only if this component's
               reporting settings call for one — generates the report in
               a background thread so a slow LLM call never blocks the
               caller on the verdict it already has.
"""
