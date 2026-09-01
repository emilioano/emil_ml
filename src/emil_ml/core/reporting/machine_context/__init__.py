"""Production-line machine parameters at inspection time.

source.py fetches readings for a given inspection (a fictional SQLite-
backed source for this POC, behind a thin interface — see
MachineContextSource — so it can later point at real equipment without
touching anything else). analyzer.py compares readings against a
component's own normal ranges (config/registry.py) to surface anomalies
like "12°C over normal" -> the searchable state "over-temperature".

This is the second context source a generated report draws on, alongside
the detection result itself — decisive when the detector's own output has
no defect class (PatchCore, the autoencoder): a machine-context anomaly
may be the only concrete lead available.
"""
