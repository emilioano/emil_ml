"""Standalone folder watcher — a production entry point, not a pipeline.

Watches every registered component's input/ directory and, when a file
appears there and finishes writing, runs it through the exact same
inspection entry point the Streamlit UI uses. Contains no detection/RAG
logic of its own — see service.py's module docstring for the full design.

service.py     WatcherService — the watchdog Observer, the discovery
               queue (events + a periodic poll safety net), the
               stability check, and the worker thread that calls
               core.inspections.orchestrator.run_inspection() per file.
__main__.py    `python -m emil_ml.watcher` — CLI entry point. Runs as its
               own long-lived process, independent of Streamlit.
"""
