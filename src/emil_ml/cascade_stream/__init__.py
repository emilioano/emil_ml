"""Standalone process for continuous cascade operation against a Kafka
topic — see service.py's module docstring for the full design. Sibling to
emil_ml.watcher, same reasoning: a Streamlit session restarts on every code
change/rerun, which would kill an in-process consumer, so this runs
independently (`python -m emil_ml.cascade_stream --component X`), started
and stopped from a terminal, never launched by the Streamlit app itself.
"""
