"""Modality handlers: how to read and preprocess raw input for a component.

Orthogonal to `model_type` (see `core/base.py`). A modality handler only
loads/preprocesses input — it never contains detection logic. It hands a
prepared object to the predictor, which is where analysis actually happens.
"""
