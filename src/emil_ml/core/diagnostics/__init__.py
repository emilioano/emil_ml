"""Diagnostic tools for understanding a component's data before trusting a model.

Not a model_type: nothing here trains or predicts. It extracts CNN
embeddings from a component's training images, projects them to 2D, and
visualizes how well approved/failed images separate — a quick read on
whether an inspection problem is even solvable with whole-image features
before committing to training and tuning a real model. See
core/diagnostics/embeddings.py and core/diagnostics/projection.py.
"""
