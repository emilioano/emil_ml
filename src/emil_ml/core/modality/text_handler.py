"""Extension point: text/log modality.

Not yet implemented. Would turn a log line, text block, or file path into
whatever a text-based predictor expects (e.g. an embedding or token tensor),
mirroring `ImageModalityHandler`'s role for the image modality. Register
`TextModalityHandler(BaseModalityHandler)` here, then add it to the modality
map in `core/registry_factory.py`, alongside any ("text", model_type)
trainer/predictor combinations.
"""
