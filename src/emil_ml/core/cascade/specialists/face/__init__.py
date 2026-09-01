"""Embedding-based face recognition — the cascade's first specialist,
triggered when the coarse stage's category is "human" (see
core/cascade/specialist_registry.py).

Two stages, both via facenet-pytorch (chosen over dlib/face_recognition
specifically to avoid a C++ compile step — facenet-pytorch is pure
Python + torch, and torch is already a project dependency via
ultralytics/YOLO; confirmed installing and running cleanly, including in
WSL, with no extra system packages):

1. MTCNN detects and crops/aligns the face in the frame.
2. InceptionResnetV1 (vggface2 pretrained weights) embeds the aligned
   face into a 512-dim vector.

Matching (predictor.py) is a nearest-neighbor lookup by L2 distance
against store.py's known-individuals table — NOT classification. Adding
a person is `store.add_known_individual(name, embedding, consented=True)`,
never retraining a model.

PRIVACY BY CONSTRUCTION — see store.py's own module docstring for the
full reasoning: the known-individuals table is the only source of
identity, consent is a required (non-defaulted) parameter to add a row,
and anyone whose embedding doesn't match a consenting row within
threshold is always "unknown", never partially identified.
"""
