# EMIL Lab — Enhanced Machine Inspection & Learning Lab

A modular, Streamlit-based platform for visual inspection: onboard
a component type, choose from seven different machine-learning methods to
detect anomalies or defects, train it on your own images, and inspect new
input — one upload at a time, from a watched folder, from an uploaded video,
or from a live Kafka stream. A separate object- and face-recognition cascade
runs alongside the anomaly-detection side, with consent-gated identity
registration and configurable reaction policies. Every inspection can
optionally get an AI-generated explanatory report, grounded in your own
uploaded documentation and live machine-context data via local
retrieval-augmented generation (RAG) — never invented from nothing.

Nothing here talks to an external cloud service by default: models run
locally (TensorFlow/PyTorch), the report-writing LLM runs locally via
[Ollama](https://ollama.com/), and the knowledge base is a local ChromaDB
store. Everything is stored in one SQLite database (`emil.db`) plus a plain
folder tree per component (`data/components/<name>/`).

## Table of contents

- [What it does](#what-it-does)
- [The seven analysis methods](#the-seven-analysis-methods)
- [Object & face recognition cascade](#object--face-recognition-cascade)
- [Live cascade streaming](#live-cascade-streaming-kafka--video--still-image)
- [AI-generated reports (RAG)](#ai-generated-reports-rag)
- [Inspection lifecycle & operator review](#inspection-lifecycle--operator-review)
- [Automated ingestion: the folder watcher](#automated-ingestion-the-folder-watcher)
- [Other tools](#other-tools-diagnostics-grid-search-component-lifecycle)
- [Pages at a glance](#pages-at-a-glance)
- [Architecture](#architecture)
- [Setup](#setup)
- [Running the app](#running-the-app)
- [Verification scripts](#verification-scripts)
- [GPU training (WSL2)](#gpu-training-wsl2)
- [Libraries & dependencies](#libraries--dependencies)

## What it does

At its core, EMIL Lab answers one question per image: **is this approved or
failed** (or, for the cascade, **what/who is in this frame**)? Around that
core, it provides everything a real inspection workflow needs beyond a bare
model:

- **Seven interchangeable ML methods** for a component — unsupervised
  (autoencoder, PatchCore, Isolation Forest), supervised (classifier, YOLO
  object detection), and two frozen/zero-training coarse classifiers used by
  the cascade.
- **A configurable object/face-recognition cascade**, independent of the
  approved/failed anomaly-detection side: detect people, animals, vehicles
  and other objects in a frame, recognize consenting registered individuals
  by face, and react per a configurable policy.
- **Continuous cascade operation** against a live Kafka topic, an uploaded
  video file, or a single still image — not just one-shot testing.
- **AI-generated inspection reports**, grounded in your own uploaded
  knowledge-base documents and live machine sensor readings, via a local
  LLM (Ollama) and local vector search (ChromaDB) — retrieval-augmented, so
  the model only ever writes about what was actually retrieved.
- **A full inspection review workflow**: acknowledge/archive lifecycle,
  bulk actions, human-verified corrections that feed back into retraining,
  and time-based retention cleanup.
- **Automated, unattended ingestion** via a folder watcher — drop a file in
  a component's `input/` folder and it gets inspected automatically, no UI
  needed.
- **Pre-training diagnostics** (UMAP/PCA class-separability visualization)
  to sanity-check whether a dataset is even separable before spending time
  training.
- **Grid search** over hyperparameters, run against disposable scratch
  copies of a component so the live model is never at risk.
- **Full component lifecycle management** — soft-delete/trash with
  restore, and permanent deletion with an itemized impact preview covering
  every subsystem a component touches.

## The seven analysis methods

Every component picks exactly one `model_type` at creation time. All are
`modality="image"` today (a `"text"` modality is recognized in the code but
not yet implemented).

| Method | What it does | Backing library / model | Training data |
|---|---|---|---|
| **Autoencoder** | Unsupervised reconstruction-error anomaly detection — trains only on approved images, flags anything it reconstructs badly. Two score modes: whole-image average error, or worst-single-pixel error (for small localized defects). | TensorFlow/Keras — a custom symmetric convolutional autoencoder (4 strided Conv2D encoder blocks → dense latent bottleneck → 4 Conv2DTranspose decoder blocks). | Approved only (failed images optional, used only to validate the threshold). |
| **Classifier** | Supervised binary CNN classifier, approved vs. failed, via transfer learning. Two-phase training (frozen-base head, then fine-tuned top layers), class weighting, light augmentation. | TensorFlow/Keras — frozen ImageNet-pretrained backbone (MobileNetV2 or EfficientNetB0) + a small dense head. | **Requires both** approved and failed examples — the only method that does. |
| **YOLO (object detection)** | Localizes and draws a box around a specific object/defect instead of judging the whole image. "Presence" (found = failed) or "absence" (missing = failed) decision rules. | PyTorch via **Ultralytics** — fine-tuned from a pretrained YOLO11 checkpoint (`yolo11n.pt` by default). | Approved images with bounding-box annotations (3 ways to provide them — see below). |
| **PatchCore** | Unsupervised, patch-level anomaly detection via a memory bank of normal-patch embeddings and nearest-neighbor scoring — strong on small, localized defects a whole-image method would miss. Produces a heatmap, no annotation needed. | PyTorch via **anomalib** (Intel), `wide_resnet50_2` backbone by default, run through anomalib's `Engine`. | Approved only (failed images optional, used only for threshold/AUROC). |
| **Isolation Forest** | Classic isolation-forest anomaly detection applied to CNN embeddings rather than raw pixels. Fast to train (seconds, no epochs). | scikit-learn `IsolationForest` on top of a frozen MobileNetV2 feature extractor. | Approved only (failed embeddings optional, for validation). |
| **ResNet coarse classifier** | Frozen, zero-training, generic ImageNet-1k classification — repurposes the verdict to carry a coarse category (e.g. "animal", "vehicle", "uncertain") rather than approved/failed. One of two coarse-detector options for the cascade. | TensorFlow/Keras — frozen `ResNet50` (ImageNet weights), top-5 predictions mapped to coarse categories. | None — no training performed at all. |
| **Object & face cascade (COCO detector)** | Frozen, zero-training, multi-object COCO detector — the cascade's coarse stage. Detects *every* object in a frame (not just a single top-1 label) with boxes, mapped to the same coarse-category vocabulary. Auto-ready the instant it's created; drives the whole cascade described below. | PyTorch via Ultralytics — stock, frozen COCO-pretrained YOLO checkpoint. | None — no training performed at all. |

A few things every method shares: settings (image size, epochs, batch size,
and a large set of method-specific "Advanced settings") are stored per
component and editable later without re-uploading training data; an
optional evaluation report (confusion matrices, ROC/PR curves, score
histograms, loss curves — via matplotlib) is generated after every
successful training run; and every trained/frozen predictor is reachable
through the exact same `pipeline.inspect()` entry point regardless of which
method backs it, so the UI never branches on model type.

**Why TensorFlow and PyTorch are never imported into the same process**: on
at least one real GPU/driver combination, TensorFlow and PyTorch's Triton
JIT compiler have been observed to segfault when both frameworks are loaded
together. Every trainer/predictor's framework import is therefore deferred
inside its own factory function (`core/registry_factory.py`), not at module
top level — a session that only ever uses YOLO-family components never
loads TensorFlow at all, and vice versa.

## Object & face recognition cascade

Independent of the approved/failed anomaly-detection side, a **4-step,
fully data-driven pipeline** runs against an "Object & face cascade"
component:

1. **Coarse detection** — the COCO detector (or the ResNet coarse
   classifier) finds every object in the frame with a box, confidence, and
   COCO class, mapped to a shared coarse-category vocabulary (`human`,
   `animal`, `vehicle`, `other`, `uncertain`).
2. **Category → specialist dispatch** — a **per-component setting**
   (editable in the UI, no code change) decides which categories activate
   which specialist. Out of the box, only `human → face recognition` is
   active; a category with no specialist configured is still detected and
   reported with its real class/category, just without further
   identification — a normal outcome, not a gap.
3. **Specialist identification** — the face-recognition specialist
   (**facenet-pytorch**: MTCNN for detection/alignment, InceptionResnetV1 on
   vggface2 weights for 512-dim embeddings) matches against a database of
   registered individuals.
4. **Reaction policy** — a configurable action set (`log`, `display`,
   `alert`, `save_frame`) plus a priority, keyed by `(specialist,
   identity_key)`. `"unknown"` is a first-class identity with its own
   policy row, not a special case.

**Consent and privacy**: only people explicitly registered, with consent,
are ever identified by name — everyone else is always `"unknown"`. A person
can have **multiple registered photos** (recommended, under varying
conditions), matched by minimum distance to *any* of their embeddings
rather than to a centroid — more robust to real-world variation. Once at
least two individuals with a couple of photos each are registered, a
**threshold-calibration** view shows the actual observed
same-person-vs-different-person embedding-distance distributions, so the
match threshold is calibrated against real data instead of guessed.
Registered photos are stored to disk (downscaled) alongside the embeddings
so the registration UI can show what was actually registered; withdrawing
consent for a person permanently deletes every one of their embeddings and
photos, not just their database row.

## Live cascade streaming (Kafka / video / still image)

The cascade isn't limited to one-shot testing. A dedicated **Cascade
Stream** page drives continuous operation against three input sources, all
sharing the exact same per-frame throttle-and-dispatch logic:

- **Apache Kafka** — a standalone, long-lived consumer process
  (`python -m emil_ml.cascade_stream --component <name>`, or the
  `emil-cascade-stream` console script) consumes one topic per component,
  one Kafka message = one frame (raw image bytes). It is *never* launched
  from inside Streamlit — a Streamlit session restarts on every code
  change, which would kill an in-process consumer, exactly the same
  reasoning the folder watcher already follows. The Streamlit page only
  configures settings (bootstrap servers, topic) and polls the database for
  live status (a heartbeat timestamp tells "actually running" apart from
  "crashed without saying so").
- **Video file upload** — processed synchronously in the page itself, with
  a progress bar and a live single-frame preview that refreshes in place
  frame-by-frame as it works through the video (not a growing list until
  it's finished).
- **Still image upload** — runs the cascade once, unthrottled, immediately.

A **configurable sample rate** ("check at most one frame every N seconds")
throttles all three sources by the *frame's own position* (real elapsed
time for Kafka, the video's own timeline for a file) rather than wall-clock
processing speed, so the same setting means the same thing regardless of
source. Every processed frame is thumbnailed (with detection boxes drawn)
and shown in a live-updating results feed; every frame where the cascade
**recognized** a registered individual also gets filed into
`analyzed/<identity>/<identity>_<timestamp>.png` under that component's own
folder — the same `analyzed/` convention every other model type already
uses for approved/failed images, just keyed by identity instead of verdict.
A one-click **"Clear results & images"** button purges the accumulated
per-frame images and result rows for a component (run history/counters are
kept) — there is no automatic retention job for these yet.

## AI-generated reports (RAG)

Any component can optionally generate a written explanation alongside its
verdict — gated by two settings: `reporting_enabled`, and
`reporting_condition` (`never` / `always` / `on_failed` / a specific list of
predicted classes). When triggered:

1. **Machine context** — if the component has defined machine parameters
   (name, unit, normal min/max range, wording for above/below-normal), the
   latest live reading is compared against them to produce a set of
   human-readable anomaly descriptions (e.g. "temperature over-normal").
2. **Knowledge retrieval** — a per-component knowledge base of uploaded
   `.md`/`.txt` documents (with YAML frontmatter for doc type and source) is
   chunked, embedded via a local Ollama embedding model
   (`nomic-embed-text`), and stored in **ChromaDB**. At report time, the
   verdict/defect classes and any machine-context anomalies drive a
   metadata-filtered similarity search, with both an absolute similarity
   floor and a per-doc-type relative margin to filter out merely-similar
   noise.
3. **Generation** — if nothing relevant was retrieved, the report says so
   honestly and **no LLM call happens at all** — the model is never given a
   chance to write about something ungrounded. Otherwise, a prompt is built
   from what was retrieved and sent to a local **Ollama** instance
   (`qwen3:8b` by default), streamed token-by-token — including the model's
   own visible "thinking" — so the UI can show live progress instead of a
   blank spinner. Report generation runs on a single, serialized background
   worker (Ollama can only usefully run one generation at a time on one
   GPU), so the fast verdict is never blocked waiting for it.

The full provenance is kept alongside every report: which sources were
cited, the exact prompt sent, the model's raw reasoning, and which machine
parameters were considered — all inspectable from the Inspect page and the
Inspection Station.

## Inspection lifecycle & operator review

Every inspection moves through a simple, explicit workflow:
**new → acknowledged → archived → (permanently deleted per retention)**.
The **Inspection Station** page is where an operator works through this
queue: filterable/sortable (by component, verdict, lifecycle status, report
status, verification status), bookmarkable via URL query params, with bulk
acknowledge/archive actions and per-record acknowledge/archive/revert
buttons.

A separate axis, **verified corrections**, is how a human operator's
judgment feeds back into future retraining: any inspection can be marked
"verified correct" or flagged incorrect (false positive, false negative, or
— for YOLO — a wrong-class/wrong-box correction via an interactive
box-drawing canvas). Depending on a per-component policy (off / manual
review / automatic), these verified corrections become available as extra
training data on the next training run — and an inspection with a
not-yet-incorporated verified label is protected from retention cleanup
regardless of age, so a correction is never silently lost.

Retention itself (`inspection_retention_days` per component) only ever
permanently deletes *archived* inspections older than the window, and only
via an explicit "run cleanup now" action — never automatically.

## Automated ingestion: the folder watcher

For unattended, production-style ingestion, `emil-watcher`
(`python -m emil_ml.watcher`) is a standalone long-lived process — same
"never run inside Streamlit" reasoning as the cascade-stream consumer — that
watches every active component's `input/` folder and inspects a file the
moment it appears, calling the exact same code path (`run_inspection()`)
the UI itself uses. Detection is via filesystem events (immediate) backed
by a periodic full rescan (a safety net for missed events, especially over
a network share, and how a newly onboarded component starts being watched
without a restart). A file is only acted on once its size has stopped
changing across several checks, so a camera's still-writing frame is never
read mid-write. A file that fails to process is moved to `error/` with a
logged reason, never silently dropped or retried forever.

## Other tools: diagnostics, grid search, component lifecycle

- **Diagnostics** — before investing time training, project a component's
  approved/failed training images (via a frozen CNN embedding extractor)
  into 2D with UMAP or PCA, and see a quantified separability score
  (silhouette score, intra/inter-class distance) plus a plain-language
  recommendation — including, when classes overlap heavily in whole-image
  feature space, a nudge toward YOLO (localizable defects) instead.
- **Grid search** — sweep a component's hyperparameters, training each
  combination against an isolated scratch copy so the live component is
  never touched; results are ranked by a configurable metric, with the
  winning combination reported for manual application.
- **Component lifecycle** — soft-delete moves a component to a fully
  reversible trash (a pure status flag, nothing touched); permanent
  deletion is two-phase (only ever acts on an already-trashed component)
  and cleans up every subsystem a component touches — filesystem,
  ChromaDB chunks, inspections, machine readings, training runs, and
  cascade-stream results — with an itemized impact preview shown before
  the irreversible click, calling out human-verified corrections
  specifically since they can't be regenerated.

## Pages at a glance

| Page | Purpose |
|---|---|
| **Home** | Component counts at a glance, quick links. |
| **Inspect** | Run a trained (non-cascade) component against an upload or override its threshold for the session; see the verdict, annotated result, and any generated report. |
| **Onboard** | Create/edit/train every component, upload training data and YOLO annotations (3 ways), configure reporting/knowledge base/machine parameters, run grid search, manage the whole cascade (registration, policies, calibration, one-shot test run), and the trash. |
| **Inspection Station** | The operator review queue — filter, sort, bulk-act, verify/correct. |
| **Diagnostics** | Pre-training class-separability visualization. |
| **Cascade Stream** | Continuous cascade operation: Kafka status, video/still-image processing, live results feed, cleanup. |

## Architecture

All ML/business logic lives in the importable `emil_ml` package under
`src/`. The Streamlit UI (`app/`) is a thin view layer only — the core
pipeline modules (`pipeline/inspect.py`, `core/cascade/pipeline.py`,
`training/onboard.py`) contain no Streamlit imports, so the same core could
be wrapped by a different frontend without modification. Three independent
entry points share that same core: the Streamlit app, the folder watcher,
and the cascade-stream consumer.

A few conventions worth knowing if you're reading the code:

- **Dispatch is always data, never a branch.** Which trainer/predictor a
  component uses, which specialist a coarse category activates, which
  reaction a recognized identity gets — all of these are looked up from a
  registry or a per-component setting, never an `if model_type == ...`
  scattered through business logic. Adding a new method or specialist means
  implementing an interface and registering it in one place.
- **Per-component settings, not global config**, for anything that could
  reasonably differ between components (image size, thresholds, Kafka
  topic, sample rate, ...) — stored as columns on the `components` table,
  editable from the UI without a code change.
- **Domain-specific, self-contained SQLite tables** (`training_runs`,
  `inspections`, `cascade_stream_runs`/`results`, the face-recognition
  tables, ...) each own their own schema/migrations, separate from the
  central `components` schema — a component's *settings* and its
  *history/telemetry* are deliberately different tables.
- **Everything not directly needed to run the app is gitignored**:
  `data/`, `emil.db`, `results/`, virtual environments. See
  `.gitignore`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

Optional extras (installed the same way, e.g. `pip install -e ".[cascade,kafka,video]"`):

| Extra | Adds | Needed for |
|---|---|---|
| `patchcore` | `anomalib` | The PatchCore analysis method |
| `rag` | `chromadb` | AI-generated reports' knowledge-base vector store |
| `cascade` | `facenet-pytorch` | The face-recognition specialist |
| `kafka` | `confluent-kafka` | Live cascade streaming from a Kafka topic |
| `video` | `opencv-python-headless` | Cascade video-file / live-preview frame decoding |
| `gpu` | `tensorflow` (Linux/WSL2 only) | GPU-accelerated TensorFlow methods — see below |
| `dev` | `pytest` | Running the test suite |

Every heavy/optional dependency is imported lazily inside the function that
actually needs it — installing the base package with no extras is enough to
run the app; a page or feature that needs an extra will only complain when
you actually use it.

AI-generated reports additionally need a local
[Ollama](https://ollama.com/) instance running (`http://localhost:11434` by
default) with the `qwen3:8b` and `nomic-embed-text` models pulled.

## Running the app

```bash
emil-ml
```

A console script (registered in `pyproject.toml`) that wraps
`streamlit run app/streamlit_app.py` — a plain `pip install -e .` is enough
to get a runnable `emil-ml` command. Any extra arguments are passed through
to Streamlit, e.g. `emil-ml --server.port 8502`.

For automated ingestion or live cascade streaming, run the standalone
processes alongside it (each is its own long-lived process, never started
by the Streamlit app itself):

```bash
emil-watcher                                    # watches every active component's input/ folder
emil-cascade-stream --component <component-name>  # consumes that component's configured Kafka topic
```

## Verification scripts

`scripts/` contains a large set of disposable, re-runnable `verify_*.py`
scripts used as this project's test suite — each exercises a real code path
end-to-end (often via Streamlit's own `AppTest`, driving actual widgets, or
against real images) rather than mocking. A few starting points:

```bash
python scripts/verify_foundation.py         # registry + folder layout round-trip
python scripts/verify_pipeline.py           # core inspect() pipeline
python scripts/verify_cascade_full.py       # cascade steps 2-4 + full run_cascade() end-to-end
python scripts/verify_cascade_stream_video.py  # live-stream video path end-to-end
python scripts/verify_face_photo_storage.py    # face-registration photo storage/consent-completeness
```

## GPU training (WSL2)

GPU support differs by framework, and one method is currently CPU-only
regardless of platform:

- **TensorFlow-based methods** (Autoencoder, Classifier, ResNet coarse
  classifier, plus the Isolation Forest/Diagnostics MobileNetV2 embedding
  extractor) — TensorFlow dropped GPU support on native Windows after
  2.10, so on a Windows venv these **always run on CPU** regardless of
  drivers. GPU requires a separate setup in WSL2/Ubuntu (below).
- **PyTorch-based methods** (YOLO, the COCO cascade detector, the
  face-recognition specialist) — PyTorch has working CUDA wheels for
  native Windows, so these can generally use the GPU without WSL
  (verify with `torch.cuda.is_available()` on your own machine/driver
  combination).
- **PatchCore** — as currently configured, its anomalib `Engine` is pinned
  to `accelerator="cpu"` explicitly in `core/anomaly/patchcore/adapter.py`,
  so it runs on CPU regardless of platform or GPU availability, even though
  the underlying anomalib/PyTorch stack is itself GPU-capable.

To use the GPU for the TensorFlow-based methods (verified working with an
RTX 5070 in WSL2/Ubuntu), a second, GPU-enabled setup lives in WSL2:

- **Why a separate copy under `~/emil_ml` in WSL, instead of running against
  `/mnt/c/...` directly**: building the package's editable install on the
  Windows-mounted filesystem (`/mnt/c`) fails outright with a DrvFs
  permission error on `egg_info`, and I/O there is much slower generally.
  Windows stays the source of truth for editing; the WSL copy is a
  build/run mirror.
- **The `gpu` extra**: `pyproject.toml`'s `[project.optional-dependencies].gpu`
  installs plain `tensorflow` (Linux/WSL2 only; meaningless on Windows) —
  deliberately *not* `tensorflow[and-cuda]`, see below.
- **One-time WSL setup** (already done on this machine): `uv python install
  3.12` (Ubuntu 26.04's default Python is 3.14, same TF incompatibility as
  Windows), then `uv venv --python 3.12 ~/emil_ml/.venv && uv pip install
  --python ~/emil_ml/.venv/bin/python -e '.[gpu]'` from within `~/emil_ml`.
- **TensorFlow + a CUDA-13 PyTorch (e.g. torch+cu130 for Blackwell/RTX
  50-series) in the same venv**: after installing a GPU-appropriate torch
  build, run `scripts/wsl_gpu_setup.sh` (from `~/emil_ml` in WSL) to add the
  handful of CUDA-12-SONAME libraries TensorFlow's GPU support hard-requires
  (`libcudart.so.12`, `libcublas.so.12`, `libcublasLt.so.12`,
  `libcufft.so.11`, `libcusolver.so.11`, `libnvJitLink.so.12`) *without*
  pulling in `tensorflow[and-cuda]`'s own cuDNN — that collides file-for-file
  with the cu13 cuDNN torch needs, since cuDNN's `.so` filename doesn't
  encode which CUDA major version it targets. A single cu13 cuDNN install is
  shared successfully by both frameworks; see the script's header comment
  for the full story (confirmed working 2026-07-22: TensorFlow reports a GPU
  device and runs a real Conv2D, PyTorch/YOLO training is unaffected).
- **`emil_ml/__init__.py` auto-fixes GPU discovery**: TensorFlow's RPATH-based
  auto-discovery of the pip-installed CUDA libs didn't work reliably under
  this uv-managed venv (silently fell back to CPU). The package detects this
  on import (Linux only) and re-exec's itself once with `LD_LIBRARY_PATH`
  pointed at the installed `nvidia-*` packages — no manual steps needed.
- **One caveat found the hard way**: Ultralytics' automatic mixed-precision
  startup check exercises a Triton-JIT kernel that crashes on an RTX 5070
  (Blackwell, compute capability 12.0a) — `YoloTrainer.train()` disables it
  (`amp=False`) unconditionally as a result.

Day-to-day workflow:

```powershell
# 1. After editing on Windows, mirror the project into WSL:
.\scripts\sync_to_wsl.ps1

# 2. Run/train from inside WSL:
wsl -d Ubuntu -- bash -lc "cd ~/emil_ml && .venv/bin/emil-ml"
# or, for a one-off training/inspection script:
wsl -d Ubuntu -- bash -lc "cd ~/emil_ml && .venv/bin/python scripts/verify_pipeline.py"
```

Note: the very first GPU op on a given machine/driver combo may trigger a
one-time CUDA JIT compile (TensorFlow warns it "could take 30 minutes or
longer" for compute capabilities without precompiled kernels, e.g. this RTX
50-series card). In practice this took well under a minute here — the
warning is a worst-case, not a typical case.

## Libraries & dependencies

**Core** (always installed): `tensorflow`, `streamlit`, `watchdog`,
`pillow`, `numpy`, `ultralytics` (PyTorch-based YOLO), `scipy`
(mask→bounding-box conversion), `streamlit-drawable-canvas` (manual box
annotation), `truststore` (TLS via the OS certificate store), `umap-learn` +
`scikit-learn` (diagnostics projection, PCA, Isolation Forest, evaluation
metrics), `matplotlib` (evaluation-report plots).

**Optional** (see the extras table under [Setup](#setup)): `anomalib`
(PatchCore), `chromadb` (RAG vector store), `facenet-pytorch` (face
recognition), `confluent-kafka` (live Kafka streaming), `opencv-python-headless`
(video decoding), `tensorflow` GPU build (WSL2 only).

**External services, both optional and both local-only**: [Ollama](https://ollama.com/)
(text generation via `qwen3:8b`, embeddings via `nomic-embed-text`) and an
Apache Kafka broker (only if using live cascade streaming) — neither is
required to use the rest of the app.
