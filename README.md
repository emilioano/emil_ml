# EMIL Lab — Enhanced Machine Inspection & Learning Lab

Modular, Streamlit-based POC for industrial anomaly-detection / inspection.
Onboard a component type, train a per-component autoencoder anomaly detector,
and inspect new images (upload or watched folder) as **approved** / **failed**.

## Status: Phase 1 — foundation

Implemented so far:

- `src/emil_ml/config/settings.py` — global paths and defaults
- `src/emil_ml/config/database.py` — SQLite schema + connection helpers
- `src/emil_ml/config/registry.py` — `ComponentRegistry` CRUD over the `components` table
- `src/emil_ml/utils/paths.py` — deterministic per-component folder layout, derived from a slug

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## Run the app

```bash
emil-ml
```

This is a console script (registered in `pyproject.toml`) that wraps
`streamlit run app/streamlit_app.py`, so a plain `pip install -e .` is enough
to get a runnable `emil-ml` command. Any extra arguments are passed through
to Streamlit, e.g. `emil-ml --server.port 8502`.

## Verify the foundation

```bash
python scripts/verify_foundation.py
```

This initializes `emil.db`, creates a dummy component ("Widget 42"), creates
its folder tree under `data/components/widget-42/`, exercises the registry's
update/list methods, and asserts everything round-trips correctly.

## Architecture

All ML/business logic lives in the importable `emil_ml` package. The
Streamlit UI (`app/`) is a thin view layer only — `pipeline/inspect.py` and
`training/onboard.py` contain no Streamlit imports, so the same core can
later be wrapped by a FastAPI + React frontend without modification.

## GPU training (WSL2)

TensorFlow dropped GPU support on native Windows after 2.10 — on this
machine's Windows venv, training always runs on CPU regardless of drivers.
To use the GPU (verified working with an RTX 5070 in WSL2/Ubuntu), a second,
GPU-enabled setup lives in WSL2:

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
