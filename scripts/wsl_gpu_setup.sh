#!/usr/bin/env bash
# Makes TensorFlow's GPU support work alongside a CUDA-13-targeted PyTorch
# (e.g. torch+cu130, needed for Blackwell/RTX 50-series GPUs) in the SAME
# WSL2 venv. Run from inside the venv's project root after `uv pip install
# -e .` and after installing your GPU-appropriate torch build (this script
# does not install torch — it assumes `torch.cuda.is_available()` already
# works before you run it).
#
# Background: TensorFlow 2.x's GPU support hard-requires CUDA-12-SONAME
# libraries (libcudart.so.12, libcublas.so.12, libcublasLt.so.12,
# libcufft.so.11, libcusolver.so.11, libnvJitLink.so.12 — confirmed via
# direct dlopen testing on 2026-07-22). The obvious fix, installing
# `tensorflow[and-cuda]`, pulls in its own pinned nvidia-cudnn-cu12 — but
# cuDNN's .so filename (libcudnn.so.9) does NOT encode which CUDA major
# version it was built for, so nvidia-cudnn-cu12 and nvidia-cudnn-cu13
# collide file-for-file in the shared nvidia/cudnn/lib/ directory. Whichever
# installs second partially overwrites the first, leaving an internally
# inconsistent cuDNN that crashes BOTH frameworks with
# CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH (or a raw segfault inside a
# DataLoader worker, no Python traceback).
#
# The fix: never install nvidia-cudnn-cu12 (or nvidia-cusparse-cu12 —
# same-filename problem) at all. A single cu13-targeted cuDNN/cusparse
# install is shared successfully by both frameworks (verified: TensorFlow
# loads "cuDNN version 92500" and runs a real Conv2D on GPU; PyTorch's own
# conv2d still works unchanged). Only the genuinely non-colliding cu12
# libraries (different SONAME per CUDA major version) need adding, and only
# those specific packages, via --no-deps so pip's resolver doesn't pull in
# nvidia-cusparse-cu12/nvidia-cudnn-cu12 as transitive dependencies.
#
# nvidia-cuda-nvcc-cu12 is also required — not a shared library (so a plain
# dlopen/ctypes test won't catch its absence), it ships libdevice.10.bc, the
# LLVM bitcode NVVM needs to actually compile a CUDA kernel via ptxas rather
# than fall back to the driver's own (much more limited) JIT compiler. That
# fallback silently works for simple ops but fails outright for anything
# cuDNN's autotuner searches multiple algorithm configs for — confirmed by
# reproducing the real failure: autoencoder training crashed with "Autotuner
# could not compile any configs for HLO: %cudnn-conv..." on a Conv2DTranspose
# specifically, which the isolated single-op smoke test below doesn't
# exercise. Diagnose from a real Conv2DTranspose fwd+bwd pass, not just
# Conv2D, if this class of error resurfaces.
set -euo pipefail

PYTHON="${1:-.venv/bin/python3}"

echo "Installing cu12 CUDA runtime libraries TensorFlow needs (--no-deps, to avoid pulling in colliding cudnn/cusparse cu12 packages)..."
"$PYTHON" -m pip install --no-deps \
    nvidia-cuda-runtime-cu12 \
    nvidia-cuda-nvcc-cu12 \
    nvidia-cublas-cu12 \
    nvidia-cufft-cu12 \
    nvidia-cusolver-cu12 \
    nvidia-nvjitlink-cu12

echo
echo "Verifying (see wsl_gpu_verify.py for what this actually exercises and why)..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$PYTHON" "$SCRIPT_DIR/wsl_gpu_verify.py"
