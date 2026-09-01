"""EMIL Lab - Enhanced Machine Inspection & Learning Lab."""

from __future__ import annotations

import os
import sys
from pathlib import Path

__version__ = "0.1.0"


def _configure_tf_gpu_memory_growth() -> None:
    """Prevent TensorFlow from eagerly over-reserving GPU memory upfront.

    TF's default BFC allocator pre-reserves a large contiguous block per
    process and grows it in big steps. On a GPU that doesn't have much spare
    room (observed: an 8GB laptop GPU under WSL2, where the allocator's own
    reported limit was ~5GB, not the full 8 — some VRAM is claimed
    elsewhere, e.g. WSL2's display-sharing path), that default strategy can
    run out of room and then retry the same failing allocation in a tight
    loop, flooding the log with allocator dumps instead of failing once with
    a clear error. "Allow growth" makes TF allocate incrementally instead —
    same total memory ceiling, just no wasted upfront over-reservation, so a
    real capacity shortage (e.g. batch_size/image_size too large for this
    GPU) still fails, just without the noisy retry loop.

    Must be set before TensorFlow is imported anywhere in the process (an
    env var, not a runtime call, for exactly that reason). Harmless where
    it doesn't apply: no GPU to grow into on Windows, and a no-op if
    TensorFlow is never imported at all (e.g. a YOLO-only workflow).
    """
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")


def _configure_nvidia_ld_library_path() -> None:
    """On Linux, re-exec with LD_LIBRARY_PATH pointing at pip-installed NVIDIA CUDA/cuDNN libs.

    The `tensorflow[and-cuda]` extra installs CUDA/cuDNN as regular pip
    packages rather than system libraries. TensorFlow is supposed to find
    them via RPATH automatically, but that doesn't reliably happen (observed
    under a uv-managed venv in WSL2) and silently falls back to CPU.

    Mutating os.environ mid-process does *not* fix this: glibc's dynamic
    loader only honors LD_LIBRARY_PATH at process start, not for dlopen()
    calls made later in an already-running process (confirmed by testing —
    setting it before `import tensorflow` in the same process still failed).
    So if it's missing, we set it and re-exec this exact process once,
    guarded by an env var to prevent looping. A no-op on Windows (no
    `nvidia` package there) and if the libs are already discoverable.
    """
    if sys.platform != "linux" or os.environ.get("_EMIL_ML_GPU_ENV_SET"):
        return
    try:
        import nvidia
    except ImportError:
        return

    # `nvidia` is normally a PEP 420 implicit namespace package — assembled
    # from multiple nvidia-*-cuXX wheels each contributing a subdirectory,
    # with no __init__.py of its own — so __file__ is None. __path__ (a
    # _NamespacePath, iterable of directories) works for both namespace and
    # regular packages, unlike __file__.
    lib_dirs = [
        str(p) for base in nvidia.__path__ for p in Path(base).glob("*/lib") if p.is_dir()
    ]
    if not lib_dirs:
        return

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if all(d in existing for d in lib_dirs):
        return  # already set, e.g. by a parent process

    os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([existing] if existing else []))
    os.environ["_EMIL_ML_GPU_ENV_SET"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


_configure_tf_gpu_memory_growth()
_configure_nvidia_ld_library_path()
