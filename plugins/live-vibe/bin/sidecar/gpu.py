"""One budget for the GPU backends that load at the same time: Kyutai STT or Whisper on the listener thread, Kokoro on
the main thread. A backend asks admit() before it allocates; admit() reads free VRAM, subtracts what the other
backends have claimed but not yet allocated, and says yes only if the GPU keeps `floor` GiB free after this one.
On WSL2 a GPU near full pages its memory out to system RAM (WDDM) and every model on it slows down, and the GPU is
often shared with a local LLM server this process cannot see, so CPU is the answer whenever the room is not there.

main holds a placeholder for the recognizer before either thread starts, so the recognizer gets the GPU first and
Kokoro (fast enough on the CPU) yields; a backend releases its claim once its memory shows in the reading."""
from __future__ import annotations

import os
import subprocess
import threading
from collections.abc import Callable

# Measured on an RTX 5090 under WSL2, with the CUDA context each one brings.
NEED_GIB = {"kyutai": 3.2, "whisper": 1.0, "kokoro": 1.4}
FLOOR_GIB = 3.0  # what Whisper and Kokoro leave free; Kyutai keeps its own (kyutai_cuda.MIN_FREE_GIB)

_lock = threading.Lock()
_claims: dict[str, float] = {}


def nvidia_smi_free_gib() -> float | None:
    """Free memory of the first visible GPU, from nvidia-smi: no CUDA context, so asking costs no VRAM. On WSL2 it
    counts what Windows processes hold too. None when there is no nvidia-smi or no GPU."""
    cmd = ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
    first = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    if first.isdigit():
        cmd += ["-i", first]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=True).stdout
        return float(out.split()[0]) / 1024
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


read_free_gib: Callable[[], float | None] = nvidia_smi_free_gib  # replaced in the unit checks


def hold(name: str, gib: float) -> None:
    """Claim `gib` for `name` before anything is allocated (main's placeholder for the recognizer)."""
    with _lock:
        _claims[name] = gib


def release(name: str) -> None:
    with _lock:
        _claims.pop(name, None)


def claimed(exclude: str = "") -> float:
    with _lock:
        return sum(g for n, g in _claims.items() if n != exclude)


def admit(name: str, need_gib: float, floor_gib: float = FLOOR_GIB, free_gib: float | None = None) -> str | None:
    """None, and `name` holds `need_gib`, when the GPU keeps `floor_gib` free after it and after the others' claims;
    else why not. `free_gib` is the caller's own reading (torch's); otherwise nvidia-smi is asked."""
    free = read_free_gib() if free_gib is None else free_gib
    if free is None:
        return "free GPU memory is unknown (no nvidia-smi)"
    with _lock:
        others = sum(g for n, g in _claims.items() if n != name)
        left = free - others - need_gib
        if left < floor_gib:
            held = f", {others:.1f} GiB of it claimed by the other voice models" if others else ""
            return (f"only {free:.1f} GiB of GPU memory free{held}; it needs {need_gib:.1f} GiB and leaves "
                    f"{floor_gib:.1f} GiB free")
        _claims[name] = need_gib
    return None
