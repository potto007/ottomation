"""Kyutai STT backend choice, and Kyutai STT 1B on CUDA through PyTorch (the `moshi`
package) for Linux with an Nvidia GPU.

load_kyutai() picks the backend for --stt kyutai: MLX on Apple silicon (KyutaiSTT in
speech.py), CUDA PyTorch when torch sees a GPU with room for the model, else None and
the caller falls back to Whisper. Both backends step the same model and weights
(kyutai/stt-1b-en_fr-candle: the plain PyTorch repo has no extra heads, and extra heads
0..3 are the pause predictors behind the semantic end of turn) behind the interface KyutaiTurns drives: steps,
MAX_STEPS, delay_steps, reset(), step(block) and transcribe().

The CUDA stepping follows kyutai-labs/delayed-streams-modeling
(scripts/stt_from_file_pytorch.py --vad) and moshi's server.py (streaming_forever, a
short warmup that captures the CUDA graphs, reset_streaming between uses)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from . import gpu, speech
from .protocol import log, warn

# The model (bf16), Mimi and the CUDA graphs allocate about 2.4 GiB, and with the CUDA
# context the process holds about 3.2 GiB (measured on an RTX 5090). Below this much
# free VRAM the GPU is busy with something else, and loading anyway would leave the
# driver no headroom (on WSL2 it then pages GPU memory out to system RAM).
MIN_FREE_GIB = 5.0


class KyutaiStepper(Protocol):
    """What KyutaiTurns needs from a Kyutai STT backend."""

    MAX_STEPS: int
    delay_steps: int

    @property
    def steps(self) -> int: ...

    def reset(self) -> None: ...

    def step(self, block: np.ndarray) -> tuple[str | None, speech.Pauses]: ...

    def transcribe(self, payload: Any) -> str: ...


def cuda_unavailable() -> str | None:
    """Why the CUDA backend cannot run here, or None when it can. torch is imported
    only when moshi is installed (Linux x86_64; see main.py's dependencies)."""
    if (
        importlib.util.find_spec("moshi") is None
        or importlib.util.find_spec("torch") is None
    ):
        return "the moshi package is not installed (it installs on Linux x86_64 only)"
    import torch

    if not torch.cuda.is_available():
        return "PyTorch sees no CUDA GPU"
    # nvidia-smi first: torch's reading opens a CUDA context, which costs VRAM a full GPU does not have
    free = gpu.read_free_gib()
    if free is None:
        free = torch.cuda.mem_get_info()[0] / 2**30
    need = gpu.NEED_GIB["kyutai"]
    return gpu.admit("stt", need, MIN_FREE_GIB - need, free)


def kyutai_backend() -> tuple[str | None, str]:
    """('mlx' | 'cuda', '') when Kyutai STT can run here, else (None, why not). On CUDA the recognizer's GPU claim
    ("stt" in gpu.py) now holds Kyutai's share, until the caller releases it."""
    if speech.is_apple_silicon():
        return "mlx", ""
    why = cuda_unavailable()
    return (None, why) if why else ("cuda", "")


def load_kyutai() -> KyutaiStepper | None:
    """The Kyutai STT backend for this machine, or None (logged) for Whisper."""
    backend, why = kyutai_backend()
    if backend is None:  # the default stt everywhere: a log, not a toast on every start
        log(
            "stt: Kyutai STT needs Apple silicon (MLX) or an Nvidia GPU (CUDA); "
            f"{why}; using Whisper."
        )
        return None
    try:
        return speech.KyutaiSTT() if backend == "mlx" else KyutaiCudaSTT()
    except Exception as e:  # noqa: BLE001
        warn(
            f"Kyutai STT on {backend.upper()} unavailable ({type(e).__name__}: "
            f"{str(e)[:120]}); using Whisper."
        )
        return None


class KyutaiCudaSTT:
    """Kyutai STT 1B on CUDA: 80 ms of 24 kHz audio in, at most one word piece and the
    pause probabilities (speech.Pauses) out. Built and stepped on the listener thread only (the CUDA
    graphs are captured there)."""

    where = "CUDA"
    MAX_STEPS = 4096  # matches the MLX path; KyutaiTurns resets before it
    WARMUP_STEPS = 4

    def __init__(self, device: str = "cuda") -> None:
        import torch
        from moshi.models import LMGen, loaders

        self._torch, self.device, self._steps = torch, device, 0
        config = Path(speech._fetch(speech.KYUTAI_STT_REPO, "config.json"))
        raw = json.loads(config.read_text())
        # Local paths throughout: no network once the weights are cached.
        info = loaders.CheckpointInfo.from_hf_repo(
            speech.KYUTAI_STT_REPO,
            moshi_weights=self._get(raw.get("moshi_name", "model.safetensors")),
            mimi_weights=self._get(raw["mimi_name"]),
            tokenizer=self._get(raw["tokenizer_name"]),
            config_path=config,
        )
        self.delay_steps = round(
            info.stt_config["audio_delay_seconds"]
            * speech.KYUTAI_SR
            / speech.KYUTAI_BLOCK
        )
        with torch.no_grad():
            self.mimi = info.get_mimi(device=device)
            self.tok = info.get_text_tokenizer()
            self.lm = info.get_moshi(device=device, dtype=torch.bfloat16)
            if not self.lm.extra_heads:
                raise RuntimeError(f"{speech.KYUTAI_STT_REPO} has no end-of-turn head")
            self.gen = LMGen(self.lm, temp=0, temp_text=0)
            self.mimi.streaming_forever(1)
            self.gen.streaming_forever(1)
            silence = np.zeros(speech.KYUTAI_BLOCK, np.float32)
            for _ in range(self.WARMUP_STEPS):
                self.step(silence)
            torch.cuda.synchronize()
        self.reset()
        mib = torch.cuda.memory_allocated() / 2**20
        log(
            f"stt: Kyutai STT 1B on CUDA ({torch.cuda.get_device_name()}), "
            f"{mib:.0f} MiB allocated"
        )

    @staticmethod
    def _get(name: str) -> Path:
        return Path(speech._fetch(speech.KYUTAI_STT_REPO, name))

    def reset(self) -> None:
        self.mimi.reset_streaming()
        self.gen.reset_streaming()
        self._steps = 0

    @property
    def steps(self) -> int:
        return self._steps

    def step(self, block: np.ndarray) -> tuple[str | None, speech.Pauses]:
        """One 1920-sample block -> (word piece or None, pause probabilities)."""
        torch = self._torch
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(block, np.float32))
            codes = self.mimi.encode(x.to(self.device)[None, None])
            out = self.gen.step_with_extra_heads(codes)
        self._steps += 1
        if out is None:  # still filling the model's delays; none for this model
            return None, speech.Pauses()
        tokens, heads = out
        token, *p = torch.stack(
            [tokens[0, 0, 0].float()] + [h[0, 0, 0].float() for h in heads[:4]]
        ).tolist()  # one sync
        token = int(token)
        piece = None if token in (0, 3) else self.tok.id_to_piece(token)
        return piece, speech.Pauses.of(p)

    def transcribe(self, payload: Any) -> str:
        return payload if isinstance(payload, str) else ""
