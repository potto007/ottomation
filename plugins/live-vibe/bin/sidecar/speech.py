"""Speech backends and their fallbacks: STT (Kyutai STT 1B on MLX or CUDA, or Silero + faster-whisper) and TTS (Kokoro-82M,
macOS say, or silence). Each loader degrades to the next with a warn naming the fix; none ends the process.

WhisperASR, KokoroTTS and SayTTS are ported from proto/duplex_voice.py. The Kyutai loading and stepping follow
kyutai-labs/delayed-streams-modeling (scripts/stt_from_mic_mlx.py; MIT for the Python code; weights CC-BY 4.0)."""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import wave
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .audio import JUNK_TRANSCRIPTS, SR, CannotStart, Detector, EnergyVAD, SileroVAD, Tuning, TurnDetector
from .protocol import log, warn

# Shared with the prototype, so the models it downloaded are reused.
CACHE = Path(os.environ.get("LIVE_VIBE_CACHE") or Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "duplex_voice")
SILERO_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
KOKORO_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
DOWNLOAD_TIMEOUT_S = 30  # per socket read; a stalled download fails instead of hanging the start


def download(name: str, url: str) -> Path:
    path = CACHE / name
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    log(f"download: {url}")
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    tmp.replace(path)
    return path


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


# -- STT: the recognizer and its turn detector, built on the listener thread -------------------------------------
def recognizer(stt: str, asr: str, end_silence_ms: int, speaking: Callable[[], bool]) -> tuple[Detector, Callable[[Any], str]]:
    """(detector, transcribe). Kyutai brings its own end of turn and passes text through; Whisper pairs with
    Silero (or the energy gate) and transcribes the utterance audio."""
    tune = Tuning(end_silence_ms=end_silence_ms)
    if stt == "kyutai":
        from .kyutai_cuda import load_kyutai

        k = load_kyutai()  # MLX on Apple silicon, CUDA PyTorch on an Nvidia GPU, or None
        if k is not None:
            return KyutaiTurns(k, tune, speaking), k.transcribe
    w = WhisperASR(asr)
    return TurnDetector(load_vad(), tune, speaking), w.transcribe


def load_vad():
    try:
        return SileroVAD(str(download("silero_vad.onnx", SILERO_URL)))
    except Exception as e:  # noqa: BLE001
        warn(f"Silero VAD unavailable ({type(e).__name__}: {str(e)[:120]}); using an energy gate, which is "
             "easier to trigger with noise.")
        return EnergyVAD()


class WhisperASR:
    """faster-whisper: CUDA float16 when CTranslate2 sees a GPU and its libraries load, else CPU int8."""

    def __init__(self, size: str = "base.en"):
        from faster_whisper import WhisperModel

        self.lang = "en" if size.endswith(".en") else None
        self.device = "cuda"
        if _cuda_devices() > 0:
            try:
                self.model = WhisperModel(size, device="cuda", compute_type="float16")
                self.transcribe(np.zeros(SR, np.float32))  # cuBLAS/cuDNN load on first use, so fail here
                log(f"asr: faster-whisper {size} on CUDA float16")
                return
            except Exception as e:  # noqa: BLE001
                warn(f"Whisper on CUDA failed ({type(e).__name__}: {str(e)[:120]}); using the CPU. Fix: install "
                     "CUDA 12 cuBLAS and cuDNN 9 (the CUDA toolkit, or pip nvidia-cublas-cu12 and "
                     "nvidia-cudnn-cu12 in the sidecar's environment).")
        self.device = "cpu"
        self.model = WhisperModel(size, device="cpu", compute_type="int8")
        log(f"asr: faster-whisper {size} on CPU int8")

    def transcribe(self, audio: np.ndarray) -> str:
        segs, _ = self.model.transcribe(audio, beam_size=1, language=self.lang, vad_filter=False,
                                        condition_on_previous_text=False)
        text = " ".join(s.text.strip() for s in segs).strip()
        return "" if text.lower() in JUNK_TRANSCRIPTS else text


def _cuda_devices() -> int:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count()
    except Exception:  # noqa: BLE001
        return 0


# -- Kyutai STT on MLX ----------------------------------------------------------------------------------------
# MLX asserts in Metal when two threads run it at once (measured with Kyutai TTS beside the STT); the listener
# thread is the only one that builds or steps this model.
KYUTAI_STT_REPO = "kyutai/stt-1b-en_fr-candle"  # the -mlx repo drops the extra heads that carry end of turn
KYUTAI_FILES = ("config.json", "model.safetensors", "mimi-pytorch-e351c8d8@125.safetensors", "tokenizer_en_fr_audio_8000.model")
KYUTAI_SR, KYUTAI_BLOCK = 24_000, 1920  # 80 ms steps at 12.5 Hz
BARGE_IN_WORDS = 3  # words the user must say over the assistant before it stops: bleed and "mm-hm" stay out
END_OF_TURN = 0.5  # the end-of-turn head (extra head 2, class 0): measured ~0 in speech, <0.5 in mid-sentence pauses,
# and above 0.5 from just before the last word on; it runs ahead of the words, which trail the audio by the model's
# audio delay, so a turn ends only once it has stayed above for that delay (0.5 s) and the last words are out


def kyutai_cached() -> bool:
    from huggingface_hub import try_to_load_from_cache

    return all(isinstance(try_to_load_from_cache(KYUTAI_STT_REPO, f), str) for f in KYUTAI_FILES)


_warned_download = False


def _fetch(repo: str, name: str) -> str:
    """hf_hub_download with progress lines: a first run downloads gigabytes."""
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    if isinstance(try_to_load_from_cache(repo, name), str):
        return hf_hub_download(repo, name)
    global _warned_download
    if not _warned_download:
        _warned_download = True
        warn("Kyutai STT's first run downloads its weights (about 2.4 GB); listening starts after.")
    log(f"kyutai: downloading {repo}/{name}")
    blobs = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub")) / f"models--{repo.replace('/', '--')}" / "blobs"
    done = threading.Event()

    def ticker() -> None:
        while not done.wait(10):
            parts = sorted(blobs.glob("*.incomplete"), key=lambda f: f.stat().st_mtime) if blobs.exists() else []
            if parts:
                log(f"kyutai: {name} {parts[-1].stat().st_size / 1e9:.2f} GB so far")

    threading.Thread(target=ticker, daemon=True).start()
    t = time.monotonic()
    try:
        return hf_hub_download(repo, name)
    finally:
        done.set()
        log(f"kyutai: {name} ready in {time.monotonic() - t:.0f}s")


class KyutaiSTT:
    """Kyutai STT 1B: 80 ms of 24 kHz audio in, at most one word piece and an end-of-turn probability out.
    Words trail the audio by about half a second. transcribe() passes KyutaiTurns' text through."""

    MAX_STEPS = 4096  # ~5.5 min of LmGen history; reset() before it, when nobody is talking

    def __init__(self) -> None:
        import mlx.core as mx
        import sentencepiece
        from moshi_mlx import models, utils

        self._mx, self._models, self._utils = mx, models, utils
        raw = json.load(open(_fetch(KYUTAI_STT_REPO, "config.json")))
        self.cfg = models.LmConfig.from_config_dict(raw)
        self.cfg.transformer.max_seq_len = self.cfg.transformer.context  # upstream workaround, moshi_mlx <= 0.3.0 ring kv cache
        self.lm = models.Lm(self.cfg)
        self.lm.set_dtype(mx.bfloat16)
        self.lm.load_pytorch_weights(_fetch(KYUTAI_STT_REPO, raw.get("moshi_name", "model.safetensors")), self.cfg, strict=True)
        self.tok = sentencepiece.SentencePieceProcessor(_fetch(KYUTAI_STT_REPO, raw["tokenizer_name"]))
        self.mimi = models.mimi.Mimi(models.mimi_202407(max(self.cfg.generated_codebooks, self.cfg.other_codebooks)))
        self.mimi.load_pytorch_weights(_fetch(KYUTAI_STT_REPO, raw["mimi_name"]), strict=True)
        self.delay_steps = round(raw["stt_config"]["audio_delay_seconds"] * KYUTAI_SR / KYUTAI_BLOCK)
        self.lm.warmup()
        self.reset()

    def reset(self) -> None:
        for c in self.lm.transformer_cache:
            c.reset()
        self.mimi.reset_all()
        self.gen = self._models.LmGen(model=self.lm, max_steps=self.MAX_STEPS, check=False,
                                      text_sampler=self._utils.Sampler(top_k=25, temp=0),
                                      audio_sampler=self._utils.Sampler(top_k=250, temp=0.8))

    @property
    def steps(self) -> int:
        return self.gen.step_idx

    def step(self, block: np.ndarray) -> tuple[str | None, float]:
        """One 1920-sample block -> (word piece or None, P(end of turn))."""
        codes = self.mimi.encode_step(self._mx.array(block, dtype=self._mx.float32)[None, None])
        token, heads = self.gen.step_with_extra_heads(codes.transpose(0, 2, 1)[0, :, :self.cfg.other_codebooks])
        token = token[0].item()
        p_end = heads[2][0, 0, 0].item() if len(heads) > 2 else 0.0
        return (None if token in (0, 3) else self.tok.id_to_piece(token)), p_end

    def transcribe(self, payload: Any) -> str:
        return payload if isinstance(payload, str) else ""


class KyutaiTurns:
    """TurnDetector's interface over KyutaiSTT: feed() takes 16 kHz, 32 ms frames and returns ('speech_start', p) /
    ('utterance', text) / ('discard', None). A turn starts on the first word, or while the assistant speaks on
    BARGE_IN_WORDS words; it ends on the model's end-of-turn head (see END_OF_TURN), or after end_silence_ms
    without a new word (a cap only), or at max_utterance_s."""

    IN_BLOCK = KYUTAI_BLOCK * SR // KYUTAI_SR  # 1280 input samples per 80 ms step

    def __init__(self, stt, tune: Tuning, speaking: Callable[[], bool]):
        self.stt, self.t, self.speaking = stt, tune, speaking
        self.buf = np.zeros(0, np.float32)
        self.prev = 0.0
        self.active = False  # a turn has started (speech_start sent)
        self._clear()

    def _clear(self) -> None:
        self.text, self.words, self.ends, self.started_at, self.last_word = "", 0, 0, None, 0

    def _resample(self, block: np.ndarray) -> np.ndarray:
        x = np.concatenate([[self.prev], block])  # 16 kHz -> 24 kHz, linear, continuous across blocks
        self.prev = float(block[-1])
        t = 1 + np.arange(KYUTAI_BLOCK) * (SR / KYUTAI_SR) - SR / KYUTAI_SR
        return np.interp(t, np.arange(len(x)), x).astype(np.float32)

    def feed(self, frame: np.ndarray) -> list[tuple[str, Any]]:
        self.buf = np.concatenate([self.buf, frame])
        events: list[tuple[str, Any]] = []
        while len(self.buf) >= self.IN_BLOCK:
            block, self.buf = self.buf[: self.IN_BLOCK], self.buf[self.IN_BLOCK:]
            events += self._step(self._resample(block))
        return events

    def _step(self, block: np.ndarray) -> list[tuple[str, Any]]:
        if self.stt.steps >= self.stt.MAX_STEPS - 1 or (self.started_at is None and self.stt.steps > 3000):
            self.stt.reset()
        piece, p_end = self.stt.step(block)
        step = self.stt.steps
        events: list[tuple[str, Any]] = []
        if piece is not None:
            if self.started_at is None:
                self.started_at = step
            self.text += piece.replace("▁", " ")
            self.words += piece.startswith("▁")
            self.last_word = step
            if not self.active and (self.words >= BARGE_IN_WORDS or not self.speaking()):
                self.active = True
                events.append(("speech_start", 1.0 - p_end))
        if self.started_at is None:
            return events
        self.ends = self.ends + 1 if p_end > END_OF_TURN else 0
        quiet_ms = (step - self.last_word) * 1000 * KYUTAI_BLOCK / KYUTAI_SR
        long = (step - self.started_at) * KYUTAI_BLOCK / KYUTAI_SR >= self.t.max_utterance_s
        if self.ends >= self.stt.delay_steps or quiet_ms >= self.t.end_silence_ms or long:
            text = self.text.strip()
            if self.active:
                ok = text.lower() not in JUNK_TRANSCRIPTS
                events.append(("utterance", text) if ok else ("discard", None))
            self.active = False
            self._clear()
        return events


class NoSpeech:
    """A detector that never hears anything: --fake-audio, for the lifecycle self-test."""

    active = False

    def feed(self, frame: np.ndarray) -> list[tuple[str, Any]]:
        return []


# -- TTS: anything with sample_rate and synth(text) -> float32 PCM ------------------------------------------------
def check_tts(kind: str) -> None:
    """Before any loading: say exists only on macOS."""
    if kind == "say" and sys.platform != "darwin":
        raise CannotStart(f"tts 'say' is macOS only. Use kokoro; it needs espeak-ng: {espeak_fix()}.")


def make_tts(kind: str, voice: str):
    """Kokoro, or on macOS say when Kokoro cannot speak; elsewhere a Kokoro failure is fatal, since nothing else
    could read the answers."""
    if kind == "kokoro":
        try:
            t = KokoroTTS(voice or "af_heart")
            t.synth("warm up")  # loads the graph and surfaces a broken phonemizer now, not on the first answer
            return t
        except Exception as e:  # noqa: BLE001
            fix = f" Fix: {espeak_fix()}." if find_espeak() is None else ""
            if sys.platform != "darwin":
                raise CannotStart(f"Kokoro cannot speak ({type(e).__name__}: {str(e)[:120]}).{fix}") from e
            warn(f"Kokoro unavailable ({type(e).__name__}: {str(e)[:120]}); using macOS say.{fix}")
            voice = ""  # a Kokoro voice name means nothing to say
    return SayTTS(voice or None)


def espeak_fix() -> str:
    if sys.platform == "darwin":
        return "brew install espeak-ng"
    if sys.platform == "win32":
        return "install eSpeak NG (github.com/espeak-ng/espeak-ng/releases)"
    return "sudo apt install espeak-ng (Fedora: sudo dnf install espeak-ng)"


def find_espeak() -> tuple[str, str] | None:
    """(library, data dir) of a system espeak-ng. The espeakng-loader wheel Kokoro falls back to ships a library
    whose data path was baked in on its build machine, so a system install is preferred wherever it is."""
    if os.environ.get("PHONEMIZER_ESPEAK_LIBRARY") and os.environ.get("ESPEAK_DATA_PATH"):
        return os.environ["PHONEMIZER_ESPEAK_LIBRARY"], os.environ["ESPEAK_DATA_PATH"]
    exe = shutil.which("espeak-ng")
    roots = [Path("/usr"), Path("/usr/local")]
    if exe:
        roots[:0] = [Path(exe).parent.parent, Path(exe).resolve().parent.parent, Path(exe).parent]
    libs = ["lib/libespeak-ng.dylib", "lib/libespeak-ng.so.1", "lib/x86_64-linux-gnu/libespeak-ng.so.1",
            "lib/aarch64-linux-gnu/libespeak-ng.so.1", "lib64/libespeak-ng.so.1", "libespeak-ng.dll"]
    datas = ["share/espeak-ng-data", "lib/x86_64-linux-gnu/espeak-ng-data", "lib/aarch64-linux-gnu/espeak-ng-data",
             "espeak-ng-data"]
    for root in roots:
        lib = next((root / p for p in libs if (root / p).exists()), None)
        data = next((root / p for p in datas if (root / p / "phontab").exists()), None)
        if lib and data:
            return str(lib), str(data)
    return None


class KokoroTTS:
    """Kokoro-82M via onnx; about 330 MB downloaded on first use."""

    sample_rate = 24_000

    def __init__(self, voice: str = "af_heart", speed: float = 1.05):
        from kokoro_onnx import Kokoro
        from kokoro_onnx.config import EspeakConfig

        model = download("kokoro-v1.0.onnx", KOKORO_URL + "kokoro-v1.0.onnx")
        voices = download("voices-v1.0.bin", KOKORO_URL + "voices-v1.0.bin")
        found = find_espeak()
        log(f"tts: Kokoro voice {voice}, espeak-ng {found[0] if found else 'from the espeakng-loader wheel'}")
        self.k = Kokoro(str(model), str(voices), espeak_config=EspeakConfig(*found) if found else None)
        self.voice, self.speed = voice, speed

    def synth(self, text: str) -> np.ndarray:
        samples, sr = self.k.create(text, voice=self.voice, speed=self.speed, lang="en-us")
        if int(sr) != self.sample_rate:
            raise RuntimeError(f"Kokoro returned {sr} Hz, expected {self.sample_rate}")
        return np.asarray(samples, dtype=np.float32)


class SayTTS:
    """macOS's built-in synthesizer, rendered to a WAV so playback is ours to interrupt."""

    sample_rate = 22_050

    def __init__(self, voice: str | None = None, rate: int = 190):
        self.voice, self.rate = voice, rate

    def synth(self, text: str) -> np.ndarray:
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        cmd = ["say", "-o", path, "--file-format=WAVE", f"--data-format=LEI16@{self.sample_rate}", "-r", str(self.rate)]
        if self.voice:
            cmd += ["-v", self.voice]
        try:
            subprocess.run(cmd, input=text.encode(), check=True, timeout=60, capture_output=True)
            with wave.open(path) as w:
                raw = w.readframes(w.getnframes())
            return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


class SilentTTS:
    """--fake-audio and the self-test: a short silence per sentence, so turns and their history still complete."""

    sample_rate = 16_000

    def synth(self, text: str) -> np.ndarray:
        return np.zeros(int(self.sample_rate * min(2.0, 0.05 * max(1, len(text.split())))), np.float32)

