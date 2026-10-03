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
from typing import Any, Callable, NamedTuple

import numpy as np

from . import gpu
from .audio import (JUNK_TRANSCRIPTS, SR, BargeGate, CannotStart, Detector, EnergyVAD, SileroVAD, Tuning, TurnDetector,
                    is_backchannel)
from .protocol import file_log, log, warn

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
def recognizer(stt: str, asr: str, end_silence_ms: int, speaking: Callable[[], bool],
               end_silence_long_ms: int = Tuning.end_silence_long_ms,
               guard: Any = None, experimental: bool = False) -> tuple[Detector, Callable[[Any], str]]:
    """(detector, transcribe). Kyutai brings its own end of turn and passes text through; Whisper pairs with
    Silero (or the energy gate) and transcribes the utterance audio. `guard` (an EchoGuard) lets the barge-in gate
    cut on one word. `experimental` turns on Kyutai's continuous end-of-turn wait and STT flush."""
    tune = Tuning(end_silence_ms=end_silence_ms, end_silence_long_ms=end_silence_long_ms,
                  eot_continuous=experimental, stt_flush=experimental)
    gate = BargeGate(tune, guard)
    try:
        if stt == "kyutai":
            from .kyutai_cuda import load_kyutai

            k = load_kyutai()  # MLX on Apple silicon, CUDA PyTorch on an Nvidia GPU, or None
            if k is not None:
                return KyutaiTurns(k, tune, speaking, gate), k.transcribe
        w = WhisperASR(asr)
        return TurnDetector(load_vad(), tune, speaking, gate, w.transcribe), w.transcribe
    finally:
        gpu.release("stt")  # loaded (or not): free VRAM readings show what it took from here on


def stt_gpu_need(stt: str) -> float:
    """VRAM the recognizer may take, held for it (gpu.hold) before the synthesizer loads beside it."""
    if is_apple_silicon():
        return 0.0
    return gpu.NEED_GIB["kyutai" if stt == "kyutai" else "whisper"]


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
        why = gpu.admit("stt", gpu.NEED_GIB["whisper"]) if _cuda_devices() > 0 else ""
        if _cuda_devices() > 0 and not why:
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
        log(f"asr: faster-whisper {size} on CPU int8" + (f"; not the GPU: {why}" if why else ""))

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
BARGE_IN_WORDS = 3  # the old barge-in rule, still the fallback when the echo guard cannot judge (BargeGate)
END_OF_TURN = 0.6  # pause-head threshold, as in Kyutai's Unmute. Extra heads 0..3 forecast "no new word within
# 0.5 / 1 / 2 / 3 s" (lower index: more aggressive); they are not silence detectors, so the 2 s head alone rises
# before the user stops and cut turns mid-sentence, the trailing pieces (text trails audio by ~0.5 s) arriving as a
# second utterance. A semantic end needs the 2 s and the 0.5 s heads both above this for delay_steps steps in a row
# with no new piece (each piece restarts the count) and eot_drain_ms since the last piece; see KyutaiTurns.
HEADS_IGNORE_STEPS = 12  # the heads jitter for the first steps after a model reset (Unmute ignores these too)
EOT_CONTINUE = 0.4  # below this the 2 s head forecasts more speech: the silence cap stretches to end_silence_long_ms
S2_EMA_KEEP = 0.5  # the cap tier reads the 2 s head through an EMA (about 3 steps), so one jittery step cannot pick it
CONTINUE_WORDS = frozenset(("and", "but", "or", "so", "because", "yet", "the", "a", "an", "to", "of", "in", "on",
                            "with", "like", "um", "uh"))  # a transcript ending on one of these is unfinished
TERMINAL = {".": "period", "?": "question", "!": "exclaim"}
# Tuning.eot_continuous: the shaped end-of-turn probability p behind the one wait (KyutaiTurns._p).
P_UNFINISHED = 0.3  # an unfinished tail (comma, CONTINUE_WORDS) caps p here
P_TERMINAL = 0.7  # terminal punctuation floors p here
P_NO_HEADS = 0.5  # a model without pause heads: neither "more coming" nor "done"; the text alone moves p
FLUSH_BLOCK = np.zeros(KYUTAI_BLOCK, np.float32)  # Tuning.stt_flush: 80 ms of silence; every backend's step() takes it
MAX_EXTEND_STEP_S = 10  # past max_utterance_s, a turn still forecast to go on is re-judged this often


def turn_tail(text: str) -> str:
    """How a transcript ends, as one token: 'comma', a TERMINAL class, 'ellipsis' (trailing off: neither finished
    nor unfinished), a CONTINUE_WORDS word, 'word' for any other word, or 'none'. Kyutai's English output carries
    punctuation ('are you running?'); without it only the word list applies, so the punctuation rules are no-ops."""
    text = text.rstrip()
    if not text:
        return "none"
    if text.endswith(("...", "\u2026")):
        return "ellipsis"
    if text[-1] == ",":
        return "comma"
    if text[-1] in TERMINAL:
        return TERMINAL[text[-1]]
    last = text.split()[-1].strip("\"'()[]:;-").lower()
    return last if last in CONTINUE_WORDS else "word"


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


class Pauses(NamedTuple):
    """Kyutai's four pause predictors for one step (extra heads 0..3, class 0): the probability that no new word
    comes within the next 0.5, 1, 2 and 3 s. A forecast, not a silence detector; shorter horizons are more
    aggressive. All 0.0 when the model lacks a head."""

    s05: float = 0.0
    s1: float = 0.0
    s2: float = 0.0
    s3: float = 0.0

    @classmethod
    def of(cls, values: list[float]) -> Pauses:
        """The first four head values, zero-padded."""
        return cls(*(list(values)[:4] + [0.0] * (4 - min(4, len(values)))))


class KyutaiSTT:
    """Kyutai STT 1B: 80 ms of 24 kHz audio in, at most one word piece and the pause probabilities out.
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

    def step(self, block: np.ndarray) -> tuple[str | None, Pauses]:
        """One 1920-sample block -> (word piece or None, pause probabilities). A zero block (FLUSH_BLOCK, the
        flush) is an ordinary step: mx.array copies the input."""
        mx = self._mx
        codes = self.mimi.encode_step(mx.array(block, dtype=mx.float32)[None, None])
        token, heads = self.gen.step_with_extra_heads(codes.transpose(0, 2, 1)[0, :, :self.cfg.other_codebooks])
        p = [h[0, 0, 0] for h in heads[:4]]
        mx.eval(token, *p)  # one evaluation for the token and the heads
        token = token[0].item()
        return (None if token in (0, 3) else self.tok.id_to_piece(token)), Pauses.of([x.item() for x in p])

    def transcribe(self, payload: Any) -> str:
        return payload if isinstance(payload, str) else ""


class KyutaiTurns:
    """TurnDetector's interface over KyutaiSTT: feed() takes 16 kHz, 32 ms frames and returns ('speech_start', p) /
    ('utterance', text) / ('discard', None). A turn starts on the first word; while the assistant speaks, on the
    word piece at which the BargeGate says cut (Tuning.barge_in_words words the echo guard does not place in what the
    speaker just played, or BARGE_IN_WORDS when it cannot judge). It ends on one of three triggers:
    - semantic: the 2 s pause head (s2) and the 0.5 s head (s05) both above END_OF_TURN for delay_steps
      consecutive steps (6 = 480 ms), counted from the last word piece, and no piece for eot_drain_ms. The 2 s head
      forecasts the end; the 0.5 s head confirms a pause is under way now (Kyutai's advice for mid-sentence
      cutoffs, delayed-streams-modeling issue 23); the drain keeps trailing pieces in the same utterance. The
      heads count for nothing in the first HEADS_IGNORE_STEPS steps after a model reset.
    - silence: no new piece for the cap of the tier in force, re-read every step:
        long   (end_silence_long_ms): the 2 s head (S2_EMA_KEEP EMA) below EOT_CONTINUE, the model forecasting
               more speech, or the text unfinished (turn_tail: a comma or a CONTINUE_WORDS word), whatever s2 says;
        short  (end_silence_short_ms): the text ends on . ? ! and the head is not below EOT_CONTINUE;
        normal (end_silence_ms): otherwise, the model unsure (0.4 <= s2 < 0.6, or above 0.6 but not held).
      With no heads at all (every peak 0) only the text picks the tier. Long is never below normal nor above
      max_utterance_s; short is never above normal. The industry's tiers: LiveKit 0.3 s / 2.5 s on
      P(end) < unlikely_threshold, OpenAI semantic_vad medium 4 s, AssemblyAI conservative 3.6 s, Pipecat Smart
      Turn 3 s fallback.
    - max: max_utterance_s, if the end-of-turn probability (_p) is at least END_OF_TURN or the model has no
      heads; otherwise the limit moves MAX_EXTEND_STEP_S at a time, up to max_utterance_extend_s, where the turn
      ends regardless. The log line carries extended=<s past max_utterance_s>.
    - backchannel: a turn that cut the assistant and is only backchannel words so far (audio.is_backchannel) ends
      after backchannel_quiet_ms without a new piece, so the paused voice can resume.
    Each end writes one 'kyutai turn end' line to the log file (trigger, cap tier, tail token, length, heads; no
    other text).

    Tuning.eot_continuous (experimental) replaces the three silence tiers with one wait, re-read every step:
    wait_ms = eot_wait_base_ms + (1 - p) * eot_wait_span_ms, at most max_utterance_s; the turn ends (trigger
    silence, cap continuous) once quiet_ms >= wait_ms. p is the s2 EMA clamped to [0, 1] (P_NO_HEADS without
    heads), shaped by the text and the 0.5 s head (_p). The semantic trigger stays as the early exit.

    Tuning.stt_flush (experimental): at the first step of a high-heads run (s2 and s05 above END_OF_TURN, no
    piece) in an active turn, _flush steps delay_steps + 1 silent blocks at once; the triggers above then read
    that silence as quiet until the next piece. A flushed piece aborts it. The log line gains
    flush=<steps>/<compute ms> and quiet_ms includes the flushed silence."""

    IN_BLOCK = KYUTAI_BLOCK * SR // KYUTAI_SR  # 1280 input samples per 80 ms step
    MS_PER_STEP = 1000 * KYUTAI_BLOCK / KYUTAI_SR  # 80

    def __init__(self, stt, tune: Tuning, speaking: Callable[[], bool], gate: BargeGate | None = None):
        self.stt, self.t, self.speaking = stt, tune, speaking
        self.gate = gate or BargeGate(tune)  # no guard: it cannot judge echo, so BARGE_IN_WORDS
        self.buf = np.zeros(0, np.float32)
        self.prev = 0.0
        self.active = False  # a turn has started (speech_start sent)
        self.skew = 0  # flushed model steps since the last model reset (_flush)
        self._clear()

    def _clear(self) -> None:
        self.text, self.words, self.ends, self.started_at, self.last_word = "", 0, 0, None, 0
        self.barged = False  # this turn's speech_start cut the assistant
        self.gate.reset()
        self.peak = Pauses()
        self.s2_ema: float | None = None
        self.extended = 0.0  # seconds this turn runs past max_utterance_s
        self.flushed, self.flush_ms = 0, 0.0  # this turn's flushed steps and their compute time
        self.credit = 0  # flushed steps since the last piece: silence the model heard, counted as quiet
        self.flush_armed = True  # one completed flush per pause; a piece re-arms it

    def _cap(self) -> tuple[str, float, str]:
        """(tier, cap in ms, turn_tail) for the silence cap now; see the class docstring."""
        t, tail = self.t, turn_tail(self.text)
        s2 = self.s2_ema if any(self.peak) else None  # no heads: the text alone picks the tier
        if tail == "comma" or tail in CONTINUE_WORDS or (s2 is not None and s2 < EOT_CONTINUE):
            return "long", min(max(t.end_silence_long_ms, t.end_silence_ms), t.max_utterance_s * 1000), tail
        if tail in TERMINAL.values():
            return "short", min(t.end_silence_short_ms, t.end_silence_ms), tail
        return "normal", t.end_silence_ms, tail

    def _p(self, s05: float) -> float:
        """The end-of-turn probability for the continuous wait: the s2 EMA clamped to [0, 1]; an unfinished tail
        caps it at P_UNFINISHED, terminal punctuation floors it at P_TERMINAL; above END_OF_TURN it also needs the
        0.5 s head above END_OF_TURN, else it reads END_OF_TURN."""
        p = min(1.0, max(0.0, self.s2_ema)) if any(self.peak) and self.s2_ema is not None else P_NO_HEADS
        tail = turn_tail(self.text)
        if tail == "comma" or tail in CONTINUE_WORDS:
            p = min(p, P_UNFINISHED)
        elif tail in TERMINAL.values():
            p = max(p, P_TERMINAL)
        return END_OF_TURN if p > END_OF_TURN and s05 <= END_OF_TURN else p

    def _wait(self, p: float) -> int:
        """The continuous wait in ms for end-of-turn probability p, never above max_utterance_s."""
        t = self.t
        return min(round(t.eot_wait_base_ms + (1 - p) * t.eot_wait_span_ms), round(t.max_utterance_s * 1000))

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
            self.skew = 0
        piece, pauses = self.stt.step(block)
        step = self.stt.steps - self.skew  # real time: flushed steps took none
        if self.stt.steps <= HEADS_IGNORE_STEPS:
            pauses = Pauses()
        events: list[tuple[str, Any]] = []
        if piece is not None:
            if self.started_at is None:
                self.started_at = step
            self.text += piece.replace("▁", " ")
            self.words += piece.startswith("▁")
            self.last_word, self.credit, self.flush_armed = step, 0, True
            if not self.active:
                speaking = self.speaking()
                if not speaking or self.gate.decide(self.text, self.words) == "cut":
                    self.active, self.barged = True, speaking
                    events.append(("speech_start", 1.0 - pauses.s2))
        if self.started_at is None:
            return events
        self._heads(piece, pauses)
        if (self.t.stt_flush and self.flush_armed and self.active and piece is None and self.ends == 1
                and self.stt.steps + self.stt.delay_steps + 1 < self.stt.MAX_STEPS - 1):
            pauses = self._flush(step)
        quiet_ms = (step - self.last_word + self.credit) * self.MS_PER_STEP
        cap, cap_ms, tail = self._cap()
        p = wait_ms = None
        if self.t.eot_continuous:
            p = self._p(pauses.s05)
            wait_ms = self._wait(p)
            cap, cap_ms = "continuous", wait_ms
        if self.ends >= self.stt.delay_steps and quiet_ms >= self.t.eot_drain_ms:
            trigger, cap = "semantic", "semantic"
        elif self.barged and quiet_ms >= self.t.backchannel_quiet_ms and is_backchannel(self.text):
            trigger, cap = "backchannel", "backchannel"
        elif quiet_ms >= cap_ms:
            trigger = "silence"
        elif (step - self.started_at) * self.MS_PER_STEP >= (self.t.max_utterance_s + self.extended) * 1000:
            room = max(self.t.max_utterance_s, self.t.max_utterance_extend_s) - self.t.max_utterance_s - self.extended
            if room > 0 and any(self.peak) and (p if p is not None else self._p(pauses.s05)) < END_OF_TURN:
                self.extended += min(MAX_EXTEND_STEP_S, room)  # still speaking: no mid-sentence cut yet
                return events
            trigger = "max"
        else:
            return events
        text = self.text.strip()
        sent = "none"
        if self.active:
            ok = text.lower() not in JUNK_TRANSCRIPTS
            events.append(("utterance", text) if ok else ("discard", text or None))  # a session may want "okay."
            sent = events[-1][0]
        heads = " ".join(f"{k}={pk:.2f}/{v:.2f}" for k, pk, v in zip(Pauses._fields, self.peak, pauses))
        ext = f" extended={self.extended:.0f}" if trigger == "max" else ""
        cont = f" wait_ms={wait_ms} p={p:.2f}" if p is not None else ""
        flush = f" flush={self.flushed}/{self.flush_ms:.0f}ms" if self.t.stt_flush else ""
        file_log("INFO", f"kyutai turn end: trigger={trigger}{ext} cap={cap}{cont} tail={tail} sent={sent} "
                         f"len_ms={(step - self.started_at + 1) * self.MS_PER_STEP:.0f} quiet_ms={quiet_ms:.0f}{flush} "
                         f"words={self.words} peak/final {heads}")
        self.active = False
        self._clear()
        return events

    def _heads(self, piece: str | None, pauses: Pauses) -> None:
        """One step's pause heads into the turn: peak, the s2 EMA, and the run of high steps without a piece."""
        self.peak = Pauses(*map(max, self.peak, pauses))
        self.s2_ema = pauses.s2 if self.s2_ema is None else S2_EMA_KEEP * self.s2_ema + (1 - S2_EMA_KEEP) * pauses.s2
        high = pauses.s2 > END_OF_TURN and pauses.s05 > END_OF_TURN
        self.ends = self.ends + 1 if high and piece is None else 0  # a new piece restarts the count

    def _flush(self, step: int) -> Pauses:
        """Tuning.stt_flush (Kyutai Unmute's flush): the semantic end just became plausible, so step delay_steps + 1
        silent blocks now, at compute speed, to push out the text that trails the audio, and return the last
        flushed step's heads. Their heads join the turn, and the flushed steps count as quiet (`credit`) until the
        next piece, as if that much silence had passed. A flushed piece means the user was not done: the flush
        stops there, the piece stays in the text, stamped at the current real step (quiet restarts from now, no
        credit), and a 'kyutai flush' line is logged. One completed flush per pause: a piece re-arms it.

        Model time and real time: flushed steps advance stt.steps but take no real time, so `skew` counts them
        until the next model reset and the turn's clock (len_ms, the max timer, last_word) reads stt.steps - skew.
        Mic audio that arrives meanwhile waits in the Listener's queue and in self.buf and is fed after, so the
        model hears [audio, flushed silence, audio]: nothing is dropped."""
        n, pauses, ran, cut = self.stt.delay_steps + 1, Pauses(), 0, False
        t0 = time.perf_counter()
        for ran in range(1, n + 1):
            piece, pauses = self.stt.step(FLUSH_BLOCK)
            if self.stt.steps <= HEADS_IGNORE_STEPS:
                pauses = Pauses()
            self._heads(piece, pauses)
            if piece is not None:
                self.text += piece.replace("▁", " ")
                self.words += piece.startswith("▁")
                cut = True
                break
        ms = (time.perf_counter() - t0) * 1000
        self.skew += ran
        self.flushed += ran
        self.flush_ms += ms
        if cut:
            self.last_word, self.credit = step, 0
            file_log("INFO", f"kyutai flush: aborted at step {ran}/{n} ms={ms:.0f}")
        else:
            self.credit += ran
            self.flush_armed = False
        return pauses

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


def make_tts(kind: str, voice: str, device: str = "auto"):
    """Kokoro, or on macOS say when Kokoro cannot speak; elsewhere a Kokoro failure is fatal, since nothing else
    could read the answers. Kokoro that fails on CUDA is retried on the CPU."""
    if kind == "kokoro":
        try:
            try:
                t = KokoroTTS(voice or "af_heart", device=device)
                try:
                    t.synth("warm up")  # loads the graph and surfaces a broken phonemizer now, not on the first answer
                except Exception as e:  # noqa: BLE001
                    if t.provider != "CUDAExecutionProvider":
                        raise
                    log(f"tts: Kokoro failed on CUDA ({type(e).__name__}: {str(e)[:120]}); using the CPU")
                    t = KokoroTTS(voice or "af_heart", device="cpu")
                    t.synth("warm up")
            finally:
                gpu.release("tts")  # its memory shows in free VRAM readings from here on
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


KOKORO_CPU_THREADS = 4  # measured on a Ryzen 9700X: as fast as all 16, and leaves cores for the mic and player
# Measured on an RTX 5090 (WSL2, onnxruntime-gpu 1.30): RTF ~0.02 on CUDA vs ~0.13 on 4 CPU threads, ~1.4 GiB of VRAM,
# so the CPU is the fallback whenever the GPU lacks room (gpu.admit). Exhaustive cuDNN
# search costs little here (first call per length within 2x), and kSameAsRequested keeps the arena from doubling.
KOKORO_CUDA = {"arena_extend_strategy": "kSameAsRequested", "cudnn_conv_use_max_workspace": "0"}


def kokoro_session(model: str, device: str) -> tuple[Any, str]:
    """An onnxruntime session for Kokoro: CUDA when `device` allows it, onnxruntime-gpu sees a GPU (its cuDNN and
    cuBLAS come from the nvidia wheels, loaded by preload_dlls) and the GPU has room, else the CPU."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = KOKORO_CPU_THREADS
    cuda = device != "cpu" and "CUDAExecutionProvider" in ort.get_available_providers()
    why = gpu.admit("tts", gpu.NEED_GIB["kokoro"]) if cuda else ""
    if cuda and not why:
        try:
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
            return ort.InferenceSession(model, so, providers=[("CUDAExecutionProvider", KOKORO_CUDA),
                                                              "CPUExecutionProvider"]), ""
        except Exception as e:  # noqa: BLE001
            gpu.release("tts")
            why = f"CUDA failed ({type(e).__name__}: {str(e)[:160]})"
    elif device == "cuda" and not cuda:
        why = f"no CUDA in this onnxruntime ({', '.join(ort.get_available_providers())})"
    return ort.InferenceSession(model, so, providers=["CPUExecutionProvider"]), why


class KokoroTTS:
    """Kokoro-82M via onnx; about 330 MB downloaded on first use. `device`: auto (CUDA when available), cuda, or
    cpu."""

    sample_rate = 24_000

    def __init__(self, voice: str = "af_heart", speed: float = 1.05, device: str = "auto"):
        from kokoro_onnx import Kokoro
        from kokoro_onnx.config import EspeakConfig

        model = download("kokoro-v1.0.onnx", KOKORO_URL + "kokoro-v1.0.onnx")
        voices = download("voices-v1.0.bin", KOKORO_URL + "voices-v1.0.bin")
        found = find_espeak()
        espeak = EspeakConfig(*found) if found else None
        sess, why = kokoro_session(str(model), device)
        self.provider = sess.get_providers()[0]
        log(f"tts: Kokoro voice {voice} on {self.provider}" + (f" (not the GPU: {why})" if why else "")
            + f", espeak-ng {found[0] if found else 'from the espeakng-loader wheel'}")
        self.k = Kokoro.from_session(sess, str(voices), espeak_config=espeak)
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

