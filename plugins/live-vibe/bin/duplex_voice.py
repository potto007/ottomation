#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "faster-whisper>=1.2",
#   "sounddevice>=0.5",
#   "onnxruntime>=1.20",
#   "numpy>=2",
#   "httpx>=0.28",
#   "anthropic>=1.11",
#   "kokoro-onnx>=0.6",
# ]
# ///
# Vendored unchanged from proto/duplex_voice.py at e82ae68 (ClearBridgeRIP/claude-plugins), so the installed
# plugin carries it. live_sidecar.py imports it as a library; platform handling lives there.
"""PROTOTYPE (throwaway): full duplex speech interface.

Question this prototype answers: does a cascaded local pipeline
(VAD -> ASR -> LLM -> TTS) feel full duplex when the microphone stays open
during playback, the user can barge in and cut the assistant off mid-sentence,
and background tasks report back without blocking the conversation?

Informed by proto/live_voice_mode.md and
proto/recreate_codex_voice_mode_using_local_asr_model.md.

Run (one command, uv fetches Python 3.12 and deps on first run):

    uv run proto/duplex_voice.py                 # auto-pick LLM backend
    uv run proto/duplex_voice.py --selftest      # no mic, no model, no network
    uv run proto/duplex_voice.py --llm scripted  # duplex mechanics only
    uv run proto/duplex_voice.py --llm llamacpp  # llama-server on :8080 (--jinja)
    uv run proto/duplex_voice.py --llm anthropic # needs ANTHROPIC_API_KEY
    uv run proto/duplex_voice.py --list-devices

Pipeline:  mic (16 kHz, 32 ms frames)
            -> Silero VAD (onnx) with pre-roll ring buffer and endpointing
            -> faster-whisper (CPU int8)
            -> LLM backend streaming text + tool calls
            -> sentence splitter -> TTS (Kokoro-82M via onnx; macOS say fallback) -> speaker

Duplex rules implemented:
  * The mic never closes. While the assistant speaks, VAD runs with a stricter
    barge-in threshold (higher probability, sustained longer) so the speaker
    bleed does not trigger it. Headphones make this much more reliable.
  * Barge-in stops playback, cancels the LLM stream, flushes queued speech,
    and records only what was actually spoken into the history, marked as
    cut off, so the model does not repeat itself.
  * Tool calls that take time run as background tasks. The model is told
    "queued" immediately and keeps talking. When a task finishes, its result is
    announced only when the floor is free (nobody is talking).
  * Every state change prints on one line so the behaviour is visible.

Local model (from ~/src/ai/ftl/bench/spec-decoding/README.md, M5 Pro, 2026-09-30):
  Qwen3.6-35B-A3B, bartowski Q4_0 GGUF, llama.cpp Metal, DFlash2 drafter at
  3 drafted tokens. That was the pick: 95 tok/s decode (1.35x over plain),
  0.9 s to first token, best time per turn. Q4_0 decodes 26% faster than
  UD-Q4_K_XL / MXFP4 on Metal for ~1% worse perplexity (5.63 vs 5.57).
  Sampling that the bench used and this script sends: temperature 0.6,
  top_p 0.95, top_k 20, min_p 0, presence_penalty 0. Thinking is turned off
  per request because a voice turn cannot wait for it.

    ~/src/ai/ftl/bench/spec-decoding/start_llama.sh dflash2 3   # Q4_0 target, q8_0 KV default, :8080, --jinja
    uv run proto/duplex_voice.py --llm llamacpp

TTS: Kokoro-82M (kokoro-onnx, ~330 MB download on first run) is the default and needs
espeak-ng for phonemes: `brew install espeak-ng`. The espeakng-loader wheel's bundled
library looks for its data at a path baked in on the build machine, so the script
prefers the Homebrew library. `--tts say --voice Samantha` is the zero-download fallback.

Use headphones for the best result. Say "goodbye" to exit, or press Ctrl-C.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable

import numpy as np

SR = 16_000            # mic sample rate
FRAME = 512            # 32 ms at 16 kHz, the chunk size Silero v5 expects
CACHE = Path(os.environ.get("DUPLEX_VOICE_CACHE", "~/.cache/duplex_voice")).expanduser()

SILERO_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
KOKORO_URLS = {
    "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
}

T0 = time.monotonic()


def log(tag: str, msg: str = "") -> None:
    print(f"[{time.monotonic() - T0:7.2f}s] {tag:<12} {msg}", flush=True)


def ensure_download(path: Path, url: str) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    log("download", f"{url} -> {path}")
    tmp = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)
    return path


# --------------------------------------------------------------------------
# 1. Voice activity detection
# --------------------------------------------------------------------------
class SileroVAD:
    """Silero VAD v5 via onnxruntime. Call per 512-sample float32 frame; returns P(speech)."""

    def __init__(self, path: Path):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
        names = {i.name for i in self.sess.get_inputs()}
        if not {"input", "state", "sr"} <= names:
            raise RuntimeError(f"unexpected Silero model inputs {names}; expected v5 (input, state, sr)")
        self.reset()

    def reset(self) -> None:
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, 64), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        x = np.concatenate([self.context, frame[None, :]], axis=1).astype(np.float32)
        out, self.state = self.sess.run(None, {"input": x, "state": self.state, "sr": np.array(SR, dtype=np.int64)})
        self.context = x[:, -64:]
        return float(out[0, 0])


class EnergyVAD:
    """Fallback when the Silero model cannot be fetched: adaptive RMS gate."""

    def __init__(self):
        self.noise = 2e-3

    def reset(self) -> None:
        pass

    def __call__(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(frame * frame)) + 1e-9)
        if rms < self.noise * 2:
            self.noise = 0.98 * self.noise + 0.02 * rms
        ratio = rms / (self.noise + 1e-6)
        return float(min(1.0, max(0.0, (ratio - 2.0) / 6.0)))


def load_vad(prefer_energy: bool = False):
    if prefer_energy:
        return EnergyVAD()
    try:
        return SileroVAD(ensure_download(CACHE / "silero_vad.onnx", SILERO_URL))
    except Exception as e:  # noqa: BLE001 - prototype: degrade, do not die
        log("vad", f"Silero unavailable ({e}); falling back to energy VAD")
        return EnergyVAD()


# --------------------------------------------------------------------------
# 2. Turn detection (endpointing + barge-in)
# --------------------------------------------------------------------------
@dataclass
class Tuning:
    start_prob: float = 0.5      # P(speech) to open a turn when the floor is free
    start_frames: int = 3        # sustained frames (96 ms)
    bargein_prob: float = 0.8    # stricter while the assistant is speaking
    bargein_frames: int = 8      # 256 ms sustained: speaker bleed rarely does this
    end_prob: float = 0.35       # below this counts as silence
    end_silence_ms: int = 700    # trailing silence that ends a turn
    min_speech_ms: int = 250     # shorter utterances are dropped (clicks, coughs)
    pre_roll_ms: int = 320       # audio kept from before the trigger
    max_utterance_s: float = 30.0


class TurnDetector:
    """Feeds VAD frame by frame and emits ('speech_start', p) / ('utterance', audio) / ('discard', None)."""

    def __init__(self, vad, tune: Tuning, assistant_speaking: Callable[[], bool]):
        self.vad, self.t, self.speaking = vad, tune, assistant_speaking
        ms_per_frame = 1000 * FRAME / SR
        self.end_frames = int(tune.end_silence_ms / ms_per_frame)
        self.min_frames = int(tune.min_speech_ms / ms_per_frame)
        self.max_frames = int(tune.max_utterance_s * SR / FRAME)
        self.pre: collections.deque = collections.deque(maxlen=int(tune.pre_roll_ms / ms_per_frame))
        self.active = False
        self.buf: list[np.ndarray] = []
        self.run = 0
        self.silence = 0
        self.last_p = 0.0

    def feed(self, frame: np.ndarray) -> list[tuple[str, Any]]:
        p = self.last_p = self.vad(frame)
        events: list[tuple[str, Any]] = []
        if not self.active:
            self.pre.append(frame)
            if self.speaking():
                need_p, need_n = self.t.bargein_prob, self.t.bargein_frames
            else:
                need_p, need_n = self.t.start_prob, self.t.start_frames
            self.run = self.run + 1 if p >= need_p else 0
            if self.run >= need_n:
                self.active, self.run, self.silence = True, 0, 0
                self.buf = list(self.pre)
                events.append(("speech_start", p))
        else:
            self.buf.append(frame)
            self.silence = self.silence + 1 if p < self.t.end_prob else 0
            if self.silence >= self.end_frames or len(self.buf) >= self.max_frames:
                audio = np.concatenate(self.buf)
                voiced = len(self.buf) - self.silence
                self.active, self.buf = False, []
                self.vad.reset()
                if voiced >= self.min_frames:
                    events.append(("utterance", audio))
                else:
                    events.append(("discard", None))
        return events


# --------------------------------------------------------------------------
# 3. ASR
# --------------------------------------------------------------------------
JUNK_TRANSCRIPTS = {"", ".", "you", "thank you.", "thanks.", "thank you", "bye.", "okay."}


class WhisperASR:
    def __init__(self, size: str = "base.en"):
        from faster_whisper import WhisperModel

        log("asr", f"loading faster-whisper {size} (cpu/int8)")
        self.model = WhisperModel(size, device="cpu", compute_type="int8")
        self.lang = "en" if size.endswith(".en") else None

    def transcribe(self, audio: np.ndarray) -> str:
        segs, _ = self.model.transcribe(
            audio, beam_size=1, language=self.lang, vad_filter=False, condition_on_previous_text=False
        )
        text = " ".join(s.text.strip() for s in segs).strip()
        return "" if text.lower() in JUNK_TRANSCRIPTS else text


# --------------------------------------------------------------------------
# 4. TTS + player
# --------------------------------------------------------------------------
class SayTTS:
    """macOS built-in synthesizer. Rendered to a WAV so playback is ours to interrupt."""

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


class KokoroTTS:
    """Kokoro-82M via onnx. ~330 MB download on first use."""

    ESPEAK_CANDIDATES = [  # the espeakng-loader wheel ships a library with a baked-in CI data path
        ("/opt/homebrew/lib/libespeak-ng.dylib", "/opt/homebrew/share/espeak-ng-data"),
        ("/usr/local/lib/libespeak-ng.dylib", "/usr/local/share/espeak-ng-data"),
        ("/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1", "/usr/lib/x86_64-linux-gnu/espeak-ng-data"),
    ]

    def __init__(self, voice: str = "af_heart", speed: float = 1.05):
        from kokoro_onnx import Kokoro
        from kokoro_onnx.config import EspeakConfig

        model = ensure_download(CACHE / "kokoro-v1.0.onnx", KOKORO_URLS["kokoro-v1.0.onnx"])
        voices = ensure_download(CACHE / "voices-v1.0.bin", KOKORO_URLS["voices-v1.0.bin"])
        espeak = next((EspeakConfig(lib, data) for lib, data in self.ESPEAK_CANDIDATES
                       if os.path.exists(lib) and os.path.exists(os.path.join(data, "phontab"))), None)
        log("tts", f"Kokoro voice {voice}, espeak-ng: {espeak.lib_path if espeak else 'bundled loader'}")
        self.k = Kokoro(str(model), str(voices), espeak_config=espeak)
        self.voice, self.speed = voice, speed
        self.sample_rate = 24_000

    def synth(self, text: str) -> np.ndarray:
        samples, sr = self.k.create(text, voice=self.voice, speed=self.speed, lang="en-us")
        self.sample_rate = int(sr)
        return np.asarray(samples, dtype=np.float32)


class NullTTS:
    """Self-test helper: returns a short tone so the player path is exercised without audio devices."""

    sample_rate = 16_000

    def synth(self, text: str) -> np.ndarray:
        n = int(self.sample_rate * min(2.0, 0.05 * max(1, len(text.split()))))
        t = np.arange(n) / self.sample_rate
        return (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


class Player:
    """Owns the output stream. play() blocks (run it in a thread) and reports how much was heard."""

    def __init__(self, sample_rate: int, device: int | None = None, enabled: bool = True):
        self.sample_rate, self.enabled = sample_rate, enabled
        self.stop = threading.Event()
        self.busy = threading.Event()
        self.stream = None
        if enabled:
            import sounddevice as sd

            self.stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32", latency="low", device=device)
            self.stream.start()

    def play(self, audio: np.ndarray) -> int:
        """Returns the number of samples actually played (== len(audio) when not cut off)."""
        self.busy.set()
        try:
            step = int(self.sample_rate * 0.03)
            for i in range(0, len(audio), step):
                if self.stop.is_set():
                    if self.stream is not None:
                        self.stream.abort()   # drop what is buffered, cut is immediate
                        self.stream.start()
                    return i
                chunk = np.ascontiguousarray(audio[i:i + step])
                if self.stream is not None:
                    self.stream.write(chunk)
                else:
                    time.sleep(len(chunk) / self.sample_rate)
            return len(audio)
        finally:
            self.busy.clear()

    def close(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()


_SENTENCE_END = re.compile(r'(?<=[.!?])["\')\]]*\s+')
_MD_NOISE = re.compile(r"[*_`#>]+")


class SentenceSplitter:
    """Accumulates streamed text and yields speakable sentences as soon as they close."""

    def __init__(self, max_chars: int = 220):
        self.buf, self.max_chars = "", max_chars

    def push(self, delta: str) -> list[str]:
        self.buf += delta
        out: list[str] = []
        while True:
            m = _SENTENCE_END.search(self.buf)
            if m:
                out.append(self.buf[: m.end()].strip())
                self.buf = self.buf[m.end():]
                continue
            if len(self.buf) > self.max_chars:
                cut = self.buf.rfind(", ", 0, self.max_chars)
                cut = cut + 1 if cut > 40 else self.max_chars
                out.append(self.buf[:cut].strip())
                self.buf = self.buf[cut:]
                continue
            break
        return [clean_for_speech(s) for s in out if clean_for_speech(s)]

    def flush(self) -> list[str]:
        s, self.buf = clean_for_speech(self.buf), ""
        return [s] if s else []


def clean_for_speech(s: str) -> str:
    return _MD_NOISE.sub("", s).replace("\n", " ").strip()


# --------------------------------------------------------------------------
# 5. Tools (background tasks) shared by every LLM backend
# --------------------------------------------------------------------------
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "run_background_task",
        "description": (
            "Start a shell command in the workspace as a background task and return immediately with a task id. "
            "You will receive a message when it finishes. Use it for anything that takes more than a moment: "
            "tests, builds, searches, scripts. Keep talking after calling it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Exact shell command to run."},
                "label": {"type": "string", "description": "Three to six word label for the task."},
            },
            "required": ["command", "label"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_dir",
        "description": "List the entries of a directory inside the workspace (fast, synchronous).",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Relative path, '.' for the workspace root."}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_file",
        "description": "Read the first part of a text file inside the workspace (fast, synchronous).",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
]


@dataclass
class BackgroundTask:
    id: int
    label: str
    command: str
    started: float = field(default_factory=time.monotonic)
    result: str | None = None


class ToolRunner:
    """Executes tools. Slow ones become background tasks; completions land in `done` for announcement."""

    def __init__(self, workspace: Path, on_done: Callable[[BackgroundTask], None]):
        self.ws = workspace.resolve()
        self.on_done = on_done
        self.tasks: dict[int, BackgroundTask] = {}
        self.notes: list[str] = []      # what happened during the current turn, for the history
        self._n = 0

    def _inside(self, rel: str) -> Path:
        p = (self.ws / rel).resolve()
        if p != self.ws and self.ws not in p.parents:
            raise ValueError("path escapes the workspace")
        return p

    async def call(self, name: str, args: dict[str, Any]) -> str:
        log("tool", f"{name} {json.dumps(args)[:120]}")
        try:
            if name == "run_background_task":
                return self._start(str(args["command"]), str(args.get("label") or "task"))
            if name == "list_dir":
                p = self._inside(str(args.get("path") or "."))
                names = sorted(os.listdir(p))[:60]
                self.notes.append(f"listed {args.get('path') or '.'}")
                return "\n".join(names) or "(empty)"
            if name == "read_file":
                p = self._inside(str(args["path"]))
                text = p.read_text(errors="replace")[:4000]
                self.notes.append(f"read {args['path']}")
                return text or "(empty file)"
            return f"unknown tool {name}"
        except Exception as e:  # noqa: BLE001
            return f"error: {e}"

    def _start(self, command: str, label: str) -> str:
        self._n += 1
        task = BackgroundTask(self._n, label, command)
        self.tasks[task.id] = task
        self.notes.append(f"started task #{task.id} ({label})")
        log("task", f"#{task.id} start: {label}  $ {command}")
        threading.Thread(target=self._run, args=(task,), daemon=True).start()
        return f"queued as task #{task.id}; keep talking, you will be told when it finishes"

    def _run(self, task: BackgroundTask) -> None:
        try:
            r = subprocess.run(task.command, shell=True, cwd=self.ws, capture_output=True, text=True, timeout=120)
            out = (r.stdout + r.stderr).strip()
            tail = "\n".join(out.splitlines()[-15:])[-1500:]
            task.result = f"exit {r.returncode}\n{tail or '(no output)'}"
        except subprocess.TimeoutExpired:
            task.result = "timed out after 120 s"
        except Exception as e:  # noqa: BLE001
            task.result = f"failed: {e}"
        log("task", f"#{task.id} done in {time.monotonic() - task.started:.1f}s: {task.result.splitlines()[0]}")
        self.on_done(task)


def openai_tools() -> list[dict[str, Any]]:
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}} for t in TOOL_SPECS]


def anthropic_tools() -> list[dict[str, Any]]:
    return [
        {"name": t["name"], "description": t["description"], "input_schema": t["parameters"], "strict": True, "eager_input_streaming": True}
        for t in TOOL_SPECS
    ]


# --------------------------------------------------------------------------
# 6. LLM backends. Each yields text deltas and runs its own native tool loop.
#    History is kept as plain text turns so an interrupted turn is trivial to commit.
# --------------------------------------------------------------------------
def system_prompt(workspace: Path) -> str:
    return (
        f"You are a voice assistant working inside the software workspace at {workspace}. "
        "You are heard, not read: answer in one to three short spoken sentences. No markdown, no lists, "
        "no code blocks, no URLs. Say file names and numbers plainly. "
        "If the user asks for anything that takes time, start it with run_background_task and keep talking; "
        "never wait for it and never pretend to know its result. "
        "A message starting with [task finished] is a background result: summarize it in one sentence. "
        "Bracketed notes in the history such as [started task #1] or [cut off by the user] are annotations added by "
        "the system, not words you said; never write such notes yourself. "
        "[cut off by the user] means you were interrupted there; do not repeat yourself. "
        "If the user says goodbye, say a short goodbye."
    )


class Brain:
    name = "base"

    def __init__(self, workspace: Path):
        self.system = system_prompt(workspace)
        self.history: list[dict[str, str]] = []

    async def respond(self, user_text: str, tools: ToolRunner) -> AsyncIterator[str]:
        raise NotImplementedError
        yield ""  # pragma: no cover

    def begin(self, user_text: str) -> None:
        self.history.append({"role": "user", "content": user_text})

    def commit(self, spoken: str, notes: list[str], interrupted: bool) -> None:
        text = spoken.strip()
        if notes:
            text += " [" + "; ".join(notes) + "]"
        if interrupted:
            text += " [cut off by the user]"
        self.history.append({"role": "assistant", "content": text or "(said nothing)"})


class ScriptedBrain(Brain):
    """No model. Streams canned replies slowly enough that barge-in is easy to try."""

    name = "scripted"

    async def respond(self, user_text: str, tools: ToolRunner) -> AsyncIterator[str]:
        self.begin(user_text)
        t = user_text.lower()
        if any(w in t for w in ("goodbye", "bye bye", "see you")):
            reply = "Goodbye."
        elif any(w in t for w in ("run", "test", "check", "build", "search")):
            await tools.call("run_background_task", {"command": "sleep 4; ls -1 | head -5; echo done", "label": "list workspace files"})
            reply = ("I started that in the background as a task. While it runs, you can keep talking, and I will "
                     "tell you the moment it finishes. Ask me anything else in the meantime.")
        elif "list" in t or "files" in t:
            listing = await tools.call("list_dir", {"path": "."})
            names = [n for n in listing.splitlines()[:4]]
            reply = f"The workspace contains {', '.join(names)}, among others."
        else:
            reply = (f"You said: {user_text}. Here is a deliberately long reply so you can interrupt me. "
                     "A full duplex interface keeps listening while it speaks, so you should be able to cut in at any point. "
                     "Try saying stop, or ask something new, right now. If I keep going, barge-in is not working yet.")
        for word in reply.split(" "):
            yield word + " "
            await asyncio.sleep(0.03)


class LlamaCppBrain(Brain):
    """llama-server (llama.cpp) OpenAI-compatible endpoint. Start it with --jinja so tools work.

    Sampling matches the FTL spec-decoding bench on the M5 Pro (Qwen3.6-35B-A3B Q4_0, DFlash2 n=3),
    so the measured 95 tok/s decode and ~70% draft acceptance carry over. Thinking is disabled
    per request via chat_template_kwargs; a spoken reply cannot wait for a thinking block.
    """

    name = "llamacpp"
    SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}

    def __init__(self, workspace: Path, url: str, model: str | None):
        super().__init__(workspace)
        import httpx

        self.url = url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0))
        self.model = model or "default"

    async def respond(self, user_text: str, tools: ToolRunner) -> AsyncIterator[str]:
        self.begin(user_text)
        msgs: list[dict[str, Any]] = [{"role": "system", "content": self.system}, *self.history]
        for _round in range(4):
            content, calls = "", {}
            body = {"model": self.model, "messages": msgs, "tools": openai_tools(), "tool_choice": "auto",
                    "stream": True, "max_tokens": 400, **self.SAMPLING,
                    "chat_template_kwargs": {"enable_thinking": False}}
            async with self.client.stream("POST", f"{self.url}/v1/chat/completions", json=body) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    choice = json.loads(payload)["choices"][0]
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        content += delta["content"]
                        yield delta["content"]
                    for tc in delta.get("tool_calls") or []:
                        e = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "args": ""})
                        e["id"] = tc.get("id") or e["id"]
                        fn = tc.get("function") or {}
                        e["name"] += fn.get("name") or ""
                        e["args"] += fn.get("arguments") or ""
            if not calls:
                return
            msgs.append({"role": "assistant", "content": content or None, "tool_calls": [
                {"id": e["id"] or f"call_{i}", "type": "function", "function": {"name": e["name"], "arguments": e["args"]}}
                for i, e in calls.items()]})
            for i, e in calls.items():
                try:
                    args = json.loads(e["args"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = await tools.call(e["name"], args if isinstance(args, dict) else {})
                msgs.append({"role": "tool", "tool_call_id": e["id"] or f"call_{i}", "content": result})


class AnthropicBrain(Brain):
    """Claude via the Anthropic SDK (streaming, manual tool loop, refusal fallbacks on)."""

    name = "anthropic"

    def __init__(self, workspace: Path, model: str | None):
        super().__init__(workspace)
        import anthropic

        self.client = anthropic.AsyncAnthropic()
        self.model = model or "claude-opus-5-5"

    async def respond(self, user_text: str, tools: ToolRunner) -> AsyncIterator[str]:
        self.begin(user_text)
        msgs: list[dict[str, Any]] = [dict(m) for m in self.history]
        for _round in range(4):
            async with self.client.beta.messages.stream(
                model=self.model,
                max_tokens=1024,
                system=self.system,
                tools=anthropic_tools(),
                messages=msgs,
                output_config={"effort": "low"},   # voice: latency over depth; thinking stays adaptive
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            ) as stream:
                async for ev in stream:
                    if ev.type == "text":
                        yield ev.text
                msg = await stream.get_final_message()
            if msg.stop_reason == "refusal":
                yield " I can't help with that one."
                return
            tool_uses = [b for b in msg.content if b.type == "tool_use"]
            if not tool_uses or msg.stop_reason == "max_tokens":
                return
            msgs.append({"role": "assistant", "content": msg.content})
            results = []
            for b in tool_uses:
                args = b.input if isinstance(b.input, dict) else {}
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": await tools.call(b.name, args)})
            msgs.append({"role": "user", "content": results})


def pick_brain(kind: str, workspace: Path, llm_url: str, model: str | None) -> Brain:
    if kind == "auto":
        try:
            import httpx

            httpx.get(f"{llm_url.rstrip('/')}/v1/models", timeout=1.0).raise_for_status()
            kind = "llamacpp"
        except Exception:  # noqa: BLE001
            kind = "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "scripted"
        log("llm", f"auto-selected {kind}")
    if kind == "llamacpp":
        return LlamaCppBrain(workspace, llm_url, model)
    if kind == "anthropic":
        return AnthropicBrain(workspace, model)
    return ScriptedBrain(workspace)


# --------------------------------------------------------------------------
# 7. The duplex session
# --------------------------------------------------------------------------
class Session:
    def __init__(self, vad, asr, tts, player: Player, brain: Brain, workspace: Path, tune: Tuning, mic_device=None):
        self.vad, self.asr, self.tts, self.player, self.brain, self.tune = vad, asr, tts, player, brain, tune
        self.mic_device = mic_device
        self.loop = asyncio.get_event_loop()
        self.frames: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self.announce: asyncio.Queue[BackgroundTask] = asyncio.Queue()
        self.tools = ToolRunner(workspace, self._task_done)
        self.detector = TurnDetector(vad, tune, self.assistant_speaking)
        self.state = "idle"
        self.user_talking = False
        self.turn: asyncio.Task | None = None
        self.turn_spoken: list[str] = []
        self.turn_user_text = ""
        self.quit = asyncio.Event()

    # -- state ---------------------------------------------------------------
    def set_state(self, s: str, why: str = "") -> None:
        if s != self.state:
            self.state = s
            log("state", f"{s}  {why}".rstrip())

    def assistant_speaking(self) -> bool:
        return self.player.busy.is_set()

    def _task_done(self, task: BackgroundTask) -> None:
        self.loop.call_soon_threadsafe(self.announce.put_nowait, task)

    # -- audio in -------------------------------------------------------------
    def _mic_callback(self, indata, frames, time_info, status) -> None:
        if status:
            log("mic", str(status))
        self.loop.call_soon_threadsafe(self.frames.put_nowait, indata[:, 0].copy())

    async def feed_frames(self, source: AsyncIterator[np.ndarray] | None = None) -> None:
        """Consume frames from the mic queue (or an injected source in tests) and route detector events."""
        async def from_queue():
            while not self.quit.is_set():
                yield await self.frames.get()

        async for frame in (source or from_queue()):
            for kind, payload in self.detector.feed(frame):
                if kind == "speech_start":
                    self.user_talking = True
                    if self.turn and not self.turn.done():
                        await self.barge_in(payload)
                    self.set_state("user_speaking", f"p={payload:.2f}")
                elif kind == "discard":
                    self.user_talking = False
                    self.set_state("idle", "too short, ignored")
                elif kind == "utterance":
                    self.user_talking = False
                    await self.on_utterance(payload)

    # -- turn handling --------------------------------------------------------
    async def barge_in(self, p: float) -> None:
        log("barge-in", f"user spoke over the assistant (p={p:.2f}); cutting playback and the LLM stream")
        self.player.stop.set()
        assert self.turn is not None
        self.turn.cancel()
        try:
            await self.turn
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        self.player.stop.clear()

    async def on_utterance(self, audio: np.ndarray) -> None:
        self.set_state("transcribing", f"{len(audio) / SR:.1f}s of audio")
        t = time.monotonic()
        text = await asyncio.to_thread(self.asr.transcribe, audio)
        log("asr", f"{time.monotonic() - t:.2f}s  -> {text!r}")
        if not text:
            self.set_state("idle", "empty transcript")
            return
        print(f"\n  YOU: {text}\n", flush=True)
        self.turn = asyncio.create_task(self.run_turn(text))

    async def run_turn(self, user_text: str, is_event: bool = False) -> None:
        """Stream LLM -> sentences -> TTS -> player, with synthesis pipelined one sentence ahead."""
        self.turn_spoken, self.tools.notes = [], []
        self.turn_user_text = user_text
        self.set_state("thinking")
        speech_q: asyncio.Queue[tuple[str, np.ndarray] | None] = asyncio.Queue(maxsize=2)
        interrupted = False
        t_start = time.monotonic()

        async def speaker():
            first = True
            while True:
                item = await speech_q.get()
                if item is None:
                    return
                sentence, audio = item
                if first:
                    log("latency", f"first audio {time.monotonic() - t_start:.2f}s after turn start")
                    first = False
                self.set_state("speaking")
                print(f"  ASSISTANT: {sentence}", flush=True)
                played = await asyncio.to_thread(self.player.play, audio)
                if played >= len(audio):
                    self.turn_spoken.append(sentence)
                else:
                    heard = sentence[: int(len(sentence) * played / max(1, len(audio)))].rstrip()
                    self.turn_spoken.append(heard + "...")
                    log("cut", f"{played / self.player.sample_rate:.2f}s of {len(audio) / self.player.sample_rate:.2f}s heard: {heard!r}")
                    return

        speak_task = asyncio.create_task(speaker())
        splitter = SentenceSplitter()
        try:
            async def sentences():
                async for delta in self.brain.respond(user_text, self.tools):
                    for s in splitter.push(delta):
                        yield s
                for s in splitter.flush():
                    yield s

            async for s in sentences():
                audio = await asyncio.to_thread(self.tts.synth, s)
                await speech_q.put((s, audio))
            await speech_q.put(None)
            await speak_task
        except asyncio.CancelledError:
            interrupted = True
            self.player.stop.set()
            if self.player.busy.is_set():          # let play() return so the cut point is recorded
                await asyncio.wait({speak_task}, timeout=1.0)
            speak_task.cancel()
            raise
        finally:
            spoken = " ".join(self.turn_spoken)
            self.brain.commit(spoken, list(self.tools.notes), interrupted)
            if interrupted:
                log("history", f"committed partial turn: {spoken[:80]!r} [cut off]")
            else:
                self.set_state("idle", f"turn done in {time.monotonic() - t_start:.1f}s")
                if any(w in user_text.lower() for w in ("goodbye", "bye bye")) and not is_event:
                    self.quit.set()

    async def announcer(self) -> None:
        """Deliver background results only when the floor is free: nobody talking, no turn running."""
        while not self.quit.is_set():
            task = await self.announce.get()
            while self.user_talking or (self.turn and not self.turn.done()) or self.detector.active:
                await asyncio.sleep(0.1)
            log("announce", f"task #{task.id} result reaches the model now")
            self.turn = asyncio.create_task(self.run_turn(f"[task finished] #{task.id} {task.label}: {task.result}", is_event=True))
            await asyncio.wait({self.turn})

    # -- lifecycle ------------------------------------------------------------
    async def run(self) -> None:
        import sounddevice as sd

        stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=FRAME, device=self.mic_device,
                                callback=self._mic_callback)
        stream.start()
        log("ready", "listening. Speak naturally; interrupt whenever you like. Say goodbye to exit.")
        tasks = [asyncio.create_task(self.feed_frames()), asyncio.create_task(self.announcer())]
        try:
            await self.quit.wait()
            if self.turn and not self.turn.done():
                await asyncio.wait({self.turn}, timeout=10)
        finally:
            for t in tasks:
                t.cancel()
            stream.stop()
            stream.close()
            self.player.close()
            log("bye", "session ended")


# --------------------------------------------------------------------------
# 8. Self-test: whole pipeline without a microphone, speaker, or model
# --------------------------------------------------------------------------
class FakeLlamaServer:
    """Minimal OpenAI-compatible SSE server: first request answers with a tool call, the next with text."""

    def __init__(self):
        from http.server import BaseHTTPRequestHandler, HTTPServer

        calls = {"n": 0}

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def do_GET(self):
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
                self.wfile.write(b'{"object":"list","data":[{"id":"fake"}]}')

            def do_POST(self):
                n = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(n))
                calls["n"] += 1
                self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
                has_tool_result = any(m.get("role") == "tool" for m in body["messages"])
                if not has_tool_result:
                    chunks = [
                        {"delta": {"content": "Sure, "}},
                        {"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "run_background_task", "arguments": '{"command": "echo hi"'}}]}},
                        {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ', "label": "say hi"}'}}]}},
                        {"delta": {}, "finish_reason": "tool_calls"},
                    ]
                else:
                    chunks = [{"delta": {"content": w}} for w in ["I started ", "that. ", "It will ", "report back soon."]]
                    chunks.append({"delta": {}, "finish_reason": "stop"})
                for c in chunks:
                    self.wfile.write(f"data: {json.dumps({'choices': [c]})}\n\n".encode()); self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.calls = calls


async def selftest(args) -> int:
    ok = True

    def check(cond: bool, what: str) -> None:
        nonlocal ok
        ok = ok and bool(cond)
        log("PASS" if cond else "FAIL", what)

    # VAD + turn detection on synthesized speech
    vad = load_vad(args.energy_vad)
    tts = SayTTS(args.voice, args.rate) if sys.platform == "darwin" else NullTTS()
    wav = await asyncio.to_thread(tts.synth, "Hello there, this is a quick test of the duplex pipeline.")
    log("tts", f"{type(tts).__name__} produced {len(wav) / tts.sample_rate:.2f}s at {tts.sample_rate} Hz")
    # resample to 16 kHz by linear interpolation (prototype quality is fine here)
    n16 = int(len(wav) * SR / tts.sample_rate)
    speech16 = np.interp(np.linspace(0, len(wav) - 1, n16), np.arange(len(wav)), wav).astype(np.float32)
    silence = np.zeros(SR, dtype=np.float32)
    signal = np.concatenate([silence, speech16, silence])
    noise = (0.002 * np.random.default_rng(0).standard_normal(len(signal))).astype(np.float32)
    signal = signal + noise
    det = TurnDetector(vad, Tuning(), lambda: False)
    events = []
    for i in range(0, len(signal) - FRAME + 1, FRAME):
        events += det.feed(signal[i:i + FRAME])
    kinds = [k for k, _ in events]
    log("vad", f"{type(vad).__name__} events: {kinds}")
    check("speech_start" in kinds, "VAD detects speech onset")
    utts = [p for k, p in events if k == "utterance"]
    check(len(utts) == 1, "VAD endpoints exactly one utterance")

    # barge-in gating: the same audio with the assistant 'speaking' needs the stricter threshold
    det2 = TurnDetector(vad, Tuning(), lambda: True)
    vad.reset()
    ev2 = [k for i in range(0, len(signal) - FRAME + 1, FRAME) for k, _ in det2.feed(signal[i:i + FRAME])]
    check("speech_start" in ev2, "clear speech still barges in while assistant is speaking")

    # ASR
    if utts:
        asr = WhisperASR(args.asr)
        t = time.monotonic()
        text = await asyncio.to_thread(asr.transcribe, utts[0])
        log("asr", f"{time.monotonic() - t:.2f}s -> {text!r}")
        check("duplex" in text.lower() or "pipeline" in text.lower(), "ASR transcribes the synthesized sentence")

    # sentence splitter
    sp = SentenceSplitter()
    got = sp.push("One two. Three four! Five") + sp.push(" six? Seven") + sp.flush()
    check(got == ["One two.", "Three four!", "Five six?", "Seven"], f"sentence splitter {got}")

    # llama.cpp backend against the fake server (streaming parse + tool round trip)
    fake = FakeLlamaServer()
    ws = Path(args.workspace).resolve()
    done: list[BackgroundTask] = []
    tools = ToolRunner(ws, done.append)
    brain = LlamaCppBrain(ws, fake.url, None)
    out = "".join([d async for d in brain.respond("say hi in the background", tools)])
    check(out.startswith("Sure, ") and "report back soon." in out, f"llamacpp stream text {out!r}")
    check(fake.calls["n"] == 2 and tools.notes == ["started task #1 (say hi)"], f"tool round trip {tools.notes}")
    for _ in range(50):
        if done:
            break
        await asyncio.sleep(0.1)
    check(bool(done) and "hi" in (done[0].result or ""), f"background task completed: {done[0].result if done else None!r}")
    brain.commit("Sure, I started that.", tools.notes, interrupted=True)
    check(brain.history[-1]["content"].endswith("[cut off by the user]"), "interrupted turn committed to history")

    # anthropic backend constructs (no network)
    try:
        AnthropicBrain(ws, None)
        check(True, "anthropic backend constructs (needs ANTHROPIC_API_KEY or ant auth to run)")
    except Exception as e:  # noqa: BLE001
        check("api_key" in str(e).lower() or "ANTHROPIC_API_KEY" in str(e), f"anthropic backend needs credentials: {e}")

    # whole session loop with injected frames, silent player, scripted brain
    player = Player(tts.sample_rate, enabled=False)
    sess = Session(vad, asr, tts, player, ScriptedBrain(ws), ws, Tuning())
    vad.reset()

    async def frames():
        for i in range(0, len(signal) - FRAME + 1, FRAME):
            yield signal[i:i + FRAME]
            await asyncio.sleep(0)

    await sess.feed_frames(frames())
    if sess.turn:
        await asyncio.wait({sess.turn}, timeout=30)
    check(sess.turn is not None and sess.turn.done() and len(sess.brain.history) == 2, "session: utterance -> turn -> history")
    check(bool(sess.turn_spoken), f"session spoke {len(sess.turn_spoken)} sentence(s)")

    # barge-in: start a long turn, interrupt it mid-way
    sess.set_state("idle")
    sess.turn = asyncio.create_task(sess.run_turn("tell me something long"))
    while sess.state != "speaking" and not sess.turn.done():
        await asyncio.sleep(0.05)
    await asyncio.sleep(0.3)   # mid-sentence
    await sess.barge_in(0.99)
    last = sess.brain.history[-1]["content"]
    check(sess.turn.done() and last.endswith("[cut off by the user]") and "..." in last,
          f"barge-in cancels turn mid-sentence, partial history: {last[:70]!r}")
    log("RESULT", "ALL PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", choices=["auto", "scripted", "llamacpp", "anthropic"], default="auto")
    ap.add_argument("--llm-url", default=os.environ.get("LLAMA_URL", "http://127.0.0.1:8080"), help="llama-server base URL")
    ap.add_argument("--model", default=None, help="model id (anthropic default claude-opus-5-5; llama-server ignores it)")
    ap.add_argument("--asr", default="base.en", help="faster-whisper size: tiny.en, base.en, small.en")
    ap.add_argument("--tts", choices=["kokoro", "say"], default="kokoro", help="kokoro needs espeak-ng (brew install espeak-ng)")
    ap.add_argument("--voice", default=None, help="say voice (e.g. Samantha) or kokoro voice (e.g. af_heart)")
    ap.add_argument("--rate", type=int, default=190, help="say words per minute")
    ap.add_argument("--workspace", default=".", help="directory the tools may touch")
    ap.add_argument("--mic", type=int, default=None, help="input device index (see --list-devices)")
    ap.add_argument("--speaker", type=int, default=None, help="output device index")
    ap.add_argument("--energy-vad", action="store_true", help="skip Silero, use the RMS gate")
    ap.add_argument("--end-silence-ms", type=int, default=700)
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="run the pipeline on synthesized audio, no mic/model")
    args = ap.parse_args()

    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return 0
    if args.selftest:
        return asyncio.run(selftest(args))

    workspace = Path(args.workspace).resolve()
    vad = load_vad(args.energy_vad)
    asr = WhisperASR(args.asr)
    if args.tts == "kokoro":
        try:
            tts = KokoroTTS(args.voice or "af_heart")
            tts.synth("warm up")   # first call loads the graph and surfaces a broken phonemizer early
        except Exception as e:  # noqa: BLE001
            log("tts", f"Kokoro unavailable ({e}); falling back to macOS say. Fix: brew install espeak-ng")
            tts = SayTTS(None, args.rate)
    else:
        tts = SayTTS(args.voice, args.rate)
    player = Player(tts.sample_rate, device=args.speaker)
    brain = pick_brain(args.llm, workspace, args.llm_url, args.model)
    log("llm", f"backend {brain.name}" + (f" model {brain.model}" if hasattr(brain, "model") else ""))
    import sounddevice as sd

    din, dout = sd.query_devices(args.mic, "input")["name"], sd.query_devices(args.speaker, "output")["name"]
    log("audio", f"mic: {din}  |  speaker: {dout}")
    if "speaker" in dout.lower() and "microphone" in din.lower():
        log("warn", "built-in speakers + mic: playback will bleed into the mic. Barge-in uses a stricter gate; headphones are better.")
    tune = Tuning(end_silence_ms=args.end_silence_ms)

    async def go():
        sess = Session(vad, asr, tts, player, brain, workspace, tune, mic_device=args.mic)
        await sess.run()

    try:
        asyncio.run(go())
    except KeyboardInterrupt:
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
