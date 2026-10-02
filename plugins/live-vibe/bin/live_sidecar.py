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
#   "moshi-mlx>=0.3.0; sys_platform == 'darwin' and platform_machine == 'arm64'",
# ]
# ///
"""Audio sidecar for the live-vibe mod. It owns the mic and speaker in two modes:

  --mode live   (/live)      The mod owns the conversation: utterances go to Claude, Claude's answers come
                             back through POST /speak.
  --mode front  (/livevibe)  A voice FRONT runs here: a small fast model the user talks with directly. Its one
                             tool, delegate, hands real work to Claude (the vibe director); Claude's answers come
                             back through POST /event and the front relays them once the floor is free.

Reuses duplex_voice.py beside it (vendored from proto/): Silero VAD, TurnDetector, faster-whisper, Kokoro/say
TTS, Player, SentenceSplitter, and for the front its Session (barge-in, announcer, commit only what was heard),
LlamaCppBrain and AnthropicBrain. The prototype's dependency list plus moshi-mlx for the Kyutai backends.

Audio backends: --stt kyutai (Kyutai STT 1B on MLX, Apple silicon: streaming words and its own semantic end of
turn; about 2.4 GB of weights on first use) or whisper (Silero VAD + faster-whisper); --tts kokoro or say.

stdout, one JSON object per line:
  {"type":"ready","port":N}
  {"type":"state","state":"loading|listening|user_speaking|transcribing|thinking|speaking"}
  {"type":"utterance","text":"..."}                      live: what the user said
  {"type":"barge_in"}                                    live: the user spoke over playback; playback stopped
  {"type":"spoken","text":"...","cut":b}                 live: what was actually heard of one /speak
  {"type":"delegate","text":"..."}                       front: work for Claude
  {"type":"switch_model","model":"sonnet"}               front: a spoken model switch, for Claude
  {"type":"transcript","role":"user|front","text":"..."} front: the voice conversation, for the screen
  {"type":"warn","text":"..."}                           something the user should see
  {"type":"log","text":"..."}
HTTP on 127.0.0.1:N:  POST /speak <text> (live)   POST /event <text> (front)   POST /stop   POST /quit

  uv run --script live_sidecar.py --selftest    front mode end to end: no mic, speaker, model or network
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import duplex_voice as dv  # noqa: E402

IS_MAC = sys.platform == "darwin"
KYUTAI_OK = IS_MAC and __import__("platform").machine() == "arm64"  # moshi-mlx is installed only there
# Debian/Ubuntu on arm64 keeps espeak-ng under its own triplet; the prototype lists Homebrew and x86_64 Linux.
dv.KokoroTTS.ESPEAK_CANDIDATES.append(("/usr/lib/aarch64-linux-gnu/libespeak-ng.so.1", "/usr/lib/aarch64-linux-gnu/espeak-ng-data"))

_out = threading.Lock()
_capture: list[dict[str, Any]] | None = None  # the self-test reads emits here instead of stdout


def emit(**obj) -> None:
    if _capture is not None:
        _capture.append(obj)
        return
    with _out:  # the speaker thread and the main loop both write; keep lines whole
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()


dv.log = lambda tag, msg="": emit(type="log", text=f"{tag} {msg}".strip())  # keep stdout JSON-only
dv.print = lambda *_, **__: None  # the prototype's console transcript; the front emits its own

_FENCE = re.compile(r"```.*?```", re.S)
_TABLE_ROW = re.compile(r"^\s*\|.*$", re.M)
_URL = re.compile(r"https?://\S+")


def speakable(text: str) -> str:
    """Drop what cannot be read aloud; dv.clean_for_speech strips the inline markdown."""
    text = _FENCE.sub(" code omitted. ", text)
    text = _TABLE_ROW.sub("", text)
    return _URL.sub("a link", text)


# -- audio seams: the STT and TTS backends are picked here ---------------------------------------
def make_stt(args):
    """The recognizer, and with it the turn detector: whisper pairs with Silero, kyutai brings its own."""
    if args.stt == "kyutai" and not KYUTAI_OK:
        emit(type="log", text="Kyutai STT runs on MLX, Apple silicon only; using Whisper.")
    elif args.stt == "kyutai":
        try:
            return KyutaiSTT()
        except Exception as e:  # noqa: BLE001
            emit(type="warn", text=f"Kyutai STT unavailable ({type(e).__name__}: {str(e)[:120]}); using Whisper.")
    return Whisper(args.asr)


class Whisper(dv.WhisperASR):
    """The prototype's faster-whisper, on an Nvidia GPU (float16) when CTranslate2 finds one and its CUDA
    libraries load, else on the CPU (int8)."""

    def __init__(self, size: str = "base.en"):
        import ctranslate2
        from faster_whisper import WhisperModel

        self.lang = "en" if size.endswith(".en") else None
        if ctranslate2.get_cuda_device_count() > 0:
            try:
                self.model = WhisperModel(size, device="cuda", compute_type="float16")
                self.transcribe(np.zeros(dv.SR, np.float32))  # cuBLAS and cuDNN load on first use, not at init
                emit(type="log", text=f"asr: faster-whisper {size} on CUDA (float16)")
                return
            except Exception as e:  # noqa: BLE001
                emit(type="warn", text=f"Whisper on CUDA failed ({str(e)[:120]}); using the CPU. Fix: install CUDA 12 cuBLAS and cuDNN 9.")
        self.model = WhisperModel(size, device="cpu", compute_type="int8")
        emit(type="log", text=f"asr: faster-whisper {size} on the CPU (int8)")


def make_detector(stt, tune: dv.Tuning, speaking: Callable[[], bool]):
    """Kyutai ends a turn on its own semantic signal; Whisper needs Silero and the prototype's endpointing."""
    if isinstance(stt, KyutaiSTT):
        return KyutaiTurns(stt, tune, speaking)
    return dv.TurnDetector(dv.load_vad(), tune, speaking)


class NoSpeech(SystemExit):
    """A missing system piece the sidecar cannot work without; main() reports it once and exits."""


ESPEAK_FIX = "brew install espeak-ng" if IS_MAC else "sudo apt install espeak-ng (or your distribution's espeak-ng)"


def make_tts(args):
    """The synthesizer: anything with sample_rate and synth(text) -> float32 PCM."""
    kind, voice = args.tts, args.voice or None
    if kind == "kokoro":
        try:
            t = dv.KokoroTTS(voice or "af_heart")
            t.synth("warm up")
            return t
        except Exception as e:  # noqa: BLE001
            if not IS_MAC:
                raise NoSpeech(f"Kokoro cannot speak ({str(e)[:120]}). Fix: {ESPEAK_FIX}.") from e
            emit(type="log", text=f"Kokoro unavailable ({e}); using macOS say. Fix: {ESPEAK_FIX}")
            voice = None
    if not IS_MAC:
        raise NoSpeech(f"tts 'say' is macOS only. Use kokoro; it needs {ESPEAK_FIX}.")
    return dv.SayTTS(voice)


def check_audio() -> None:
    """PortAudio and a default mic and speaker, before minutes of model loading."""
    try:
        import sounddevice as sd
    except OSError as e:  # the wheel bundles PortAudio on macOS and Windows, not on Linux
        raise NoSpeech(f"PortAudio is missing ({e}). Fix: sudo apt install libportaudio2.") from e
    for kind in ("input", "output"):
        try:
            sd.query_devices(kind=kind)
        except Exception as e:  # noqa: BLE001 - a headless or SSH session has no sound devices
            raise NoSpeech(f"no default {kind} device ({e}). Voice needs a local microphone and speaker.") from e


# -- Kyutai on MLX ----------------------------------------------------------------------------------
# Loading and stepping follow kyutai-labs/delayed-streams-modeling (scripts/stt_from_mic_mlx.py; MIT for the
# Python code). Weights: CC-BY 4.0. MLX asserts in Metal when two threads run it at once (measured with Kyutai TTS
# beside the STT); the STT steps on one thread at a time, so it needs no lock while nothing else here uses MLX.
KYUTAI_STT_REPO = "kyutai/stt-1b-en_fr-candle"  # the -mlx repo drops the extra heads that carry end of turn
KYUTAI_SR, KYUTAI_BLOCK = 24_000, 1920  # 80 ms steps at 12.5 Hz
BARGE_IN_WORDS = 3  # words the user must say over the assistant before it stops: bleed and "mm-hm" stay out
END_OF_TURN = 0.5  # the end-of-turn head (extra head 2, class 0): measured ~0 in speech, <0.5 in mid-sentence pauses,
# and above 0.5 from just before the last word on; it runs ahead of the words, which trail the audio by the model's
# audio delay, so a turn ends only once it has stayed above for that delay (0.5 s) and the last words are out

_warned_download = False


def _fetch(repo: str, name: str) -> str:
    """hf_hub_download with log lines for the user: a first run downloads gigabytes."""
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    if isinstance(try_to_load_from_cache(repo, name), str):
        return hf_hub_download(repo, name)
    global _warned_download
    if not _warned_download:
        _warned_download = True
        emit(type="warn", text="Kyutai STT's first run downloads its weights (about 2.4 GB); listening starts after.")
    emit(type="log", text=f"kyutai: downloading {repo}/{name}")
    blobs = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub")) / f"models--{repo.replace('/', '--')}" / "blobs"
    done = threading.Event()

    def ticker():
        while not done.wait(10):
            parts = sorted(blobs.glob("*.incomplete"), key=lambda f: f.stat().st_mtime) if blobs.exists() else []
            if parts:
                emit(type="log", text=f"kyutai: {name} {parts[-1].stat().st_size / 1e9:.2f} GB so far")

    threading.Thread(target=ticker, daemon=True).start()
    t = time.monotonic()
    try:
        return hf_hub_download(repo, name)
    finally:
        done.set()
        emit(type="log", text=f"kyutai: {name} ready in {time.monotonic() - t:.0f}s")


def kyutai_cached() -> bool:
    """True when every Kyutai file is on disk (the self-test skips otherwise)."""
    from huggingface_hub import try_to_load_from_cache

    files = [(KYUTAI_STT_REPO, f) for f in ("config.json", "model.safetensors", "mimi-pytorch-e351c8d8@125.safetensors",
                                              "tokenizer_en_fr_audio_8000.model")]
    return all(isinstance(try_to_load_from_cache(r, f), str) for r, f in files)


def _kyutai_lm(repo: str):
    import mlx.core as mx
    import sentencepiece
    from moshi_mlx import models

    raw = json.load(open(_fetch(repo, "config.json")))
    cfg = models.LmConfig.from_config_dict(raw)
    cfg.transformer.max_seq_len = cfg.transformer.context  # upstream workaround for moshi_mlx <= 0.3.0's ring kv cache
    lm = models.Lm(cfg)
    lm.set_dtype(mx.bfloat16)
    lm.load_pytorch_weights(_fetch(repo, raw.get("moshi_name", "model.safetensors")), cfg, strict=True)
    tok = sentencepiece.SentencePieceProcessor(_fetch(repo, raw["tokenizer_name"]))
    mimi = models.mimi.Mimi(models.mimi_202407(max(cfg.generated_codebooks, cfg.other_codebooks)))
    mimi.load_pytorch_weights(_fetch(repo, raw["mimi_name"]), strict=True)
    return raw, cfg, lm, tok, mimi


class KyutaiSTT:
    """Kyutai STT 1B: 80 ms of 24 kHz audio in, at most one word piece and an end-of-turn probability out.
    Words trail the audio by about half a second. Stands in for Whisper; transcribe() passes the text through."""

    MAX_STEPS = 4096  # ~5.5 min of LmGen history; reset() before it, when nobody is talking

    def __init__(self):
        from moshi_mlx import models, utils

        self._models, self._utils = models, utils
        raw, self.cfg, self.lm, self.tok, self.mimi = _kyutai_lm(KYUTAI_STT_REPO)
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

    def step(self, block) -> tuple[str | None, float]:
        """One 1920-sample block -> (word piece or None, P(end of turn))."""
        import mlx.core as mx

        codes = self.mimi.encode_step(mx.array(block, dtype=mx.float32)[None, None])
        token, heads = self.gen.step_with_extra_heads(codes.transpose(0, 2, 1)[0, :, :self.cfg.other_codebooks])
        token = token[0].item()
        p_end = heads[2][0, 0, 0].item() if len(heads) > 2 else 0.0
        return (None if token in (0, 3) else self.tok.id_to_piece(token)), p_end

    def transcribe(self, payload) -> str:
        return payload if isinstance(payload, str) else ""


class KyutaiTurns:
    """dv.TurnDetector's interface over KyutaiSTT: feed() takes the prototype's 16 kHz, 32 ms frames and returns
    ('speech_start', p) / ('utterance', text) / ('discard', None). A turn starts on the first word, or while the
    assistant speaks on BARGE_IN_WORDS words; it ends on the model's end-of-turn head (see END_OF_TURN), or after endSilenceMs
    without a new word (a cap only), or at max_utterance_s."""

    IN_BLOCK = KYUTAI_BLOCK * dv.SR // KYUTAI_SR  # 1280 input samples per 80 ms step

    def __init__(self, stt: KyutaiSTT, tune: dv.Tuning, speaking: Callable[[], bool]):
        self.stt, self.t, self.speaking = stt, tune, speaking
        self.buf = np.zeros(0, np.float32)
        self.prev = 0.0
        self.active = False  # a turn has started (speech_start sent)
        self._clear()

    def _clear(self) -> None:
        self.text, self.words, self.ends, self.started_at, self.last_word = "", 0, 0, None, 0

    def _resample(self, block):
        x = np.concatenate([[self.prev], block])  # 16 kHz -> 24 kHz, linear, continuous across blocks
        self.prev = float(block[-1])
        t = 1 + np.arange(KYUTAI_BLOCK) * (dv.SR / KYUTAI_SR) - dv.SR / KYUTAI_SR
        return np.interp(t, np.arange(len(x)), x).astype(np.float32)

    def feed(self, frame) -> list[tuple[str, Any]]:
        self.buf = np.concatenate([self.buf, frame])
        events: list[tuple[str, Any]] = []
        while len(self.buf) >= self.IN_BLOCK:
            block, self.buf = self.buf[: self.IN_BLOCK], self.buf[self.IN_BLOCK:]
            events += self._step(self._resample(block))
        return events

    def _step(self, block) -> list[tuple[str, Any]]:
        if self.stt.steps >= self.stt.MAX_STEPS - 1 or (self.started_at is None and self.stt.steps > 3000):
            self.stt.reset()
        piece, p_end = self.stt.step(block)
        step = self.stt.steps
        events: list[tuple[str, Any]] = []
        if piece is not None:
            if self.started_at is None:
                self.started_at = step
            self.text += piece.replace("\u2581", " ")
            self.words += piece.startswith("\u2581")
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
                ok = text.lower() not in dv.JUNK_TRANSCRIPTS
                events.append(("utterance", text) if ok else ("discard", None))
            self.active = False
            self._clear()
        return events


def serve(routes: dict[str, Callable[[str], object]]) -> int:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode()
            route = routes.get(self.path)
            if route is None:
                self.send_response(404)
            else:
                route(body)
                self.send_response(204)
            self.end_headers()

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv.server_port


# ===========================================================================================
# live mode: the mod owns the conversation
# ===========================================================================================
class Sidecar:
    def __init__(self, args):
        self.state: str | None = None
        self.set_state("loading")
        self.asr = make_stt(args)
        self.tts = make_tts(args)
        self.player = dv.Player(self.tts.sample_rate)
        self.detector = make_detector(self.asr, dv.Tuning(end_silence_ms=args.end_silence_ms), self.player.busy.is_set)
        self.frames: queue.Queue = queue.Queue()
        self.speak_q: queue.Queue[str | None] = queue.Queue()
        self.quit = threading.Event()

    def set_state(self, s: str) -> None:
        if s != self.state:
            self.state = s
            emit(type="state", state=s)

    # -- speaker thread: one /speak at a time, sentence by sentence ------------
    def speaker(self) -> None:
        while True:
            text = self.speak_q.get()
            if text is None:
                return
            splitter = dv.SentenceSplitter()
            sentences = splitter.push(speakable(text)) + splitter.flush()
            heard: list[str] = []
            cut = False
            self.set_state("speaking")
            for s in sentences:
                if self.player.stop.is_set():
                    cut = True
                    break
                audio = self.tts.synth(s)  # ponytail: synth not pipelined one ahead; add a 1-deep queue if gaps annoy
                played = self.player.play(audio)
                if played < len(audio):
                    heard.append(s[: int(len(s) * played / max(1, len(audio)))].rstrip() + "...")
                    cut = True
                    break
                heard.append(s)
            self.player.stop.clear()
            emit(type="spoken", text=" ".join(heard), cut=cut)
            self.set_state("listening")

    def stop_speaking(self) -> None:
        while True:
            try:
                self.speak_q.get_nowait()
            except queue.Empty:
                break
        if self.player.busy.is_set():
            self.player.stop.set()
        emit(type="barge_in")

    def routes(self) -> dict[str, Callable[[str], object]]:
        def quit_(_: str) -> None:
            self.quit.set()
            self.speak_q.put(None)

        return {"/speak": self.speak_q.put, "/stop": lambda _: self.stop_speaking(), "/quit": quit_}

    # -- main thread: mic frames -> detector -> whisper -------------------------
    def run(self) -> None:
        import sounddevice as sd

        threading.Thread(target=self.speaker, daemon=True).start()
        stream = sd.InputStream(samplerate=dv.SR, channels=1, dtype="float32", blocksize=dv.FRAME,
                                callback=lambda indata, *_: self.frames.put(indata[:, 0].copy()))
        stream.start()
        self.set_state("listening")
        try:
            while not self.quit.is_set():
                try:
                    frame = self.frames.get(timeout=0.5)
                except queue.Empty:
                    continue
                for kind, payload in self.detector.feed(frame):
                    if kind == "speech_start":
                        if self.player.busy.is_set() or not self.speak_q.empty():
                            self.stop_speaking()
                        self.set_state("user_speaking")
                    elif kind == "discard":
                        self.set_state("listening")
                    elif kind == "utterance":
                        self.set_state("transcribing")
                        text = self.asr.transcribe(payload)
                        if text:
                            emit(type="utterance", text=text)
                        self.set_state("speaking" if self.player.busy.is_set() else "listening")
        finally:
            stream.stop()
            stream.close()
            self.player.close()


# ===========================================================================================
# front mode (/livevibe): a voice front that delegates to Claude
# ===========================================================================================
EVENT = "[task finished]"
EVENT_CHARS = 6000  # a long report is cut for the small front model; Claude's full answer is on screen

DELEGATE = {
    "name": "delegate",
    "description": (
        "Hand coding, investigation, file or command work to the coding agent. Include the full request in "
        "plain language with the relevant context from the conversation. Returns at once; the result arrives "
        f"later as a message starting with {EVENT}."
    ),
    "parameters": {
        "type": "object",
        "properties": {"request": {"type": "string", "description": "The complete request, self-contained."}},
        "required": ["request"],
        "additionalProperties": False,
    },
}

# The prototype's spoken-style rules merged with the core of Oh My Pi's live instructions.
FRONT_PROMPT = (
    "You are the voice of a coding assistant working in the software project at {workspace}. "
    "You are heard, not read: answer in one to three short spoken sentences. No markdown, no lists, "
    "no code blocks, no URLs. Say file names and numbers plainly.\n"
    "You and the coding agent are one assistant, not separate agents. Delegate all repository work, coding, "
    "tool use and verification with the delegate tool; never attempt it and never guess at files or results. "
    "Keep the conversation natural while work runs: say in a few words that you are on it. Never claim changes, "
    f"findings or verification before a {EVENT} message reports them. A new request while work is running is "
    "a new delegation. Answer greetings and ordinary conversation directly, without delegating.\n"
    f"A message starting with {EVENT} is the result of earlier work, not words from the user: present it "
    "naturally as your own result in one or two spoken sentences, and say so if work is still running. Never "
    "mention delegation, the agent or the protocol.\n"
    "Bracketed notes in the history such as [delegated: ...] or [cut off by the user] are annotations added by "
    "the system, not words you said; never write such notes yourself. [cut off by the user] means you were "
    "interrupted there; do not repeat yourself. If the user says goodbye, say a short goodbye."
)

# Models that reject AnthropicBrain's output_config.effort and server-side fallback beta.
_NO_EFFORT = re.compile(r"haiku-4-5|sonnet-4-5")


class _WithoutEffort:
    """Stands in for AsyncAnthropic inside AnthropicBrain and drops what an older model rejects."""

    def __init__(self, client):
        self.raw = client
        self.beta = SimpleNamespace(messages=self)

    def stream(self, **kw):
        for k in ("output_config", "betas", "fallbacks"):
            kw.pop(k, None)
        return self.raw.beta.messages.stream(**kw)


def make_front_brain(backend: str, url: str, model: str) -> dv.Brain:
    dv.TOOL_SPECS[:] = [DELEGATE]  # the prototype's tool list, read by openai_tools()/anthropic_tools()
    ws = Path.cwd()
    if backend == "anthropic":
        brain: dv.Brain = dv.AnthropicBrain(ws, model or "claude-haiku-4-5")
        if _NO_EFFORT.search(brain.model):
            brain.client = _WithoutEffort(brain.client)
    else:
        brain = dv.LlamaCppBrain(ws, url, model or None)
    brain.system = FRONT_PROMPT.format(workspace=ws)
    return brain


def _where(brain: dv.Brain) -> str:
    return f"{brain.url} model {brain.model}" if isinstance(brain, dv.LlamaCppBrain) else f"Anthropic {brain.model}"


async def warm_up(brain: dv.Brain) -> bool:
    """One tiny request with the real system prompt and tools, so the first spoken turn is not a cold one."""
    t = time.monotonic()
    try:
        if isinstance(brain, dv.LlamaCppBrain):
            body = {"model": brain.model, "max_tokens": 1, "stream": False, **brain.SAMPLING,
                    "messages": [{"role": "system", "content": brain.system}, {"role": "user", "content": "Hi."}],
                    "tools": dv.openai_tools(), "chat_template_kwargs": {"enable_thinking": False}}
            (await brain.client.post(f"{brain.url}/v1/chat/completions", json=body)).raise_for_status()
        else:
            client = getattr(brain.client, "raw", brain.client)
            await client.messages.create(model=brain.model, max_tokens=1, messages=[{"role": "user", "content": "Hi."}])
    except Exception as e:  # noqa: BLE001
        emit(type="warn", text=f"front model unreachable at {_where(brain)} ({type(e).__name__}: {str(e)[:160]}). "
                               "Until it answers, speech goes straight to Claude.")
        return False
    emit(type="log", text=f"front: {brain.name} at {_where(brain)}, warm in {time.monotonic() - t:.1f}s")
    return True


def spoken_model(pattern: re.Pattern[str] | None, text: str) -> str | None:
    """The mod's own pattern (--switch-pattern) over the mod's own normalization."""
    if pattern is None:
        return None
    m = pattern.search(re.sub(r"\s+", " ", re.sub(r"[^a-z ]+", " ", text.lower())).strip())
    return m.group(1) if m else None


class Delegator:
    """Takes the place of the prototype's ToolRunner: its one tool hands work to Claude over stdout."""

    def __init__(self):
        self.notes: list[str] = []

    async def call(self, name: str, args: dict[str, Any]) -> str:
        request = str(args.get("request") or "").strip()
        if name != "delegate" or not request:
            return f"error: use delegate with a request (got {name})"
        emit(type="delegate", text=request)
        self.notes.append(f"delegated: {request[:100]}")
        return f"Handed off; it runs in the background. Keep talking. The result arrives as a {EVENT} message."


class Echo(dv.Brain):
    """Says its input verbatim: Claude's answer read out when the front model is down."""

    name = "echo"

    async def respond(self, user_text: str, tools) -> Any:
        self.begin(user_text)
        yield user_text


class _Guarded:
    """Wraps the front brain so a failed request ends the turn cleanly (the prototype's speaker task included)."""

    def __init__(self, brain: dv.Brain):
        self.inner = brain
        self.error: Exception | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    async def respond(self, user_text: str, tools) -> Any:
        try:
            async for delta in self.inner.respond(user_text, tools):
                yield delta
        except Exception as e:  # noqa: BLE001
            self.error = e


class FrontSession(dv.Session):
    """The prototype's duplex session with the front brain, one tool, and Claude's answers as announcements."""

    def __init__(self, vad, asr, tts, player, brain: dv.Brain, tune: dv.Tuning, switch: re.Pattern[str] | None):
        super().__init__(vad, asr, tts, player, brain, Path.cwd(), tune)
        self.tools = Delegator()
        self.switch = switch
        self.events: asyncio.Queue[str] = asyncio.Queue()
        self.hearing = False  # transcribing: the floor is not free yet
        self.bg: set[asyncio.Task] = set()

    def set_state(self, s: str, why: str = "") -> None:
        if s != self.state:
            self.state = s
            emit(type="state", state="listening" if s == "idle" else s)

    async def on_utterance(self, audio) -> None:
        self.hearing = True
        try:
            self.set_state("transcribing")
            text = await asyncio.to_thread(self.asr.transcribe, audio)
            if not text:
                self.set_state("idle")
                return
            emit(type="transcript", role="user", text=text)
            model = spoken_model(self.switch, text)
            if model:
                emit(type="switch_model", model=model)  # the mod runs /model and posts the result as an event
                self.set_state("idle")
                return
            self.turn = asyncio.create_task(self.run_turn(text))
        finally:
            self.hearing = False

    async def run_turn(self, user_text: str, is_event: bool = False) -> None:
        brain = self.brain
        guard = self.brain = _Guarded(brain)
        try:
            await super().run_turn(user_text, is_event)
        finally:
            self.brain = brain
            self._said(brain)
        if guard.error is not None:  # the front model is down or broke mid-stream
            e = guard.error
            emit(type="warn", text=f"front model failed at {_where(brain)} ({type(e).__name__}: {str(e)[:160]})")
            if is_event:
                await self.say(user_text.removeprefix(EVENT))
            else:
                emit(type="delegate", text=user_text)  # the request still reaches Claude

    @staticmethod
    def _said(brain: dv.Brain) -> None:
        last = brain.history[-1] if brain.history else {}
        if last.get("role") == "assistant" and last["content"] != "(said nothing)":
            emit(type="transcript", role="front", text=last["content"].strip())

    async def say(self, text: str) -> None:
        """Read the first two sentences of `text` aloud without the front model."""
        splitter = dv.SentenceSplitter()
        short = " ".join((splitter.push(speakable(text)) + splitter.flush())[:2])
        brain, self.brain = self.brain, Echo(Path.cwd())
        try:
            await dv.Session.run_turn(self, short, True)
        finally:
            self.brain = brain

    async def announcer(self) -> None:
        """Claude's answers reach the front only when the floor is free: nobody talking, no turn running."""
        while not self.quit.is_set():
            text = await self.events.get()
            while self.user_talking or self.hearing or self.detector.active or (self.turn and not self.turn.done()):
                await asyncio.sleep(0.1)
            self.turn = asyncio.create_task(self.run_turn(f"{EVENT} {text.strip()[:EVENT_CHARS]}", is_event=True))
            await asyncio.wait({self.turn})

    async def interrupt(self) -> None:
        if self.turn and not self.turn.done():
            await self.barge_in(1.0)
            self.set_state("idle")

    def shutdown(self) -> None:
        self.player.stop.set()
        if self.turn and not self.turn.done():
            self.turn.cancel()
        self.quit.set()

    def routes(self) -> dict[str, Callable[[str], object]]:
        """HTTP handler threads hand everything to the event loop."""
        def soon(fn: Callable[..., object]) -> Callable[[str], object]:
            return lambda body: self.loop.call_soon_threadsafe(fn, body)

        def stop(_: str) -> None:
            task = asyncio.create_task(self.interrupt())
            self.bg.add(task)
            task.add_done_callback(self.bg.discard)

        return {"/event": soon(self.events.put_nowait), "/stop": soon(stop), "/quit": soon(lambda _: self.shutdown())}


def run_front(args) -> int:
    emit(type="state", state="loading")
    switch = re.compile(args.switch_pattern) if args.switch_pattern else None
    asr = make_stt(args)
    tts = make_tts(args)
    player = dv.Player(tts.sample_rate)

    async def go() -> int:
        try:
            brain = make_front_brain(args.front_backend, args.front_url, args.front_model)
        except Exception as e:  # noqa: BLE001
            emit(type="warn", text=f"front backend {args.front_backend} cannot start ({e}). Anthropic needs "
                                   "ANTHROPIC_API_KEY or `ant auth login`; or set frontBackend to llamacpp.")
            return 2
        await warm_up(brain)
        tune = dv.Tuning(end_silence_ms=args.end_silence_ms)
        sess = FrontSession(None, asr, tts, player, brain, tune, switch)
        sess.detector = make_detector(asr, tune, sess.assistant_speaking)
        sess.state = "loading"
        emit(type="ready", port=serve(sess.routes()))
        sess.set_state("idle")
        await sess.run()
        return 0

    try:
        return asyncio.run(go())
    except KeyboardInterrupt:
        return 0


# ===========================================================================================
# self-test: the front mode without a mic, speaker, model or network
# ===========================================================================================
def _fake_front() -> tuple[str, list[dict[str, Any]]]:
    """OpenAI-compatible SSE server: delegates 'fix' requests, relays [task finished], talks long otherwise."""
    seen: list[dict[str, Any]] = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            self.send_response(200)
            if not body.get("stream"):
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"choices":[{"message":{"role":"assistant","content":"Hi"}}]}')
                return
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            last = body["messages"][-1]
            if last["role"] == "tool":
                deltas = [{"content": "I'll let you know."}]
            elif last["content"].startswith(EVENT):
                deltas = [{"content": "All the tests pass now."}]
            elif "fix" in last["content"]:
                args = json.dumps({"request": "Fix the failing test in parser.py"})
                deltas = [{"content": "On it. "},
                          {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "delegate", "arguments": args[:20]}}]},
                          {"tool_calls": [{"index": 0, "function": {"arguments": args[20:]}}]}]
            else:
                deltas = [{"content": f"This is sentence number {i} of a long answer. "} for i in range(12)]
            for d in deltas:
                self.wfile.write(f"data: {json.dumps({'choices': [{'delta': d}]})}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_port}", seen


class _FakeASR:
    def __init__(self, *texts: str):
        self.texts = list(texts)

    def transcribe(self, _audio) -> str:
        return self.texts.pop(0)


async def selftest() -> int:
    global _capture
    import httpx
    import numpy as np

    ok = True

    def check(cond: bool, what: str) -> None:
        nonlocal ok
        ok = ok and bool(cond)
        print(f"{'PASS' if cond else 'FAIL'}  {what}", flush=True)

    def emitted(kind: str) -> list[dict[str, Any]]:
        return [o for o in out if o["type"] == kind]

    async def until(cond: Callable[[], bool], s: float = 10) -> bool:
        for _ in range(int(s / 0.05)):
            if cond():
                return True
            await asyncio.sleep(0.05)
        return False

    _capture = out = []
    url, seen = _fake_front()
    brain = make_front_brain("llamacpp", url, "")
    check(await warm_up(brain), "warm-up reaches the front server")
    w = seen[-1]
    check([t["function"]["name"] for t in w["tools"]] == ["delegate"] and w["chat_template_kwargs"] == {"enable_thinking": False}
          and w["max_tokens"] == 1 and "one assistant" in w["messages"][0]["content"],
          "warm-up sends the front prompt, only delegate, thinking off")

    # The mod's SPOKEN_SWITCH source, as it passes it in --switch-pattern.
    switch = re.compile(r"^(?:(?:ok|okay|hey|alright) )?(?:claude )?(?:please )?(?:(?:switch|change|swap|set)(?: over)?(?: the)?(?: model)? to|use)(?: the)? (opus|sonnet|haiku|fable)(?: model)?(?: please)?$")
    asr = _FakeASR("Please fix the failing test.", "Okay, switch to Sonnet.")
    sess = FrontSession(dv.EnergyVAD(), asr, dv.NullTTS(), dv.Player(16_000, enabled=False), brain, dv.Tuning(), switch)
    port = serve(sess.routes())
    ann = asyncio.create_task(sess.announcer())

    await sess.on_utterance(np.zeros(1600, dtype=np.float32))
    await asyncio.wait({sess.turn})
    check(emitted("delegate") == [{"type": "delegate", "text": "Fix the failing test in parser.py"}], f"utterance -> delegate {emitted('delegate')}")
    check("[delegated: Fix the failing test" in brain.history[-1]["content"] and sess.turn_spoken, f"front spoke and noted it: {brain.history[-1]['content']!r}")
    check([o["role"] for o in emitted("transcript")] == ["user", "front"], "transcript: user then front")

    n = len(brain.history)
    await sess.on_utterance(np.zeros(1600, dtype=np.float32))
    check(emitted("switch_model") == [{"type": "switch_model", "model": "sonnet"}] and len(brain.history) == n,
          "spoken model switch goes to the mod, not the front")

    async with httpx.AsyncClient() as http:
        r = await http.post(f"http://127.0.0.1:{port}/event", content="Fixed parser.py; all 41 tests pass.")
        check(r.status_code == 204, "POST /event accepted")
        await until(lambda: brain.history[-1]["content"] == "All the tests pass now.")
        check(brain.history[-2]["content"].startswith(f"{EVENT} Fixed parser.py"), "event becomes a [task finished] turn")
        check(brain.history[-1]["content"] == "All the tests pass now.", f"front relays it: {brain.history[-1]['content']!r}")

        sess.turn = asyncio.create_task(sess.run_turn("tell me something long"))
        await until(lambda: sess.state == "speaking")
        await asyncio.sleep(0.3)
        await http.post(f"http://127.0.0.1:{port}/stop")
        await until(lambda: sess.turn.done())
        last = brain.history[-1]["content"]
        check(last.endswith("[cut off by the user]") and "..." in last, f"POST /stop cuts the front, commits what was heard: {last[:60]!r}")

        dead = make_front_brain("llamacpp", "http://127.0.0.1:9", "")
        check(not await warm_up(dead) and "127.0.0.1:9" in emitted("warn")[-1]["text"], f"dead front URL: {emitted('warn')[-1]['text'][:90]!r}")
        sess.brain = dead
        out.clear()
        sess.turn = asyncio.create_task(sess.run_turn("what time is it"))
        await asyncio.wait({sess.turn})
        check(emitted("delegate") == [{"type": "delegate", "text": "what time is it"}] and emitted("warn"),
              "front down: the utterance goes straight to Claude")
        sess.turn = asyncio.create_task(sess.run_turn(f"{EVENT} It is noon. The clock is in the corner. More.", is_event=True))
        await asyncio.wait({sess.turn})
        check(sess.turn_spoken == ["It is noon.", "The clock is in the corner."], f"front down: Claude's answer read out {sess.turn_spoken}")

        await http.post(f"http://127.0.0.1:{port}/quit")
        check(await until(sess.quit.is_set, 2), "POST /quit ends the session")
    ann.cancel()

    ok = await selftest_kyutai(check) and ok

    try:
        ab = make_front_brain("anthropic", "", "")
        check(isinstance(ab.client, _WithoutEffort) and ab.model == "claude-haiku-4-5", "anthropic front: haiku without effort/fallbacks")
    except Exception as e:  # noqa: BLE001
        check("api_key" in str(e).lower() or "auth" in str(e).lower(), f"anthropic front needs credentials: {e}")
    print("ALL PASS" if ok else "SOME CHECKS FAILED", flush=True)
    return 0 if ok else 1


async def selftest_kyutai(check: Callable[[bool, str], None]) -> bool:
    """Kyutai STT on synthesized speech; skipped unless the weights are already downloaded."""
    import platform

    if sys.platform != "darwin" or platform.machine() != "arm64" or not kyutai_cached():
        print("SKIP  kyutai backends: weights not downloaded (or not Apple silicon)", flush=True)
        return True
    ok = True

    def chk(cond: bool, what: str) -> None:
        nonlocal ok
        ok = ok and bool(cond)
        check(cond, what)

    say = dv.SayTTS()
    wav = await asyncio.to_thread(say.synth, "Please open the parser file and fix the failing test.")
    speech = np.interp(np.linspace(0, len(wav) - 1, int(len(wav) * dv.SR / say.sample_rate)), np.arange(len(wav)), wav).astype(np.float32)
    voiced = np.flatnonzero(np.abs(speech) > 0.02)
    onset, offset = 2 + voiced[0] / dv.SR, 2 + voiced[-1] / dv.SR
    noise = (0.002 * np.random.default_rng(0).standard_normal(dv.SR * 7 + len(speech))).astype(np.float32)
    signal = np.concatenate([np.zeros(dv.SR * 2, np.float32), speech, np.zeros(dv.SR * 5, np.float32)]) + noise

    t = time.monotonic()
    stt = await asyncio.to_thread(KyutaiSTT)
    print(f"      kyutai STT loaded in {time.monotonic() - t:.1f}s", flush=True)

    def run(det) -> list[tuple[float, str, Any]]:
        got = []
        for i in range(0, len(signal) - dv.FRAME + 1, dv.FRAME):
            got += [((i + dv.FRAME) / dv.SR, k, v) for k, v in det.feed(signal[i:i + dv.FRAME])]
        return got

    stt.reset()
    t = time.monotonic()
    ev = await asyncio.to_thread(run, KyutaiTurns(stt, dv.Tuning(end_silence_ms=1500), lambda: False))
    took = time.monotonic() - t
    start = next((at for at, k, _ in ev if k == "speech_start"), None)
    utt = next(((at, v) for at, k, v in ev if k == "utterance"), None)
    one = sum(k == "utterance" for _, k, _ in ev) == 1
    chk(start is not None and utt is not None and one and "test" in utt[1].lower(), f"kyutai STT turn, one utterance: {[(round(a, 2), k, v) for a, k, v in ev]}")
    if start is not None and utt is not None:
        print(f"      kyutai STT: first word {start - onset:.2f}s after speech onset, end of turn {utt[0] - offset:.2f}s "
              f"after speech end, {took / (len(signal) / dv.SR):.2f}x realtime compute", flush=True)

    stt.reset()
    ev2 = await asyncio.to_thread(run, KyutaiTurns(stt, dv.Tuning(end_silence_ms=1500), lambda: True))
    start2 = next((at for at, k, _ in ev2 if k == "speech_start"), None)
    chk(start is not None and start2 is not None and start2 > start, f"kyutai barge-in waits for {BARGE_IN_WORDS} words: {start} -> {start2}")

    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["live", "front"], default="live")
    ap.add_argument("--stt", choices=["kyutai", "whisper"], default="kyutai", help="speech-to-text backend")
    ap.add_argument("--asr", default="base.en", help="faster-whisper size")
    ap.add_argument("--tts", choices=["kokoro", "say"], default="kokoro")
    ap.add_argument("--voice", default=None)
    ap.add_argument("--end-silence-ms", type=int, default=1500)
    ap.add_argument("--front-backend", choices=["llamacpp", "anthropic"], default="llamacpp")
    ap.add_argument("--front-url", default="http://127.0.0.1:8080", help="OpenAI-compatible server (llama-server)")
    ap.add_argument("--front-model", default="", help="empty: the server's model, or claude-haiku-4-5 for anthropic")
    ap.add_argument("--switch-pattern", default="", help="regex whose group 1 is a model a spoken switch names")
    ap.add_argument("--selftest", action="store_true", help="front mode without mic, speaker, model or network")
    args = ap.parse_args()
    if args.selftest:
        return asyncio.run(selftest())
    try:
        check_audio()
        return run(args)
    except NoSpeech as e:
        emit(type="warn", text=f"voice cannot start: {e.code}")
        return 2


def run(args) -> int:
    if args.mode == "front":
        return run_front(args)
    sc = Sidecar(args)
    emit(type="ready", port=serve(sc.routes()))
    sc.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
