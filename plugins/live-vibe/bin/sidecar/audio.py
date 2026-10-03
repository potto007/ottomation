"""Audio in and out: device choice, VAD and turn detection, the interruptible player, sentence splitting, and
Voice (TTS pipelined one sentence ahead of playback, cut on demand, recording only what was heard).

Ported from proto/duplex_voice.py (the throwaway prototype): SileroVAD, EnergyVAD, Tuning, TurnDetector, Player,
SentenceSplitter and the speaking half of Session.run_turn. Their mechanics are the prototype's measured ones:
pre-roll, a stricter barge-in gate while the assistant speaks, abort-and-restart for an instant cut, and a cut
sentence committed in proportion to the samples actually played."""
from __future__ import annotations

import asyncio
import collections
import contextlib
import queue
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Protocol

import numpy as np

from .echo import EchoCanceller, EchoGuard, EchoReference, make_canceller, stream_time
from .protocol import log, warn

SR = 16_000  # mic sample rate
FRAME = 512  # 32 ms at 16 kHz, the chunk Silero v5 expects
JUNK_TRANSCRIPTS = {"", ".", "you", "thank you.", "thanks.", "thank you", "bye.", "okay."}


class CannotStart(RuntimeError):
    """A missing system piece voice cannot work without (PortAudio, a device, a synthesizer): main reports it in
    one warn naming the fix and exits 2, before minutes of model loading where it can."""


def load_sounddevice():
    """sounddevice bundles PortAudio on macOS and Windows; on Linux it needs the system library."""
    try:
        import sounddevice as sd
    except OSError as e:  # "PortAudio library not found"
        raise CannotStart(f"PortAudio is missing ({e}). Fix: sudo apt install libportaudio2 "
                          "(Fedora: sudo dnf install portaudio).") from e
    return sd


def pick_device(sd, spec: str, kind: str) -> int | None:
    """--mic / --speaker: a device index, or a case-insensitive name substring; empty is the system default."""
    if not spec:
        return None
    channels = "max_input_channels" if kind == "input" else "max_output_channels"
    devices = list(sd.query_devices())
    if spec.isdigit():
        i = int(spec)
        if i < len(devices) and devices[i][channels] > 0:
            return i
    else:
        for i, d in enumerate(devices):
            if d[channels] > 0 and spec.lower() in d["name"].lower():
                return i
    names = ", ".join(f"{i}: {d['name']}" for i, d in enumerate(devices) if d[channels] > 0)
    warn(f"no {kind} device matches {spec!r}; using the default. Available: {names}")
    return None


def check_devices(sd, mic: int | None, speaker: int | None) -> None:
    """A mic and a speaker exist (a headless or SSH session has neither), and the prototype's heuristic: a laptop's
    own speakers and mic together bleed playback into the mic."""
    names = []
    for device, kind in ((mic, "input"), (speaker, "output")):
        try:
            names.append(sd.query_devices(device, kind)["name"].lower())
        except Exception as e:  # noqa: BLE001 - PortAudioError or ValueError: no such device
            raise CannotStart(f"no {kind} device ({e}). Voice needs a local microphone and speaker.") from e
    din, dout = names
    log(f"audio: mic {din!r}, speaker {dout!r}")
    if ("speaker" in dout and "microphone" in din) or ("built-in" in dout and "built-in" in din):
        warn("built-in speakers and mic: your own answers bleed into the mic. Barge-in uses a stricter gate, "
             "but headphones work much better.")


# -- VAD and turn detection ------------------------------------------------------------------------------------
class SileroVAD:
    """Silero VAD v5 via onnxruntime. Call per 512-sample float32 frame; returns P(speech)."""

    def __init__(self, path: str):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
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
    """Fallback when Silero cannot load: an adaptive RMS gate."""

    def __init__(self) -> None:
        self.noise = 2e-3

    def reset(self) -> None:
        pass

    def __call__(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(frame * frame)) + 1e-9)
        if rms < self.noise * 2:
            self.noise = 0.98 * self.noise + 0.02 * rms
        return float(min(1.0, max(0.0, (rms / (self.noise + 1e-6) - 2.0) / 6.0)))


@dataclass
class Tuning:
    start_prob: float = 0.5  # P(speech) to open a turn when the floor is free
    start_frames: int = 3  # sustained frames (96 ms)
    bargein_prob: float = 0.8  # stricter while the assistant is speaking
    bargein_frames: int = 8  # 256 ms sustained: speaker bleed rarely does this
    end_prob: float = 0.35  # below this counts as silence
    end_silence_ms: int = 700  # trailing silence that ends a turn
    min_speech_ms: int = 250  # shorter utterances are dropped (clicks, coughs)
    pre_roll_ms: int = 320  # audio kept from before the trigger
    max_utterance_s: float = 30.0
    eot_drain_ms: int = 400  # Kyutai: no new word piece for this long before a semantic end of turn, so the
    # pieces that trail the audio (~0.5 s) land in the same utterance


class Detector(Protocol):
    active: bool

    def feed(self, frame: np.ndarray) -> list[tuple[str, Any]]: ...


class TurnDetector:
    """Feeds VAD frame by frame and returns ('speech_start', p) / ('utterance', audio) / ('discard', None)."""

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

    def feed(self, frame: np.ndarray) -> list[tuple[str, Any]]:
        p = self.vad(frame)
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
                self.pre.clear()
                self.vad.reset()
                events.append(("utterance", audio) if voiced >= self.min_frames else ("discard", None))
        return events


class Listener:
    """Owns the mic. The input callback only queues frames; one thread builds the recognizer (so it loads while
    the main thread loads TTS, and MLX runs on the thread that made it) and turns frames into turn events,
    transcribing on the way, so the session sees speech_start / transcribing / utterance(text) / discard."""

    MAX_FRAMES = 400  # ~12.8 s of audio waiting: beyond it the recognizer has stalled, and old frames are dropped

    def __init__(self, build: Callable[[], tuple[Detector, Callable[[Any], str]]], quit: threading.Event,
                 guard: EchoGuard | None = None, reference: EchoReference | None = None, aec: bool = False):
        self.build, self.quit = build, quit
        self.guard, self.reference, self.aec = guard, reference, aec
        self.canceller: EchoCanceller | None = None
        self.frames: queue.Queue[tuple[np.ndarray, np.ndarray | None]] = queue.Queue(self.MAX_FRAMES)
        self.post: Callable[[str, Any], None] = lambda kind, payload: None
        self.detector: Detector | None = None
        self.error: BaseException | None = None
        self.built = threading.Event()
        self.stream = None
        self.problem: str | None = None  # set by the audio callback, reported by the thread
        self.dropped = 0
        self._clean = np.zeros(0, np.float32)  # cancelled samples short of a frame
        self.thread = threading.Thread(target=self._run, name="listener", daemon=True)

    @property
    def active(self) -> bool:
        """The detector is inside a user turn: the floor is not free."""
        return bool(self.detector and self.detector.active)

    def start(self) -> None:
        self.thread.start()

    def wait_built(self) -> bool:
        while not self.built.wait(0.2):
            if self.quit.is_set():
                return False
        return self.detector is not None

    def open_mic(self, sd, device: int | None) -> None:
        # ponytail: capture asks the host API for 16 kHz mono; CoreAudio, PulseAudio/PipeWire and MME resample.
        # A raw ALSA hw device that refuses 16 kHz fails here; resample in _callback if that ever matters.
        self.stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=FRAME, device=device,
                                     callback=self._callback)
        self.stream.start()

    def _callback(self, indata, frames, time_info, status) -> None:
        try:
            if status:
                self.problem = f"mic: {status}"
            ref = None
            if self.reference is not None:  # what the speaker played while this block was captured
                adc = stream_time(time_info.inputBufferAdcTime, time_info.currentTime)
                ref = self.reference.take(frames, adc, SR, time_info.currentTime)
            self.frames.put_nowait((indata[:, 0].copy(), ref))
        except queue.Full:
            self.dropped += 1
        except Exception as e:  # noqa: BLE001 - an exception escaping a PortAudio callback stops the stream
            self.problem = f"mic callback: {type(e).__name__}: {e}"

    def _frames(self, mic: np.ndarray, ref: np.ndarray | None) -> list[np.ndarray]:
        """The detector's frames: the mic as captured, or echo-cancelled and re-cut into FRAME samples."""
        if self.canceller is None or ref is None:
            return [mic]
        self._clean = np.concatenate([self._clean, self.canceller.process(mic, ref)])
        n = len(self._clean) // FRAME
        out = [self._clean[i * FRAME:(i + 1) * FRAME] for i in range(n)]
        self._clean = self._clean[n * FRAME:]
        return out

    def _run(self) -> None:
        try:
            self.detector, transcribe = self.build()
            if self.aec:
                self.canceller = make_canceller(True)
        except BaseException as e:  # noqa: BLE001 - reported by main
            self.error = e
            return
        finally:
            self.built.set()
        failures = 0
        overlapped = False  # the user's turn began while the assistant could be heard
        while not self.quit.is_set():
            try:
                mic, ref = self.frames.get(timeout=0.2)
            except queue.Empty:
                continue
            if self.problem or self.dropped:
                log(self.problem or f"mic: {self.dropped} frames dropped (recognizer behind)")
                self.problem, self.dropped = None, 0
            events: list[tuple[str, Any]] = []
            try:
                for frame in self._frames(mic, ref):
                    events += self.detector.feed(frame)
            except Exception as e:  # noqa: BLE001
                failures += 1
                if failures in (1, 10, 100):
                    log(f"listener: recognizer step failed ({failures}x): {type(e).__name__}: {e}")
                continue
            for kind, payload in events:
                if kind == "speech_start":
                    overlapped = self.guard is not None and self.guard.active()
                if kind != "utterance":
                    self.post(kind, payload)
                    continue
                self.post("transcribing", None)
                try:
                    text = transcribe(payload).strip()
                except Exception as e:  # noqa: BLE001
                    log(f"listener: transcription failed: {type(e).__name__}: {e}")
                    text = ""
                if text and self.guard is not None:
                    kept = self.guard.strip(text, overlapped or self.guard.active())
                    if kept != text:
                        log(f"echo: dropped {text!r}" if not kept else f"echo: {text!r} -> {kept!r}")
                    text = kept
                if text:
                    self.post("utterance", text)
                else:
                    self.post("discard", None)

    def close(self) -> None:
        self.quit.set()
        if self.stream is not None:
            with contextlib.suppress(Exception):
                self.stream.abort()
                self.stream.close()
        if self.thread.is_alive():
            self.thread.join(timeout=2)


# -- output ------------------------------------------------------------------------------------------------
class Player:
    """Owns the output stream. play() blocks (run it in a thread), stops within 10 ms of `cancel` being set, and
    returns how many samples were played. sd=None plays silently in real time (the self-test).

    The stream is callback-driven with LATENCY_S of buffer. Measured on WSL2 (ALSA -> PulseAudio -> WSLg's
    RDPSink), the old latency="low" (8.7 ms) blocking writes underflowed about once a second while Kokoro
    synthesized the next sentence on the CPU (11 in 15 s; none idle), each one an audible crackle; 150 ms had
    none under the same load. A barge-in still cuts at once: abort() drops the buffer. The callback also hands
    what it played to the echo reference, and tells the echo guard how long the speaker sounds."""

    LATENCY_S = 0.15
    STALL_S = 1.0  # no callback for this long while playing: the device is gone

    def __init__(self, sd, sample_rate: int, device: int | None = None, reference: EchoReference | None = None,
                 guard: EchoGuard | None = None):
        self.sample_rate = sample_rate
        self.reference, self.guard = reference, guard
        self.stream = None
        self._lock = threading.Lock()  # one play() at a time; close() waits for it
        self._mu = threading.Lock()  # the clip and position, between play() and the callback
        self._clip: np.ndarray | None = None
        self._pos = 0
        self._done = threading.Event()
        self._last_callback = 0.0
        self._broken = False
        self.underflows = 0
        self.problem: str | None = None  # set by the callback, logged by play()
        if sd is not None:
            self.stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32",
                                          latency=self.LATENCY_S, device=device, callback=self._callback)
            self.stream.start()

    def _callback(self, outdata, frames, time_info, status) -> None:
        try:
            self._last_callback = time.monotonic()
            out = outdata[:, 0]
            n = 0
            with self._mu:
                clip = self._clip
                if clip is not None:
                    n = min(frames, len(clip) - self._pos)
                    out[:n] = clip[self._pos:self._pos + n]
                    self._pos += n
                    if self._pos >= len(clip):
                        self._clip = None
                        self._done.set()
            out[n:] = 0.0
            if status.output_underflow and n:
                self.underflows += 1
            latency = float(self.stream.latency) if self.stream is not None else self.LATENCY_S
            if n and self.guard is not None:
                self.guard.sounding(time.monotonic() + latency + n / self.sample_rate)
            if self.reference is not None:
                dac = stream_time(time_info.outputBufferDacTime, time_info.currentTime, latency)
                self.reference.played(out, dac)
        except Exception as e:  # noqa: BLE001 - an exception escaping a PortAudio callback stops the stream
            outdata.fill(0)
            self.problem = f"speaker callback: {type(e).__name__}: {e}"

    def play(self, audio: np.ndarray, cancel: threading.Event) -> int:
        with self._lock:
            if self.stream is None or self._broken:
                step = int(self.sample_rate * 0.03)
                for i in range(0, len(audio), step):
                    if cancel.is_set():
                        return i
                    cancel.wait(min(step, len(audio) - i) / self.sample_rate)
                return len(audio)
            clip = np.ascontiguousarray(audio, dtype=np.float32)
            if not len(clip):
                return 0
            self._done.clear()
            self._last_callback = time.monotonic()
            under = self.underflows
            with self._mu:
                self._clip, self._pos = clip, 0
            try:
                while not self._done.wait(0.01):
                    if cancel.is_set():
                        with self._mu:
                            played, self._clip = self._pos, None
                        with contextlib.suppress(Exception):
                            self.stream.abort()  # drop what is buffered: the cut is immediate
                            self.stream.start()
                        return played
                    if time.monotonic() - self._last_callback > self.STALL_S:
                        raise RuntimeError("the output stream stopped")
            except Exception as e:  # noqa: BLE001 - an unplugged device must not kill the session
                with self._mu:
                    self._clip = None
                self._broken = True
                warn(f"speaker failed ({type(e).__name__}: {e}); answers continue silently.")
            finally:
                if self.problem:
                    log(self.problem)
                    self.problem = None
                if self.underflows > under:
                    log(f"speaker: {self.underflows - under} underflow(s) in one sentence")
            return len(audio)

    def close(self) -> None:
        got = self._lock.acquire(timeout=1.0)
        try:
            if self.stream is not None:
                with contextlib.suppress(Exception):
                    self.stream.abort()
                    self.stream.close()
                self.stream = None
        finally:
            if got:
                self._lock.release()


_SENTENCE_END = re.compile(r'(?<=[.!?])["\')\]]*\s+')
_MD_NOISE = re.compile(r"[*_`#>]+")
_FENCE = re.compile(r"```.*?```", re.S)
_TABLE_ROW = re.compile(r"^\s*\|.*$", re.M)
_URL = re.compile(r"https?://\S+")


def clean_for_speech(s: str) -> str:
    return _MD_NOISE.sub("", s).replace("\n", " ").strip()


def speakable(text: str) -> str:
    """Drop what cannot be read aloud; clean_for_speech strips the inline markdown per sentence."""
    text = _FENCE.sub(" code omitted. ", text)
    text = _TABLE_ROW.sub("", text)
    return _URL.sub("a link", text)


class SentenceSplitter:
    """Accumulates streamed text and returns speakable sentences as soon as they close."""

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
        return [c for c in map(clean_for_speech, out) if c]

    def flush(self) -> list[str]:
        s, self.buf = clean_for_speech(self.buf), ""
        return [s] if s else []


def split_all(text: str) -> list[str]:
    sp = SentenceSplitter()
    return sp.push(speakable(text)) + sp.flush()


async def aiter_list(items: list[str]) -> AsyncIterator[str]:
    for s in items:
        yield s


class Voice:
    """Speaks a stream of sentences: synthesis runs one sentence ahead of playback, so the next sentence is ready
    when the current one ends. Cancelling the task that awaits speak() cuts playback at once and still records
    the part that was heard (play() is waited for before anything is committed)."""

    def __init__(self, tts, player: Player, speaking: threading.Event, guard: EchoGuard | None = None):
        self.tts, self.player = tts, player
        self.speaking = speaking  # set from the first sentence until speak() ends: the barge-in gate reads it
        self.guard = guard  # remembers each sentence as it starts, so its echo can be told from the user

    async def _synth(self, text: str) -> np.ndarray | None:
        try:
            return await asyncio.to_thread(self.tts.synth, text)
        except Exception as e:  # noqa: BLE001 - one unspeakable sentence is skipped, not fatal
            log(f"tts: failed on {text[:40]!r}: {type(e).__name__}: {e}")
            return None

    async def speak(self, sentences: AsyncIterator[str], heard: list[str],
                    on_play: Callable[[], None] = lambda: None) -> bool:
        """Plays every sentence; appends each one heard (a cut one in proportion, ending '...') to `heard`, and
        calls `on_play` as each starts. Returns True when playback stopped early; raises CancelledError when
        cancelled."""
        cancel = threading.Event()
        ready: asyncio.Queue[tuple[str, np.ndarray] | None] = asyncio.Queue(maxsize=1)
        cut = False

        async def play() -> None:
            nonlocal cut
            while (item := await ready.get()) is not None:
                text, audio = item
                self.speaking.set()
                if self.guard is not None:
                    self.guard.spoke(text)
                on_play()
                played = await asyncio.to_thread(self.player.play, audio, cancel)
                if played < len(audio):
                    heard.append(text[: int(len(text) * played / max(1, len(audio)))].rstrip() + "...")
                    cut = True
                    return
                heard.append(text)

        playing = asyncio.create_task(play())

        async def hand(item: tuple[str, np.ndarray] | None) -> bool:
            """Queue for playback; False once playback has ended early."""
            put = asyncio.ensure_future(ready.put(item))
            await asyncio.wait({put, playing}, return_when=asyncio.FIRST_COMPLETED)
            if put.done():
                return True
            put.cancel()
            return False

        try:
            async with contextlib.aclosing(sentences) as it:  # closing it closes the model's HTTP stream too
                async for s in it:
                    audio = await self._synth(s)
                    if audio is not None and len(audio) and not await hand((s, audio)):
                        break
            await hand(None)
            await playing
        finally:
            if not playing.done():  # cancelled or failed: cut now, but let play() report what was heard
                cancel.set()
                await asyncio.wait({playing}, timeout=1.0)
                playing.cancel()
            self.speaking.clear()
        return cut
