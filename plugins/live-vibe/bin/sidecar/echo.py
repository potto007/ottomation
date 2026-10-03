"""Echo handling for full duplex over open speakers: the mic hears the assistant, and without this its own
words come back as user turns (and its own voice barges in on itself).

Three layers, each a fallback for the one before:
  EchoReference + EchoCanceller  WebRTC AEC3 (livekit's audio processing module) on every mic frame, with what
                                 the speaker played, aligned by DAC/ADC time, as the reference. Offline, with a
                                 synthetic room at 150-600 ms of delay, it kept Silero under the barge-in gate
                                 through the echo and kept the user's words over it.
  EchoGuard.active()             the barge-in gate stays strict while playing and for TAIL_S after, since the
                                 speaker keeps sounding after play() returns (output buffer, RDP and room).
  EchoGuard.strip()              drops the runs of an utterance that repeat what was just spoken."""
from __future__ import annotations

import collections
import difflib
import re
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from .protocol import log

APM_RATE = 16_000
APM_BLOCK = APM_RATE // 100  # the module takes exactly 10 ms frames


class StreamClock:
    """A stream's sample position on the shared PortAudio clock: anchored on the first callback's DAC/ADC time,
    then advanced by sample count, so it is continuous; it follows the reported time slowly and re-anchors on a
    jump (a restarted stream)."""

    JUMP_S = 0.25
    FOLLOW = 0.02

    def __init__(self, rate: int):
        self.rate = rate
        self.t: float | None = None

    def place(self, reported: float, frames: int) -> float:
        """Start time of this block; advances past it."""
        if self.t is None or abs(reported - self.t) > self.JUMP_S:
            self.t = reported
        else:
            self.t += self.FOLLOW * (reported - self.t)
        start = self.t
        self.t += frames / self.rate
        return start


class EchoReference:
    """What the speaker played, at the canceller's rate, in a ring indexed by the time it reaches the DAC. The
    output callback writes ahead (played); the mic callback reads the span its block was captured in (take),
    which the output wrote earlier, and clears it so an idle speaker reads as silence."""

    SECONDS = 4

    def __init__(self, out_rate: int):
        self.ratio = APM_RATE / out_rate
        self.ring = np.zeros(APM_RATE * self.SECONDS, np.float32)
        self.out_clock = StreamClock(out_rate)
        self.in_clock: StreamClock | None = None
        self.prev = 0.0
        self.next_k: int | None = None  # the first ring index the next output block may write
        self._lock = threading.Lock()

    def played(self, block: np.ndarray, dac_time: float) -> None:
        start = self.out_clock.place(dac_time, len(block)) * APM_RATE
        pos = start + np.arange(len(block)) * self.ratio
        k0 = int(np.ceil(start))
        if self.next_k is not None and abs(k0 - self.next_k) <= 2:
            k0 = self.next_k  # continuous: no gap or overlap from rounding
        k1 = int(np.ceil(start + len(block) * self.ratio))
        if k1 <= k0:
            return
        x = np.concatenate([[start - self.ratio], pos])
        y = np.concatenate([[self.prev], block])
        vals = np.interp(np.arange(k0, k1, dtype=np.float64), x, y).astype(np.float32)
        self.prev = float(block[-1])
        self.next_k = k1
        idx = np.arange(k0, k1) % len(self.ring)
        with self._lock:
            self.ring[idx] = vals

    def take(self, frames: int, adc_time: float, rate: int) -> np.ndarray:
        """The reference for a mic block of `frames` at `rate` (the mic's) captured from `adc_time`."""
        if self.in_clock is None:
            self.in_clock = StreamClock(rate)
        start = self.in_clock.place(adc_time, frames)
        n = round(frames * APM_RATE / rate)
        idx = (round(start * APM_RATE) + np.arange(n)) % len(self.ring)
        with self._lock:
            out = self.ring[idx].copy()
            self.ring[idx] = 0.0
        if n != frames:
            out = np.interp(np.linspace(0, n - 1, frames), np.arange(n), out).astype(np.float32)
        return out


def stream_time(primary: float, current: float, offset: float = 0.0) -> float:
    """A callback's DAC/ADC time, or its current time (plus the stream latency) where the host API gives none."""
    if primary:
        return primary
    return (current or time.monotonic()) + offset


class EchoCanceller:
    """WebRTC AEC3 with noise suppression and a high-pass filter, run on the listener thread: process() takes
    16 kHz mic samples with the matching reference and returns cleaned samples (10 ms behind)."""

    def __init__(self) -> None:
        from livekit import rtc

        self._rtc = rtc
        self.apm = rtc.AudioProcessingModule(echo_cancellation=True, noise_suppression=True,
                                             high_pass_filter=True, auto_gain_control=False)
        self.mic = np.zeros(0, np.float32)
        self.ref = np.zeros(0, np.float32)
        self.failed = False

    def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        if self.failed:
            return mic
        self.mic = np.concatenate([self.mic, mic])
        self.ref = np.concatenate([self.ref, ref])
        n = len(self.mic) // APM_BLOCK * APM_BLOCK
        out = np.empty(n, np.float32)
        try:
            for i in range(0, n, APM_BLOCK):
                r = self._frame(self.ref[i:i + APM_BLOCK])
                self.apm.process_reverse_stream(r)
                self.apm.set_stream_delay_ms(0)  # the reference is aligned to the DAC already
                m = self._frame(self.mic[i:i + APM_BLOCK])
                self.apm.process_stream(m)
                out[i:i + APM_BLOCK] = np.frombuffer(bytes(m.data), np.int16) / np.float32(32768)
        except Exception as e:  # noqa: BLE001 - echo cancelling is an aid; listening must go on without it
            self.failed = True
            log(f"echo: canceller failed ({type(e).__name__}: {e}); raw mic from here on")
            return mic
        self.mic, self.ref = self.mic[n:], self.ref[n:]
        return out

    def _frame(self, x: np.ndarray) -> Any:
        pcm = (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)
        return self._rtc.AudioFrame(pcm.tobytes(), APM_RATE, 1, APM_BLOCK)


def make_canceller(enabled: bool) -> EchoCanceller | None:
    if not enabled:
        log("echo: canceller off (--aec off)")
        return None
    try:
        c = EchoCanceller()
        c.process(np.zeros(APM_BLOCK, np.float32), np.zeros(APM_BLOCK, np.float32))
    except Exception as e:  # noqa: BLE001
        log(f"echo: canceller unavailable ({type(e).__name__}: {str(e)[:120]}); relying on the barge-in "
            "gate and the transcript filter. Headphones avoid echo entirely.")
        return None
    log("echo: WebRTC AEC3 canceller on")
    return c


_WORD = re.compile(r"[a-z0-9']+")


def words_of(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class EchoGuard:
    """What the assistant said lately and whether its voice may still be in the room."""

    TAIL_S = 0.8  # after the last sample is handed to the device: output buffer, RDP and room
    MEMORY_S = 45.0  # how long a spoken sentence can come back as echo
    RUN = 3  # repeated words in a row that count as echo
    MOSTLY = 0.6  # an utterance this much echo is all echo

    def __init__(self, speaking: Callable[[], bool]):
        self.speaking = speaking
        self.sound_until = 0.0  # time.monotonic() when the last audio handed to the device has been heard
        self.said: collections.deque[tuple[float, list[str]]] = collections.deque(maxlen=64)
        self._lock = threading.Lock()

    def sounding(self, until: float) -> None:
        self.sound_until = max(self.sound_until, until)

    def active(self) -> bool:
        """The barge-in gate applies: speaking, or within the tail of the last sound."""
        return self.speaking() or time.monotonic() < self.sound_until + self.TAIL_S

    def spoke(self, text: str) -> None:
        with self._lock:
            self.said.append((time.monotonic(), words_of(text)))

    def strip(self, text: str, overlapped: bool) -> str:
        """`text` without the runs that repeat recent speech; '' when it is mostly echo. Only an utterance that
        overlapped playback (or its tail) can be echo."""
        if not overlapped:
            return text
        now = time.monotonic()
        with self._lock:
            recent = [w for t, ws in self.said if now - t <= self.MEMORY_S for w in ws]
        heard = words_of(text)
        if not heard or not recent:
            return text
        sm = difflib.SequenceMatcher(None, heard, recent, autojunk=False)
        echo = [False] * len(heard)
        for b in sm.get_matching_blocks():
            if b.size >= self.RUN or (b.size == len(heard) and b.size > 0):
                echo[b.a:b.a + b.size] = [True] * b.size
        n = sum(echo)
        if not n:
            return text
        if n >= self.MOSTLY * len(heard):
            return ""
        return _keep(text, echo)


def _keep(text: str, echo: list[bool]) -> str:
    """The words of `text` (with their punctuation) not marked as echo."""
    out: list[str] = []
    i = 0
    for token in text.split():
        n = len(_WORD.findall(token.lower()))
        if n == 0:
            continue
        if not any(echo[i:i + n]):
            out.append(token)
        i += n
    return " ".join(out)
