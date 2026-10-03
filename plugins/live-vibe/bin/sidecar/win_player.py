# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "numpy>=2",
#   "sounddevice>=0.5",
# ]
# ///
"""The sidecar's speaker on Windows, for WSL: the sidecar (winplayer.py) starts this through WSL interop with uv.exe
and streams the synthesized speech over its stdin, so playback goes straight to WASAPI instead of through WSLg's RDP
audio, which crackles. Standalone on purpose: it imports nothing from the sidecar and runs on Windows Python.

stdin, binary frames: a header "<cI" (kind, payload length), then the payload.
  J  JSON control: {"op":"open","rate":24000}  {"op":"cancel","id":n}  {"op":"ping","t":x}  {"op":"quit"}
  A  audio: "<III" (clip id, clip length, offset of this chunk; all in samples), then float32 mono samples
stdout, one JSON object per line ("t" names it):
  hello   the process runs: pid, python
  open    the stream: device, hostapi, rate, device_rate, latency (s), how (wasapi, wasapi-convert, default)
  played  id, at, n, dac: samples at..at+n of clip id reach the speaker at perf-counter time dac
  end     id, n: the clip's last sample went to the device
  cut     id, played, ms: a cancel took effect after `played` samples of clip id; ms to drop the device buffer
  pong    t (the sidecar's), w (this process's perf counter on receipt)
  xrun    total device underflows so far
  error   text; bye  why, frames, xruns
Times are time.perf_counter() (QueryPerformanceCounter on Windows): the sidecar maps them to its clock with ping and
pong. The process exits on stdin EOF (the sidecar closed it, exited or crashed), on quit, and when no frame arrives
for --watchdog seconds (the sidecar sends a ping every second), so it cannot outlive the sidecar.

  uv run --script win_player.py [--speaker NAME] [--latency S] [--watchdog S] [--list] [--fake]
--fake drives the same engine from a timer instead of a device: the sidecar's unit checks run it on Linux."""
from __future__ import annotations

import argparse
import collections
import json
import os
import queue
import struct
import sys
import threading
import time
from typing import Any

import numpy as np

HEADER = struct.Struct("<cI")
AUDIO = struct.Struct("<III")
MAX_PAYLOAD = 8 << 20


class Out:
    """JSON lines to stdout from one writer thread: the audio callback only queues."""

    def __init__(self, stream: Any) -> None:
        self.stream = stream
        self.q: queue.SimpleQueue[bytes | None] = queue.SimpleQueue()
        self.thread = threading.Thread(target=self._run, name="out", daemon=True)
        self.thread.start()

    def send(self, **obj: Any) -> None:
        self.q.put((json.dumps(obj, separators=(",", ":")) + "\n").encode())

    def _run(self) -> None:
        while (line := self.q.get()) is not None:
            try:
                self.stream.write(line)
                self.stream.flush()
            except (OSError, ValueError):
                os._exit(0)  # the sidecar is gone

    def close(self) -> None:
        self.q.put(None)
        self.thread.join(1.0)


class Clip:
    def __init__(self, cid: int, total: int) -> None:
        self.id, self.total = cid, total
        self.chunks: collections.deque[np.ndarray] = collections.deque()
        self.consumed = 0  # samples handed to the device


class Engine:
    """Clips in order; the device callback takes from the head. A clip whose next samples have not arrived yet plays
    silence until they do (the sidecar sends a whole clip at once, so that only happens on a slow pipe)."""

    FOLLOW = 0.02
    JUMP_S = 0.02

    def __init__(self, out: Out, rate: int) -> None:
        self.out, self.rate = out, rate
        self.lock = threading.Lock()
        self.clips: collections.deque[Clip] = collections.deque()
        self.dropped: set[int] = set()  # cancelled ids: late chunks of them are ignored
        self.frames = 0  # device frames so far
        self.xruns = 0
        self.t: float | None = None  # perf-counter DAC time of frame self.frames, kept continuous

    def add(self, cid: int, total: int, offset: int, samples: np.ndarray) -> None:
        with self.lock:
            if cid in self.dropped:
                return
            clip = next((c for c in self.clips if c.id == cid), None)
            if clip is None:
                clip = Clip(cid, total)
                self.clips.append(clip)
            if len(samples):
                clip.chunks.append(samples)

    def cancel(self, cid: int) -> int:
        """Drops every clip up to and including cid; returns how much of cid was played."""
        with self.lock:
            played = 0
            self.dropped.add(cid)
            while self.clips and self.clips[0].id <= cid:
                c = self.clips.popleft()
                if c.id == cid:
                    played = c.consumed
            return played

    def clock(self, reported: float) -> float:
        """The DAC time of this block's first frame: anchored, advanced by frame count, following the report slowly
        and re-anchoring on a jump (a restarted stream), so consecutive blocks never overlap or leave gaps."""
        if self.t is None or abs(reported - self.t) > self.JUMP_S:
            self.t = reported
        else:
            self.t += self.FOLLOW * (reported - self.t)
        return self.t

    def fill(self, out: np.ndarray, dac: float, underflow: bool) -> None:
        frames = len(out)
        start = self.clock(dac)
        self.t = start + frames / self.rate
        self.frames += frames
        if underflow:
            self.xruns += 1
            self.out.send(t="xrun", total=self.xruns)
        n = 0
        with self.lock:
            while n < frames and self.clips:
                c = self.clips[0]
                if not c.chunks:
                    if c.consumed >= c.total:
                        self.clips.popleft()
                        self.out.send(t="end", id=c.id, n=c.consumed)
                        continue
                    break  # the rest of this clip has not arrived
                chunk = c.chunks[0]
                k = min(frames - n, len(chunk))
                out[n:n + k] = chunk[:k]
                self.out.send(t="played", id=c.id, at=c.consumed, n=k, dac=round(start + n / self.rate, 6))
                c.consumed += k
                n += k
                if k == len(chunk):
                    c.chunks.popleft()
                else:
                    c.chunks[0] = chunk[k:]
                if not c.chunks and c.consumed >= c.total:
                    self.clips.popleft()
                    self.out.send(t="end", id=c.id, n=c.consumed)
        out[n:] = 0.0


def wasapi_index(sd: Any) -> int | None:
    for i, h in enumerate(sd.query_hostapis()):
        if "WASAPI" in h["name"]:
            return i
    return None


def pick(sd: Any, spec: str) -> tuple[int | None, str]:
    """A WASAPI output device whose name contains `spec`, else WASAPI's default output, else the host's default."""
    api = wasapi_index(sd)
    if api is None:
        return None, "no WASAPI host API"
    devices = list(sd.query_devices())
    if spec:
        for i, d in enumerate(devices):
            if d["hostapi"] == api and d["max_output_channels"] > 0 and spec.lower() in d["name"].lower():
                return i, f"matches {spec!r}"
    default = sd.query_hostapis(api)["default_output_device"]
    note = f"no WASAPI output matches {spec!r}; Windows default" if spec else "Windows default"
    return (default if default >= 0 else None), note


def open_stream(sd: Any, engine: Engine, spec: str, latency: float, out: Out) -> Any:
    """WASAPI shared mode at the clip rate, Windows converting to the mix format (auto_convert); then plain WASAPI
    (a device whose mix rate is the clip rate); last, the default host API (MME), which resamples too. No resampling
    here, so the sample positions this reports are the clip's own."""
    device, note = pick(sd, spec)

    def callback(outdata: Any, frames: int, time_info: Any, status: Any) -> None:
        try:
            now = time.perf_counter()
            dac, cur = time_info.outputBufferDacTime, time_info.currentTime
            ahead = dac - cur if dac and cur and 0 <= dac - cur < 2.0 else stream.latency
            engine.fill(outdata[:, 0], now + ahead, bool(status.output_underflow))
        except Exception as e:  # noqa: BLE001 - an exception escaping the callback stops the stream
            outdata.fill(0)
            out.send(t="error", text=f"callback: {type(e).__name__}: {e}")

    tries: list[tuple[str, int | None, Any]] = []
    if device is not None:
        tries.append(("wasapi-convert", device, sd.WasapiSettings(auto_convert=True)))
        tries.append(("wasapi", device, None))
    tries.append(("default", None, None))
    errors = []
    for how, dev, extra in tries:
        try:
            stream = sd.OutputStream(samplerate=engine.rate, channels=1, dtype="float32", latency=latency,
                                     device=dev, callback=callback, extra_settings=extra)
            stream.start()
        except Exception as e:  # noqa: BLE001
            errors.append(f"{how}: {type(e).__name__}: {e}")
            continue
        info = sd.query_devices(stream.device)
        out.send(t="open", device=info["name"], hostapi=sd.query_hostapis(info["hostapi"])["name"], rate=engine.rate,
                 device_rate=int(info["default_samplerate"]), latency=round(float(stream.latency), 4), how=how,
                 note=note, tried=errors)
        return stream
    raise RuntimeError("; ".join(errors))


class FakeStream:
    """A timer in place of a device: 10 ms blocks, DAC time 30 ms ahead."""

    latency = 0.03

    def __init__(self, engine: Engine, out: Out) -> None:
        self.engine = engine
        self.quit = threading.Event()
        out.send(t="open", device="fake", hostapi="none", rate=engine.rate, device_rate=engine.rate,
                 latency=self.latency, how="fake", note="", tried=[])
        threading.Thread(target=self._run, name="fake-device", daemon=True).start()

    def _run(self) -> None:
        block = self.engine.rate // 100
        buf = np.zeros(block, np.float32)
        nxt = time.perf_counter()
        while not self.quit.is_set():
            self.engine.fill(buf, time.perf_counter() + self.latency, False)
            nxt += block / self.engine.rate
            time.sleep(max(0.0, nxt - time.perf_counter()))

    def abort(self) -> None:
        self.quit.set()

    def close(self) -> None:
        self.quit.set()


def read_exact(f: Any, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        got = f.read(n - len(buf))
        if not got:
            return None
        buf += got
    return buf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", default="", help="part of a WASAPI output device's name; empty: the Windows default")
    ap.add_argument("--latency", type=float, default=0.05, help="output buffer, seconds")
    ap.add_argument("--watchdog", type=float, default=15.0, help="exit after this long without a frame")
    ap.add_argument("--list", action="store_true", help="print the WASAPI output devices as JSON and exit")
    ap.add_argument("--fake", action="store_true", help="no device: a timer consumes the audio")
    args = ap.parse_args()
    if args.list:
        import sounddevice as sd

        api = wasapi_index(sd)
        rows = [{"index": i, "name": d["name"], "rate": d["default_samplerate"]} for i, d in enumerate(sd.query_devices())
                if d["hostapi"] == api and d["max_output_channels"] > 0]
        print(json.dumps(rows))
        return 0

    inp = sys.stdin.buffer
    out = Out(sys.stdout.buffer)
    out.send(t="hello", pid=os.getpid(), python=sys.version.split()[0])
    last_rx = [time.monotonic()]
    stream: Any = None
    engine: Engine | None = None
    why = "stdin closed"

    def watchdog() -> None:
        while True:
            time.sleep(0.5)
            if time.monotonic() - last_rx[0] > args.watchdog:
                out.send(t="bye", why="watchdog: the sidecar went quiet", frames=engine.frames if engine else 0,
                         xruns=engine.xruns if engine else 0)
                out.close()
                with_stream_abort(stream)
                os._exit(0)

    threading.Thread(target=watchdog, name="watchdog", daemon=True).start()
    try:
        while True:
            head = read_exact(inp, HEADER.size)
            if head is None:
                break
            kind, size = HEADER.unpack(head)
            if size > MAX_PAYLOAD:
                why = f"frame too large ({size})"
                break
            body = read_exact(inp, size)
            if body is None:
                break
            last_rx[0] = time.monotonic()
            if kind == b"A":
                if engine is None:
                    continue
                cid, total, offset = AUDIO.unpack_from(body)
                engine.add(cid, total, offset, np.frombuffer(body, np.float32, offset=AUDIO.size).copy())
                continue
            msg = json.loads(body)
            op = msg.get("op")
            if op == "ping":
                out.send(t="pong", t0=msg.get("t"), w=time.perf_counter())
            elif op == "open" and engine is None:
                engine = Engine(out, int(msg["rate"]))
                try:
                    if args.fake:
                        stream = FakeStream(engine, out)
                    else:
                        import sounddevice as sd

                        stream = open_stream(sd, engine, args.speaker, args.latency, out)
                except Exception as e:  # noqa: BLE001
                    out.send(t="error", text=f"cannot open the speaker: {type(e).__name__}: {e}")
                    why = "no speaker"
                    break
            elif op == "cancel" and engine is not None:
                cid = int(msg["id"])
                t = time.perf_counter()
                played = engine.cancel(cid)
                if played and stream is not None and not args.fake:
                    try:  # drop what the device still buffers, as the local player does: the cut is immediate
                        stream.abort()
                        stream.start()
                    except Exception as e:  # noqa: BLE001
                        out.send(t="error", text=f"restart after cancel: {type(e).__name__}: {e}")
                out.send(t="cut", id=cid, played=played, ms=round(1000 * (time.perf_counter() - t), 1))
            elif op == "quit":
                why = "quit"
                break
    except Exception as e:  # noqa: BLE001
        out.send(t="error", text=f"{type(e).__name__}: {e}")
        why = "error"
    with_stream_abort(stream)
    out.send(t="bye", why=why, frames=engine.frames if engine else 0, xruns=engine.xruns if engine else 0)
    out.close()
    sys.stdout.flush()
    os._exit(0)


def with_stream_abort(stream: Any) -> None:
    if stream is None:
        return
    try:
        stream.abort()
        stream.close()
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    sys.exit(main())
