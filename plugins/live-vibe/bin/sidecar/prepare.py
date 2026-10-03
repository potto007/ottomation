"""--setup (/live setup): everything a first /live would need, checked and fetched now, with a line per step.

By the time this runs, `uv run --script` has installed the Python packages. It then checks PortAudio and the
devices, loads (downloading on first use) the synthesizer and the recognizer the settings name, speaks a sentence
into the recognizer to prove the pair works, plays one sentence through the speaker, and asks the front server what
it serves. Nothing here installs a system package: a missing one is a `fail` line naming the command.

stdout lines, on top of log and warn:
  {"type":"progress","text":"..."}                        a slow step starting
  {"type":"check","name":"...","status":"ok|warn|fail","text":"..."}
  {"type":"done","ok":b}                                   ok: no step failed
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import threading
import time

import numpy as np

from .audio import FRAME, SR, CannotStart
from .protocol import Lifecycle, emit

PROBE = "Please open the parser file and fix the failing test."


class Report:
    def __init__(self) -> None:
        self.ok = True

    def __call__(self, name: str, status: str, text: str) -> None:
        self.ok = self.ok and status != "fail"
        emit(type="check", name=name, status=status, text=text)


def progress(text: str) -> None:
    emit(type="progress", text=text)


def run(args: argparse.Namespace) -> int:
    life = Lifecycle()
    life.install()
    life.on_quit(lambda: os._exit(130))  # nothing to clean up: partial downloads are .part / .incomplete files
    check = Report()
    emit(type="check", name="python", status="ok",
         text=f"Python {sys.version.split()[0]} with the sidecar's packages (uv installed them)")

    sd, mic, speaker = audio_devices(check, args)
    tts = synthesizer(check, args)
    recognized = recognizer(check, args)
    if tts is not None and recognized is not None:
        round_trip(check, tts, *recognized)
    if tts is not None and sd is not None:
        play(check, sd, tts, speaker)
    front(check, args)
    emit(type="done", ok=check.ok)
    return 0 if check.ok else 1


def audio_devices(check: Report, args: argparse.Namespace):
    from . import audio

    try:
        sd = audio.load_sounddevice()
    except CannotStart as e:
        check("audio", "fail", str(e))
        return None, None, None
    mic, speaker = audio.pick_device(sd, args.mic, "input"), audio.pick_device(sd, args.speaker, "output")
    try:
        audio.check_devices(sd, mic, speaker)
    except CannotStart as e:
        check("audio", "fail", str(e))
        return None, None, None
    names = [sd.query_devices(d, k)["name"] for d, k in ((mic, "input"), (speaker, "output"))]
    check("audio", "ok", f"PortAudio; mic {names[0]!r}, speaker {names[1]!r}")
    return sd, mic, speaker


def synthesizer(check: Report, args: argparse.Namespace):
    from . import speech

    if args.tts == "kokoro":
        found = speech.find_espeak()
        if found:
            check("espeak-ng", "ok", found[0])
        else:
            check("espeak-ng", "warn", f"no system espeak-ng; Kokoro tries the copy bundled with its wheel. "
                                       f"If it fails below: {speech.espeak_fix()}")
    try:
        speech.check_tts(args.tts)
        if args.tts == "kokoro":
            progress("Kokoro: loading (downloads about 330 MB on first run)")
        t = time.monotonic()
        tts = speech.make_tts(args.tts, args.voice)
    except CannotStart as e:
        check("synthesizer", "fail", str(e))
        return None
    except Exception as e:  # noqa: BLE001
        check("synthesizer", "fail", f"{type(e).__name__}: {str(e)[:200]}")
        return None
    took = time.monotonic() - t
    if isinstance(tts, speech.KokoroTTS):
        check("synthesizer", "ok", f"Kokoro, voice {tts.voice}, ready in {took:.0f}s")
    elif args.tts == "kokoro":
        check("synthesizer", "warn", "Kokoro failed (see the warning above); macOS say speaks instead")
    else:
        check("synthesizer", "ok", "macOS say")
    return tts


def recognizer(check: Report, args: argparse.Namespace):
    from . import speech

    kyutai = args.stt == "kyutai" and speech.is_apple_silicon()
    if kyutai:
        progress("Kyutai STT: loading (downloads about 2.4 GB on first run)")
    else:
        progress(f"Whisper {args.asr} and Silero VAD: loading (downloads about 150 MB on first run)")
    t = time.monotonic()
    try:
        det, transcribe = speech.recognizer(args.stt, args.asr, args.end_silence_ms, lambda: False)
    except Exception as e:  # noqa: BLE001
        check("recognizer", "fail", f"{type(e).__name__}: {str(e)[:200]}")
        return None
    took = time.monotonic() - t
    if isinstance(det, speech.KyutaiTurns):
        check("recognizer", "ok", f"Kyutai STT 1B on MLX, ready in {took:.0f}s")
    else:
        w = getattr(transcribe, "__self__", None)
        where = "an Nvidia GPU (CUDA float16)" if getattr(w, "device", "") == "cuda" else "the CPU (int8)"
        vad = "Silero VAD" if isinstance(getattr(det, "vad", None), speech.SileroVAD) else "an energy gate (Silero failed)"
        note = "; Kyutai is Apple silicon only" if args.stt == "kyutai" else ""
        status = "warn" if args.stt == "kyutai" and speech.is_apple_silicon() else "ok"
        check("recognizer", status, f"Whisper {args.asr} on {where} with {vad}, ready in {took:.0f}s{note}")
    return det, transcribe


def round_trip(check: Report, tts, det, transcribe) -> None:
    """The synthesizer's own voice through the recognizer: both halves load and agree, no mic needed."""
    progress("round trip: speaking a sentence into the recognizer")
    try:
        wav = tts.synth(PROBE)
        speech_16k = np.interp(np.linspace(0, len(wav) - 1, int(len(wav) * SR / tts.sample_rate)),
                               np.arange(len(wav)), wav).astype(np.float32)
        signal_ = np.concatenate([np.zeros(SR, np.float32), speech_16k, np.zeros(SR * 5, np.float32)])
        signal_ += (0.002 * np.random.default_rng(0).standard_normal(len(signal_))).astype(np.float32)
        events = []
        for i in range(0, len(signal_) - FRAME + 1, FRAME):
            events += det.feed(signal_[i:i + FRAME])
        utts = [v for k, v in events if k == "utterance"]
        heard = " ".join(transcribe(u) for u in utts).strip()
    except Exception as e:  # noqa: BLE001
        check("round trip", "fail", f"{type(e).__name__}: {str(e)[:200]}")
        return
    if "test" in heard.lower():
        check("round trip", "ok", f"said {PROBE!r}, heard {heard!r}")
    else:
        check("round trip", "warn", f"said {PROBE!r}, heard {repr(heard) if heard else 'nothing'}")


def play(check: Report, sd, tts, speaker) -> None:
    from .audio import Player

    try:
        player = Player(sd, tts.sample_rate, speaker)
        try:
            player.play(tts.synth("Live voice is set up."), threading.Event())
            time.sleep(0.3)  # let the device drain before the stream closes
        finally:
            player.close()
    except Exception as e:  # noqa: BLE001
        check("speaker", "fail", f"{type(e).__name__}: {str(e)[:200]}")
        return
    check("speaker", "ok", "played \"Live voice is set up.\"; if you did not hear it, set the speaker option")


def front(check: Report, args: argparse.Namespace) -> None:
    """Only /livevibe needs it, so a front that is down warns rather than fails."""
    if args.front_backend == "anthropic":
        try:
            from .front import AnthropicBrain

            brain = AnthropicBrain("", args.front_model)
            asyncio.run(brain.warm_up())
            check("front", "ok", f"{brain.where} answers")
        except Exception as e:  # noqa: BLE001
            check("front", "warn", f"Anthropic: {type(e).__name__}: {str(e)[:160]}. Needs ANTHROPIC_API_KEY or "
                                   "`ant auth login`; only /livevibe uses it")
        return
    import httpx

    url = args.front_url.rstrip("/")
    try:
        r = httpx.get(f"{url}/v1/models", timeout=3.0)
        r.raise_for_status()
        ids = [str(m.get("id")) for m in r.json().get("data", [])]
    except Exception as e:  # noqa: BLE001
        check("front", "warn", f"nothing answers at {url} ({type(e).__name__}); only /livevibe needs it: start "
                               "llama-server with --jinja there, or /livevibe url <url>")
        return
    if args.front_model and args.front_model not in ids:
        check("front", "warn", f"{url} serves {', '.join(ids) or 'nothing'}, not {args.front_model!r}; a router "
                               "loads it on demand, a plain llama-server ignores the name")
    else:
        check("front", "ok", f"{url} serves {', '.join(ids) or 'a model'}")
