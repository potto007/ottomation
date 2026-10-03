"""--setup (/live setup): everything a first /live would need, checked and fetched now, with a line per step.

By the time this runs, `uv run --script` has installed the Python packages. It then checks PortAudio and the
devices, loads (downloading on first use) the synthesizer and the recognizer the settings name, speaks a sentence
into the recognizer to prove the pair works, plays one sentence through the speaker, and asks the front server what
it serves; with frontUrl empty it first fetches the managed llama-server and its model, and starts and stops it once. Nothing here installs a system package: a missing one is a `fail` line naming the command.

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

    log_path = getattr(args, "log_path", None)
    check("log", "ok" if log_path else "warn", f"sidecar log: {log_path}" if log_path else "sidecar log file unavailable")
    sd, mic, speaker = audio_devices(check, args)
    managed_front(check, args)  # first, as in a live run: the front model has the first claim on the GPU
    from . import gpu, speech

    gpu.hold("stt", speech.stt_gpu_need(args.stt))  # as in a live run: the recognizer claims before Kokoro
    tts = synthesizer(check, args)
    recognized = recognizer(check, args)
    if tts is not None and recognized is not None:
        round_trip(check, tts, *recognized)
    if tts is not None and sd is not None:
        play(check, sd, tts, speaker, args)
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
        tts = speech.make_tts(args.tts, args.voice, args.tts_device)
    except CannotStart as e:
        check("synthesizer", "fail", str(e))
        return None
    except Exception as e:  # noqa: BLE001
        check("synthesizer", "fail", f"{type(e).__name__}: {str(e)[:200]}")
        return None
    took = time.monotonic() - t
    if isinstance(tts, speech.KokoroTTS):
        where = "GPU (CUDA)" if tts.provider == "CUDAExecutionProvider" else "CPU"
        check("synthesizer", "ok", f"Kokoro on the {where}, voice {tts.voice}, ready in {took:.0f}s")
    elif args.tts == "kokoro":
        check("synthesizer", "warn", "Kokoro failed (see the warning above); macOS say speaks instead")
    else:
        check("synthesizer", "ok", "macOS say")
    return tts


def recognizer(check: Report, args: argparse.Namespace):
    from . import speech
    from .kyutai_cuda import kyutai_backend

    backend, why = kyutai_backend() if args.stt == "kyutai" else (None, "")
    if backend:
        progress(f"Kyutai STT on {backend.upper()}: loading (downloads about 2.4 GB on first run)")
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
        where = getattr(det.stt, "where", "MLX")
        check("recognizer", "ok", f"Kyutai STT 1B on {where}, ready in {took:.0f}s")
    else:
        w = getattr(transcribe, "__self__", None)
        where = "an Nvidia GPU (CUDA float16)" if getattr(w, "device", "") == "cuda" else "the CPU (int8)"
        vad = "Silero VAD" if isinstance(getattr(det, "vad", None), speech.SileroVAD) else "an energy gate (Silero failed)"
        note = f"; Kyutai needs Apple silicon or an Nvidia GPU: {why}" if why else ""
        status = "warn" if backend else "ok"  # Kyutai could run here but failed to load
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


def play(check: Report, sd, tts, speaker, args: argparse.Namespace) -> None:
    from . import winplayer
    from .audio import Player

    backend, why = winplayer.resolve(getattr(args, "speaker_backend", "auto"))
    if backend == "windows" and windows_speaker(check, tts, args):
        return
    if backend == "local" and getattr(args, "speaker_backend", "auto") == "windows":
        check("windows speaker", "warn", f"{why}; the local speaker plays instead")
    try:
        player = Player(sd, tts.sample_rate, speaker)
        try:
            player.play(tts.synth("Live voice is set up."), threading.Event())
            time.sleep(Player.LATENCY_S + 0.3)  # let the device drain before the stream closes
        finally:
            player.close()
    except Exception as e:  # noqa: BLE001
        check("speaker", "fail", f"{type(e).__name__}: {str(e)[:200]}")
        return
    check("speaker", "ok", "played \"Live voice is set up.\"; if you did not hear it, set the speaker option")


def windows_speaker(check: Report, tts, args: argparse.Namespace) -> bool:
    """Under WSL: prepare the Windows player (uv.exe, then Python, numpy and sounddevice in its own cache under
    %LOCALAPPDATA%\\live-vibe) and play the sentence through it. False: it failed, and the local speaker is tried."""
    from . import winplayer

    progress("Windows player: preparing uv, Python and sounddevice on Windows (first run downloads about 60 MB)")
    launch = winplayer.Launch(args.speaker, prepare=True, progress=progress).start()
    p = launch.player(tts.sample_rate, timeout=winplayer.SETUP_TIMEOUT_S, report=lambda _: None)
    if p is None:
        check("windows speaker", "warn", f"unavailable ({launch.error[:200]}); /live plays through WSLg, which "
                                         "can crackle. speakerBackend local skips this")
        return False
    try:
        p.play(tts.synth("Live voice is set up."), threading.Event())
        time.sleep(float(p.info.get("latency", 0.05)) + 0.3)  # let the device drain before the stream closes
        broken = p.broken
    finally:
        p.close()
    check("speaker", "warn" if broken else "ok", f"{p.summary()}; played \"Live voice is set up.\" with "
          f"{p.underflows} underflow(s); if you did not hear it, set the speaker option (part of a Windows device name)")
    return not broken


def front(check: Report, args: argparse.Namespace) -> None:
    """Only /livevibe needs it, so a front that is down warns rather than fails."""
    from . import front_server

    if front_server.managed(args):
        return  # managed_front() checked it
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


def managed_front(check: Report, args: argparse.Namespace) -> None:
    """frontUrl empty: fetch (or find) llama-server and the model, start it as /livevibe would, stop it."""
    from . import front_server as fs

    if not fs.managed(args):
        return
    s = fs.settings(args)
    try:
        binary = fs.ensure_binary(s, progress)
        model = fs.ensure_model(s, progress)
    except Exception as e:  # noqa: BLE001 - only /livevibe needs it
        check("front", "warn", f"managed llama-server: {str(e)[:300]}. Only /livevibe needs it; set frontServerBin "
                               "and frontServerModel to files you have, or frontUrl to a server")
        return
    progress(f"front: starting llama-server with {model.name}")
    server = fs.FrontServer(s, progress).start()
    try:
        ok = server.wait_ready()
    finally:
        server.stop()
    if not ok:
        check("front", "warn", f"managed llama-server did not start: {server.error[:300]}")
        return
    where = "GPU" if server.on_gpu else f"CPU ({server.where}; slow)"
    check("front", "ok", f"managed llama-server {fs.version(binary)} with {model.name} on {where}, ready in "
                         f"{server.ready_s:.0f}s; log {server.log_file}")
