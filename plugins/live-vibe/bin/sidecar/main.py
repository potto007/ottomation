#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "numpy>=2",
#   "sounddevice>=0.5",
#   "onnxruntime>=1.20",
#   "faster-whisper>=1.2",
#   "kokoro-onnx>=0.6",
#   "httpx>=0.28",
#   "anthropic>=1.11",
#   "moshi-mlx>=0.3.0; sys_platform == 'darwin' and platform_machine == 'arm64'",
# ]
# ///
"""Audio sidecar for the live-vibe mod. It owns the mic and speaker in two modes:

  --mode live   (/live)      The mod owns the conversation: utterances go to Claude, Claude's answers come
                             back through POST /speak.
  --mode front  (/livevibe)  A voice front runs here: a small fast model the user talks with directly. Its one
                             tool, delegate, hands real work to Claude; Claude's answers come back through
                             POST /event and the front relays them once the floor is free.

Backends: --stt kyutai (Kyutai STT 1B on MLX, Apple silicon; streaming words and its own end of turn; about
2.4 GB of weights on first use; elsewhere it falls back to whisper) or whisper (Silero VAD + faster-whisper,
CUDA when there is a GPU); --tts kokoro (needs espeak-ng) or say (macOS).

stdout carries only these JSON lines (library output goes to stderr):
  {"type":"ready","port":N,"token":"..."}                every POST must send X-Live-Token: <token>
  {"type":"state","state":"loading|listening|user_speaking|transcribing|thinking|speaking"}
  {"type":"utterance","text":"..."}                      live: what the user said
  {"type":"barge_in"}                                    live: playback was stopped (speech over it, or /stop)
  {"type":"spoken","text":"...","cut":b}                 live: what was actually heard of one /speak
  {"type":"delegate","text":"..."}                       front: work for Claude
  {"type":"switch_model","model":"sonnet"}               front: a spoken model switch, for Claude
  {"type":"transcript","role":"user|front","text":"..."} front: the voice conversation, for the screen
  {"type":"warn","text":"..."}                           something the user should see
  {"type":"log","text":"..."}
HTTP on 127.0.0.1:N:  POST /speak <text> (live)   POST /event <text> (front)   POST /stop   POST /quit
A missing system piece (PortAudio, a mic or speaker, espeak-ng off macOS, say off macOS) is one warn naming the
fix and exit code 2; a backend that fails to load otherwise falls back with a warn.

  uv run --script bin/sidecar/main.py --setup       /live setup: check the system, fetch the models, test the voice
  uv run --script bin/sidecar/main.py --unit        pure checks, a second, no models
  uv run --script bin/sidecar/main.py --selftest    both modes end to end: no mic, speaker or network
"""
from __future__ import annotations

import os
import sys

# Run as a script, Python puts this folder first on sys.path, where audio.py or session.py could shadow a
# library's import; replace it with the parent, so these modules import as the package `sidecar`.
_here = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0] or ".") == _here:
    sys.path[0] = os.path.dirname(_here)
else:
    sys.path.insert(0, os.path.dirname(_here))

import argparse  # noqa: E402
import asyncio  # noqa: E402
import re  # noqa: E402
import secrets  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

from sidecar import protocol  # noqa: E402
from sidecar.protocol import emit, log, warn  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["live", "front"], default="live")
    ap.add_argument("--stt", choices=["kyutai", "whisper"], default="kyutai", help="speech-to-text backend")
    ap.add_argument("--asr", default="base.en", help="faster-whisper size for --stt whisper")
    ap.add_argument("--tts", choices=["kokoro", "say"], default="kokoro")
    ap.add_argument("--voice", default="", help="Kokoro voice (af_heart) or say voice (Samantha)")
    ap.add_argument("--end-silence-ms", type=int, default=1500)
    ap.add_argument("--mic", default="", help="input device: index or name substring (see --list-devices)")
    ap.add_argument("--speaker", default="", help="output device: index or name substring")
    ap.add_argument("--front-backend", choices=["llamacpp", "anthropic"], default="llamacpp")
    ap.add_argument("--front-url", default="http://127.0.0.1:8080", help="OpenAI-compatible server (llama-server)")
    ap.add_argument("--front-model", default="", help="empty: the server's model, or claude-haiku-4-5 for anthropic")
    ap.add_argument("--switch-pattern", default="", help="regex whose group 1 is a model a spoken switch names")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--setup", action="store_true", help="check the system, fetch the models, test the voice")
    ap.add_argument("--unit", action="store_true", help="pure unit checks")
    ap.add_argument("--selftest", action="store_true", help="both modes without mic, speaker or network")
    ap.add_argument("--fake-audio", action="store_true", help=argparse.SUPPRESS)  # the self-test's lifecycle child
    return ap.parse_args(argv)


def run(args: argparse.Namespace, life: protocol.Lifecycle) -> int:
    from sidecar import audio, speech

    emit(type="state", state="loading")
    t0 = time.monotonic()
    speech.check_tts(args.tts)
    sd = None if args.fake_audio else audio.load_sounddevice()
    mic = speaker = None
    if sd is not None:
        mic, speaker = audio.pick_device(sd, args.mic, "input"), audio.pick_device(sd, args.speaker, "output")
        audio.check_devices(sd, mic, speaker)

    # The recognizer loads on the listener thread while this one loads the synthesizer.
    speaking = threading.Event()
    if args.fake_audio:
        build = lambda: (speech.NoSpeech(), str)  # noqa: E731
    else:
        build = lambda: speech.recognizer(args.stt, args.asr, args.end_silence_ms, speaking.is_set)  # noqa: E731
    listener = audio.Listener(build, life.quit)
    listener.start()
    try:
        tts = speech.SilentTTS() if args.fake_audio else speech.make_tts(args.tts, args.voice)
    except BaseException:
        listener.close()
        raise
    try:
        player = audio.Player(sd, tts.sample_rate, speaker)
    except Exception as e:  # noqa: BLE001 - no speaker: keep listening, answer silently
        warn(f"speaker unavailable ({type(e).__name__}: {str(e)[:160]}); answers are not read aloud.")
        player = audio.Player(None, tts.sample_rate)
    try:
        if not listener.wait_built():
            if listener.error is not None:
                e = listener.error
                warn(f"speech recognition cannot start ({type(e).__name__}: {str(e)[:200]}).")
                return 1
            return 0  # quit while loading
        voice = audio.Voice(tts, player, speaking)
        return asyncio.run(serve(args, life, listener, voice, sd, mic, t0))
    finally:
        listener.close()
        player.close()


async def serve(args, life: protocol.Lifecycle, listener, voice, sd, mic, t0: float) -> int:
    from sidecar.front import FrontSession, make_brain, warm_up
    from sidecar.session import LiveSession

    brain = None
    if args.mode == "front":
        brain = make_brain(args.front_backend, args.front_url, args.front_model)
        switch = re.compile(args.switch_pattern) if args.switch_pattern else None
        sess = FrontSession(voice, lambda: listener.active, brain, switch, lambda: life.request_quit("goodbye"))
    else:
        sess = LiveSession(voice, lambda: listener.active)
    listener.post = sess.post

    def quit_(_body: str) -> bool:
        life.request_quit("/quit")
        return True

    routes = {**sess.routes(), "/quit": quit_}
    token = secrets.token_urlsafe(32)
    server = protocol.serve(routes, token)
    life.on_quit(sess.stop)
    try:
        if sd is not None:
            listener.open_mic(sd, mic)
        if brain is not None:
            sess.spawn(warm_up(brain))  # off the startup path: a cold model load must not hold `ready`
        emit(type="ready", port=server.server_port, token=token)
        log(f"ready in {time.monotonic() - t0:.1f}s")
        sess.set_state("listening")
        await sess.run()
    finally:
        server.shutdown()
        server.server_close()
    log(f"stopped: {life.reason or 'session ended'}")
    return 0


def main() -> int:
    args = parse_args()
    if args.unit:
        from sidecar import checks

        return checks.units()
    if args.selftest:
        from sidecar import checks

        return checks.selftest()
    if args.list_devices:
        from sidecar import audio

        print(audio.load_sounddevice().query_devices())
        return 0
    protocol.claim_stdout()
    if args.setup:
        from sidecar import prepare

        return prepare.run(args)
    life = protocol.Lifecycle()
    life.install()
    from sidecar.audio import CannotStart

    try:
        return run(args, life)
    except CannotStart as e:
        warn(f"voice cannot start: {e}")
        return 2
    except Exception as e:  # noqa: BLE001 - say why before going
        warn(f"sidecar failed: {type(e).__name__}: {str(e)[:200]}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Skip interpreter teardown: a daemon thread may still sit in MLX or PortAudio, and every stream is closed.
    os._exit(code)
