#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "numpy>=2",
#   "sounddevice>=0.5",
#   "onnxruntime>=1.20; sys_platform != 'linux' or platform_machine != 'x86_64'",
#   "onnxruntime-gpu[cuda,cudnn]>=1.22,<1.27; sys_platform == 'linux' and platform_machine == 'x86_64'",
#   "faster-whisper>=1.2",
#   "kokoro-onnx>=0.6",
#   "httpx>=0.28",
#   "anthropic>=1.11",
#   "moshi-mlx>=0.3.0; sys_platform == 'darwin' and platform_machine == 'arm64'",
#   "moshi>=0.2.11,<0.3; sys_platform == 'linux' and platform_machine == 'x86_64'",
#   "livekit>=1.0",
# ]
# [tool.uv]
# # kokoro-onnx requires onnxruntime, which would overwrite onnxruntime-gpu's files (one module, two wheels).
# # onnxruntime-gpu stays below 1.27, the last CUDA 12 build: torch (moshi) and CTranslate2 use CUDA 12, and
# # cuDNN's cu12 and cu13 wheels share the soname libcudnn.so.9, so one process should load one family.
# override-dependencies = ["onnxruntime>=1.20; sys_platform != 'linux' or platform_machine != 'x86_64'"]
# ///
"""Audio sidecar for the live-vibe mod. It owns the mic and speaker in two modes:

  --mode live   (/live)      The mod owns the conversation: utterances go to Claude, Claude's answers come
                             back through POST /speak.
  --mode front  (/livevibe)  A voice front runs here: a small fast model the user talks with directly. Its one
                             tool, delegate, hands real work to Claude; Claude's answers come back through
                             POST /event and the front relays them once the floor is free.

Echo: --aec on runs WebRTC AEC3 (livekit) on the mic with what the speaker played as its reference; with
or without it, barge-in stays strict for a tail after playback and utterances that repeat recent speech are dropped.

Backends: --stt kyutai (Kyutai STT 1B on MLX on Apple silicon, or on CUDA PyTorch on Linux with an Nvidia GPU;
streaming words and its own end of turn; about 2.4 GB of weights on first use; elsewhere it falls back to
whisper) or whisper (Silero VAD + faster-whisper, CUDA when there is a GPU); --tts kokoro (needs espeak-ng;
CUDA through onnxruntime-gpu on Linux x86_64 when there is a GPU, see --tts-device) or say (macOS). Each GPU
backend loads only when the GPU keeps enough memory free after it (gpu.py); otherwise it runs on the CPU.

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
import json  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import secrets  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

from pathlib import Path  # noqa: E402

from sidecar import protocol  # noqa: E402
from sidecar.protocol import emit, log, warn  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["live", "front"], default="live")
    ap.add_argument("--stt", choices=["kyutai", "whisper"], default="kyutai", help="speech-to-text backend")
    ap.add_argument("--asr", default="base.en", help="faster-whisper size for --stt whisper")
    ap.add_argument("--tts", choices=["kokoro", "say"], default="kokoro")
    ap.add_argument("--voice", default="", help="Kokoro voice (af_heart) or say voice (Samantha)")
    ap.add_argument("--tts-device", choices=["auto", "cuda", "cpu"], default="auto", help="where Kokoro runs")
    ap.add_argument("--aec", choices=["on", "off"], default="on", help="echo cancelling on the mic")
    ap.add_argument("--end-silence-ms", type=int, default=1500)
    ap.add_argument("--mic", default="", help="input device: index or name substring (see --list-devices)")
    ap.add_argument("--speaker", default="", help="output device: index or name substring")
    ap.add_argument("--speaker-backend", choices=["auto", "local", "windows"], default="auto",
                    help="auto: under WSL, play on Windows (win_player.exe through interop) instead of WSLg")
    ap.add_argument("--front-backend", choices=["llamacpp", "anthropic"], default="llamacpp")
    ap.add_argument("--front-url", default="", help="OpenAI-compatible server; empty: run a managed llama-server")
    ap.add_argument("--front-server-bin", default="", help="managed front: an existing llama-server (no download)")
    ap.add_argument("--front-server-model", default="", help="managed front: an existing GGUF (no download)")
    ap.add_argument("--front-server-log", default="", help="managed front: its log; empty: <cache>/front-server.log")
    ap.add_argument("--front-model", default="", help="empty: the server's model, or claude-haiku-4-5 for anthropic")
    ap.add_argument("--switch-pattern", default="", help="regex whose group 1 is a model a spoken switch names")
    ap.add_argument("--log-file", default="", help="persistent sidecar log; empty: <cache dir>/sidecar.log")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--setup", action="store_true", help="check the system, fetch the models, test the voice")
    ap.add_argument("--unit", action="store_true", help="pure unit checks")
    ap.add_argument("--selftest", action="store_true", help="both modes without mic, speaker or network")
    ap.add_argument("--fake-audio", action="store_true", help=argparse.SUPPRESS)  # the self-test's lifecycle child
    return ap.parse_args(argv)


def run(args: argparse.Namespace, life: protocol.Lifecycle) -> int:
    from sidecar import audio, gpu, speech
    from sidecar.echo import EchoGuard, EchoReference

    emit(type="state", state="loading")
    t0 = time.monotonic()
    speech.check_tts(args.tts)
    sd = None if args.fake_audio else audio.load_sounddevice()
    mic = speaker = None
    if sd is not None:
        mic, speaker = audio.pick_device(sd, args.mic, "input"), audio.pick_device(sd, args.speaker, "output")
        audio.check_devices(sd, mic, speaker)
    launch = None
    if sd is not None:  # under WSL the speaker can be Windows' own: started now, it comes up while the models load
        from sidecar import winplayer

        backend, why = winplayer.resolve(args.speaker_backend)
        log(f"speaker backend: {backend} ({why})")
        if backend == "windows":
            launch = winplayer.Launch(args.speaker).start()
        elif args.speaker_backend == "windows":
            warn(f"Windows speaker unavailable ({why}); playing locally instead.")
    if not args.fake_audio:  # /livevibe without frontUrl: our own llama-server, admitted to the GPU before the recognizer
        from sidecar import front_server

        front_server.start_managed(args, life.on_quit)

    # The recognizer loads on the listener thread while this one loads the synthesizer.
    speaking = threading.Event()
    guard = EchoGuard(speaking.is_set)  # the barge-in gate holds through the tail of what was played
    if args.fake_audio:
        build = lambda: (speech.NoSpeech(), str)  # noqa: E731
    else:
        build = lambda: speech.recognizer(args.stt, args.asr, args.end_silence_ms, guard.active)  # noqa: E731
    listener = audio.Listener(build, life.quit, guard, aec=sd is not None and args.aec == "on")
    if not args.fake_audio:  # the recognizer has the first claim on the GPU; Kokoro takes what is left
        gpu.hold("stt", speech.stt_gpu_need(args.stt))
    listener.start()
    try:
        tts = speech.SilentTTS() if args.fake_audio else speech.make_tts(args.tts, args.voice, args.tts_device)
    except BaseException:
        listener.close()
        if launch is not None:
            launch.abandon()
        raise
    reference = EchoReference(tts.sample_rate) if sd is not None else None
    listener.reference = reference  # read by the mic callback, which opens later
    player = None
    local = lambda: audio.Player(sd, tts.sample_rate, speaker, reference, guard)  # noqa: E731
    if launch is not None:
        player = launch.player(tts.sample_rate, reference, guard)
        if player is not None:  # if it dies mid-session, the local speaker takes over
            player.fallback = local
    if player is None:
        try:
            player = local()
        except Exception as e:  # noqa: BLE001 - no speaker: keep listening, answer silently
            warn(f"speaker unavailable ({type(e).__name__}: {str(e)[:160]}); answers are not read aloud.")
            player = audio.Player(None, tts.sample_rate)
    if launch is not None:  # after a Windows player failure, one background retry; it takes over between sentences
        def retry():
            p = winplayer.Launch(args.speaker).start().player(tts.sample_rate, reference, guard, report=log)
            if p is not None:
                p.fallback = local
            return p

        player = winplayer.Speaker(player, retry)
    try:
        if not listener.wait_built():
            if listener.error is not None:
                e = listener.error
                warn(f"speech recognition cannot start ({type(e).__name__}: {str(e)[:200]}).")
                return 1
            return 0  # quit while loading
        voice = audio.Voice(tts, player, speaking, guard)
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
            from sidecar.front_server import authorize, when_ready

            authorize(brain)  # the managed server's key, if it runs one
            sess.spawn(when_ready(warm_up(brain)))  # off the startup path: a cold model load must not hold `ready`
        emit(type="ready", port=server.server_port, token=token)
        log(f"ready in {time.monotonic() - t0:.1f}s")
        sess.set_state("listening")
        await sess.run()
    finally:
        server.shutdown()
        server.server_close()
    log(f"stopped: {life.reason or 'session ended'}")
    return 0


def plugin_version() -> str:
    try:
        manifest = Path(__file__).resolve().parents[2] / ".claude-plugin" / "plugin.json"
        return str(json.loads(manifest.read_text(encoding="utf-8")).get("version", "unknown"))
    except Exception:  # noqa: BLE001 - the header is a convenience
        return "unknown"


def session_header(argv: list[str]) -> str:
    return (f"session start: live-vibe {plugin_version()} argv={argv!r} platform={platform.platform()} "
            f"python={platform.python_version()}")


def main() -> int:
    argv_in = sys.argv[1:]
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
    args.log_path = protocol.start_file_log(args.log_file, session_header(argv_in))
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
    finally:
        from sidecar import front_server

        front_server.stop_all()  # the managed front server, if any: os._exit below skips every other cleanup


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    protocol.stop_file_log()
    # Skip interpreter teardown: a daemon thread may still sit in MLX or PortAudio, and every stream is closed.
    os._exit(code)
