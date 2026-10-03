"""--unit: the pure pieces (protocol encoder, stdout guard, SentenceSplitter, TurnDetector, KyutaiTurns' turn logic,
the Kyutai backend choice and its Whisper fallback, SSE parsing, the spoken-switch pattern). --selftest: --unit,
then both sessions end to end against a fake front server with a silent player, the token guard, the process lifecycle in a child, and the real speech backends
when their weights are already downloaded. No mic, no speaker, no network after the first downloads."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import gpu, protocol
from .checks_front_server import front_server_units
from .checks_winplayer import winplayer_units
from .audio import FRAME, SR, CannotStart, Player, check_devices, SentenceSplitter, Tuning, TurnDetector, Voice, speakable
from .echo import EchoCanceller, EchoGuard, EchoReference
from .front import (EVENT, EVENT_CHARS, HISTORY_MAX, INTERRUPTED, RETELL, _FALLBACKS, _NO_EFFORT, Brain, Delegator,
                    FrontSession, LlamaCppBrain, SpeechFilter, TurnStream, event_message, is_turn_start, make_brain,
                    parse_sse, report_brief, spoken_model, warm_up)
from .session import LiveSession
from .speech import BARGE_IN_WORDS, KYUTAI_BLOCK, KYUTAI_SR, KyutaiTurns, SilentTTS, check_tts

MAIN = Path(__file__).with_name("main.py")
# The mod's SPOKEN_SWITCH source, as register.tsx passes it in --switch-pattern.
SWITCH = re.compile(r"^(?:(?:ok|okay|hey|alright) )?(?:claude )?(?:please )?(?:(?:switch|change|swap|set)(?: over)?(?: the)?(?: model)? to|use)(?: the)? (opus|sonnet|haiku|fable)(?: model)?(?: please)?$")


class Checker:
    def __init__(self) -> None:
        self.ok = True

    def __call__(self, cond: Any, what: str) -> bool:
        self.ok = self.ok and bool(cond)
        print(f"{'PASS' if cond else 'FAIL'}  {what}", flush=True)
        return bool(cond)


# =============================================================================================================
# unit checks
# =============================================================================================================
class ScriptedVAD:
    def __init__(self, probs: list[float]):
        self.probs = list(probs)

    def reset(self) -> None:
        pass

    def __call__(self, frame) -> float:
        return self.probs.pop(0) if self.probs else 0.0


def run_detector(probs: list[float], speaking: bool = False) -> list[tuple[int, str, Any]]:
    det = TurnDetector(ScriptedVAD(probs), Tuning(), lambda: speaking)
    return [(i, k, v) for i in range(len(probs)) for k, v in det.feed(np.full(FRAME, i, np.float32))]


class FakeKyutai:
    """KyutaiSTT's stepping interface over a script of (piece, p_end) per 80 ms step."""

    MAX_STEPS = 4096
    delay_steps = 6

    def __init__(self, script: list[tuple[str | None, float]]):
        self.script, self.steps = list(script), 0

    def reset(self) -> None:
        pass

    def step(self, block) -> tuple[str | None, float]:
        self.steps += 1
        return self.script.pop(0) if self.script else (None, 0.0)

    def transcribe(self, payload: Any) -> str:
        return payload if isinstance(payload, str) else ""


def run_kyutai(script: list[tuple[str | None, float]], speaking: bool = False) -> list[tuple[int, str, Any]]:
    turns = KyutaiTurns(FakeKyutai(script), Tuning(end_silence_ms=1500), lambda: speaking)
    block = KYUTAI_BLOCK * SR // KYUTAI_SR
    return [(i, k, v) for i in range(len(script) + 25) for k, v in turns.feed(np.zeros(block, np.float32))]


@contextmanager
def patched(*patches: tuple[Any, str, Any]) -> Iterator[None]:
    """Set (object, attribute, value) for the block, then put the originals back."""
    saved = [(o, a, getattr(o, a)) for o, a, _ in patches]
    try:
        for o, a, v in patches:
            setattr(o, a, v)
        yield
    finally:
        for o, a, v in reversed(saved):
            setattr(o, a, v)


def kyutai_choice(check: Checker) -> None:
    """--stt kyutai: MLX on Apple silicon, CUDA where torch sees a GPU, else Whisper with a
    log line; a backend that throws while loading is a warn and Whisper. No models load."""
    from . import kyutai_cuda as kc
    from . import speech

    def boom() -> None:
        raise RuntimeError("no kernel image is available")

    class FakeWhisper:
        def __init__(self, size: str) -> None:
            self.size = size

        def transcribe(self, audio: Any) -> str:
            return "whisper"

    def choose(apple: bool, cuda_why: str | None, mlx: Any = FakeKyutai, cuda: Any = FakeKyutai):
        out: list[dict[str, Any]] = []
        protocol.capture(out)
        try:
            with patched((speech, "is_apple_silicon", lambda: apple), (kc, "cuda_unavailable", lambda: cuda_why),
                         (speech, "KyutaiSTT", lambda: mlx([])), (kc, "KyutaiCudaSTT", lambda: cuda([])),
                         (speech, "WhisperASR", FakeWhisper), (speech, "load_vad", lambda: ScriptedVAD([]))):
                backend = kc.kyutai_backend()
                det, transcribe = speech.recognizer("kyutai", "base.en", 1500, lambda: False)
        finally:
            protocol.capture(None)
        return backend, det, transcribe, " | ".join(str(m.get("text")) for m in out), [m["type"] for m in out]

    class Mlx(FakeKyutai):
        pass

    backend, det, _, said, _ = choose(apple=True, cuda_why="never asked", mlx=Mlx)
    check(backend == ("mlx", "") and isinstance(det, KyutaiTurns) and isinstance(det.stt, Mlx),
          f"kyutai choice: Apple silicon runs MLX {backend}")
    backend, det, transcribe, said, _ = choose(apple=False, cuda_why=None)
    check(backend == ("cuda", "") and isinstance(det, KyutaiTurns) and transcribe("hi") == "hi",
          f"kyutai choice: Linux with a CUDA GPU runs PyTorch {backend}")
    backend, det, transcribe, said, kinds = choose(apple=False, cuda_why="PyTorch sees no CUDA GPU")
    check(backend == (None, "PyTorch sees no CUDA GPU") and isinstance(det, TurnDetector)
          and transcribe(None) == "whisper" and kinds == ["log"] and "no CUDA GPU" in said and "using Whisper" in said,
          f"kyutai choice: no GPU falls back to Whisper and Silero with a log line {said!r}")
    backend, det, transcribe, said, kinds = choose(apple=False, cuda_why=None, cuda=lambda _: boom())
    check(isinstance(det, TurnDetector) and kinds == ["warn"] and "CUDA unavailable" in said and "no kernel image" in said,
          f"kyutai choice: a CUDA load failure warns and falls back to Whisper {said!r}")
    backend, det, _, said, kinds = choose(apple=True, cuda_why=None, mlx=lambda _: boom())
    check(isinstance(det, TurnDetector) and kinds == ["warn"] and "MLX unavailable" in said,
          f"kyutai choice: an MLX load failure warns and falls back to Whisper {said!r}")
    with patched((speech, "WhisperASR", FakeWhisper), (speech, "load_vad", lambda: ScriptedVAD([])),
                 (kc, "load_kyutai", boom)):
        det, _ = speech.recognizer("whisper", "base.en", 1500, lambda: False)
    check(isinstance(det, TurnDetector), "kyutai choice: --stt whisper never looks for Kyutai")

    if importlib.util.find_spec("torch") is not None:
        import torch

        free = int((kc.MIN_FREE_GIB - 1) * 2**30)
        with patched((torch.cuda, "is_available", lambda: True), (gpu, "read_free_gib", lambda: None),
                     (torch.cuda, "mem_get_info", lambda *a: (free, 32 * 2**30))):
            why = kc.cuda_unavailable()
        check(why is not None and "GiB of GPU memory free" in why, f"kyutai choice: a busy GPU is no GPU {why!r}")
        asked: list[int] = []
        with patched((torch.cuda, "is_available", lambda: True), (gpu, "read_free_gib", lambda: 2.5),
                     (torch.cuda, "mem_get_info", lambda *a: asked.append(1) or (40 * 2**30, 0))):
            why = kc.cuda_unavailable()
        check(why is not None and "2.5 GiB" in why and not asked,
              f"kyutai choice: nvidia-smi's reading decides, without a CUDA context {why!r}")
        gpu.release("stt")
        with patched((torch.cuda, "is_available", lambda: False)):
            why = kc.cuda_unavailable()
        check(why == "PyTorch sees no CUDA GPU", f"kyutai choice: torch without CUDA {why!r}")
    else:
        print("SKIP  kyutai choice: torch not installed here (not Linux x86_64), CUDA probe not exercised", flush=True)


def gpu_budget(check: Checker) -> None:
    """The shared VRAM budget and each backend's fallback to the CPU on a full GPU (mocked readings; nothing
    touches a GPU)."""
    from . import speech

    def admit(free: float | None, name: str, need: float, stt_held: float = 0.0) -> str | None:
        with patched((gpu, "read_free_gib", lambda: free)):
            if stt_held:
                gpu.hold("stt", stt_held)
            try:
                return gpu.admit(name, need)
            finally:
                gpu.release("stt")
                gpu.release(name)

    full = admit(2.5, "tts", gpu.NEED_GIB["kokoro"])
    roomy = admit(20.0, "tts", gpu.NEED_GIB["kokoro"], stt_held=gpu.NEED_GIB["kyutai"])
    shared = admit(7.0, "tts", gpu.NEED_GIB["kokoro"], stt_held=gpu.NEED_GIB["kyutai"])
    unknown = admit(None, "tts", gpu.NEED_GIB["kokoro"])
    check(full is not None and "only 2.5 GiB" in full and roomy is None and shared is not None
          and "claimed by the other" in shared and unknown is not None and not gpu.claimed(),
          f"gpu budget: room after the need and the others' claims, else why {[full, shared, unknown]}")
    gpu.hold("stt", 3.2)
    with patched((gpu, "read_free_gib", lambda: 20.0)):
        took = gpu.admit("tts", 1.4) is None and gpu.claimed(exclude="stt") == 1.4
    gpu.release("stt")
    gpu.release("tts")
    check(took and not gpu.claimed(), "gpu budget: an admitted backend holds its share until released")

    import onnxruntime as ort

    class FakeSession:
        def __init__(self, model: str, so: Any = None, providers: list[Any] | None = None) -> None:
            self.providers = [p[0] if isinstance(p, tuple) else p for p in providers or []]

        def get_providers(self) -> list[str]:
            return self.providers

    def kokoro(free: float) -> tuple[list[str], str]:
        with patched((ort, "InferenceSession", FakeSession), (gpu, "read_free_gib", lambda: free),
                     (ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]),
                     *([(ort, "preload_dlls", lambda *a, **k: None)] if hasattr(ort, "preload_dlls") else [])):
            sess, why = speech.kokoro_session("model.onnx", "auto")
        gpu.release("tts")
        return sess.get_providers(), why

    on_full, why_full = kokoro(2.5)
    on_roomy, why_roomy = kokoro(20.0)
    check(on_full == ["CPUExecutionProvider"] and "2.5 GiB" in why_full and on_roomy[0] == "CUDAExecutionProvider"
          and not why_roomy, f"kokoro: CPU on a full GPU, CUDA with room {on_full} {why_full!r}")

    class FakeWhisperModel:
        def __init__(self, size: str, device: str, compute_type: str) -> None:
            self.device = device

        def transcribe(self, *a: Any, **k: Any) -> tuple[list[Any], None]:
            return [], None

    fake_fw = type(sys)("faster_whisper")
    fake_fw.WhisperModel = FakeWhisperModel  # type: ignore[attr-defined]
    out: list[dict[str, Any]] = []
    protocol.capture(out)
    try:
        with patched((speech, "_cuda_devices", lambda: 1), (gpu, "read_free_gib", lambda: 2.5)):
            saved = sys.modules.get("faster_whisper")
            sys.modules["faster_whisper"] = fake_fw
            try:
                w = speech.WhisperASR("base.en")
            finally:
                if saved is not None:
                    sys.modules["faster_whisper"] = saved
                else:
                    del sys.modules["faster_whisper"]
    finally:
        protocol.capture(None)
        gpu.release("stt")
    said = [o["text"] for o in out]
    check(w.device == "cpu" and len(said) == 1 and "CPU int8" in said[0] and "2.5 GiB" in said[0],
          f"whisper: CPU on a full GPU, one log line {said}")


def tool_pairs_ok(history: list[dict[str, Any]]) -> bool:
    """Every OpenAI tool result answers a call made earlier in the history, and every call is answered."""
    asked: set[str] = set()
    answered: set[str] = set()
    for m in history:
        asked |= {c["id"] for c in m.get("tool_calls") or []}
        if m["role"] == "tool":
            if m["tool_call_id"] not in asked:
                return False
            answered.add(m["tool_call_id"])
    return asked == answered


def front_history(check: Checker) -> None:
    """The front's history shape: delegations as structured tool calls (never bracketed text among the words),
    a cut turn flagged on the next user message, trimming by whole turns, and the speech filter."""
    b = Brain()
    b.begin("fix the parser")
    call = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "delegate", "arguments": '{"request": "Fix it"}'}}]}
    b.record(call, {"role": "tool", "tool_call_id": "c1", "content": "Handed off"})
    said = b.commit("On it.", False)
    roles = [m["role"] for m in b.history]
    words = [m["content"] for m in b.history if m["role"] == "assistant" and m["content"]]
    check(said == "On it." and roles == ["user", "assistant", "tool", "assistant"] and words == ["On it."]
          and not any("[" in w for w in words) and tool_pairs_ok(b.history),
          f"front history: a delegation is a tool call and its result, the words stay clean {roles}")
    b.begin("tell me more")
    said = b.commit("Well, the first...", True)
    b.begin("stop")
    check(b.history[-1]["content"] == f"{INTERRUPTED} stop" and not any("cut off" in str(m["content"]) for m in b.history),
          "front history: a cut turn is flagged on the next user message, not among the assistant's words")
    b.commit("", False)
    check(b.history[-1]["role"] == "user", "front history: nothing heard, no assistant message")

    b = Brain()
    for i in range(HISTORY_MAX):
        b.begin(f"u{i}")
        b.record({"role": "assistant", "content": None, "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": "delegate", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": f"c{i}", "content": "ok"})
        b.commit(f"a{i}", False)
        b.begin(f"chat{i}")
        b.commit(f"sure{i}", False)
    b.begin("last")
    check(len(b.history) <= HISTORY_MAX and is_turn_start(b.history[0]) and tool_pairs_ok(b.history),
          f"front history: trimming drops whole turns, no orphan tool call or result ({len(b.history)} messages)")
    a = Brain()
    a.history = [{"role": "user", "content": "x"}, {"role": "assistant", "content": [{"type": "tool_use", "id": "t"}]},
                 {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t"}]}]
    check(is_turn_start(a.history[0]) and not is_turn_start(a.history[2]),
          "front history: an Anthropic tool_result message is not a turn start")

    def filtered(deltas: list[str]) -> tuple[str, list[str], list[str]]:
        f = SpeechFilter()
        out = "".join(f.push(d) for d in deltas) + f.flush()
        return out, f.notes, f.delegations()

    got = filtered(["<th", "ink>\n\n</th", "ink>\n\nI'm doing well."])
    check(got[0].strip() == "I'm doing well." and not got[1], f"speech filter: a think block split across deltas {got}")
    got = filtered(["<think>plan", " this</think>Fine. 2 < 3, and a <b>."])
    check(got[0] == "Fine. 2 < 3, and a <b>.", f"speech filter: thinking dropped, other angle brackets kept {got}")
    got = filtered(["On it. [deleg", "ated: Open the parser file and fix the te", "st] Back soon."])
    check(got[0] == "On it.  Back soon." and got[2] == ["Open the parser file and fix the test"],
          f"speech filter: a delegation note is not spoken and becomes a request {got}")
    got = filtered(["Done. [cut off by the user]", " [delegated: half a note"])
    check(got[0].strip() == "Done." and got[2] == ["half a note"], f"speech filter: notes, even unclosed, are silent {got}")
    got = filtered(["[", "x" * 500, " tail"])
    check(got[0].startswith("[x") and got[0].endswith("tail"), "speech filter: a lone '[' in long text is spoken")

    raw = json.dumps({"say": 'On it, "quick" fix: café \\ done.', "delegate": "Fix parser.py"})
    for size in (1, 3, 7, len(raw)):
        t = TurnStream()
        out = "".join(t.push(raw[i:i + size]) for i in range(0, len(raw), size))
        if out != 'On it, "quick" fix: café \\ done.' or t.fields.get("delegate") != "Fix parser.py":
            break
    check(out == 'On it, "quick" fix: café \\ done.' and t.fields == {"say": out, "delegate": "Fix parser.py"},
          f"json turn: say streams as decoded text in any chunking, delegate is read whole {out!r} {t.fields}")
    t = TurnStream()
    out = t.push(json.dumps({"say": "you're welcome'}{", "delegate": ""}))
    check(out == "you're welcome'", f"json turn: braces echoed inside say are not spoken {out!r}")
    t = TurnStream()
    first = t.push('{"say": "Hel')
    check(first == "Hel" and "say" not in t.fields, "json turn: speech starts before the field closes")
    t = TurnStream()
    out = t.push("  ") + t.push("Plain reply.") + t.push(" More.")
    check(out == "Plain reply. More." and t.plain and not t.fields, "json turn: a reply not starting with { is plain")
    b = LlamaCppBrain("sys", "http://127.0.0.1:9", "")
    b.begin("fix it")
    b.current.request = "Fix it now"
    b.commit("", True)
    b.begin("hi")
    b.commit("Hello.", False)
    check([json.loads(m["content"]) for m in b.history if m["role"] == "assistant"]
          == [{"say": "", "delegate": "Fix it now"}, {"say": "Hello.", "delegate": ""}]
          and b.history[2]["content"] == f"{INTERRUPTED} hi" and b.system.endswith(LlamaCppBrain.PROTOCOL),
          "json turn: history keeps each turn as JSON, a delegation even when nothing was heard")
    b = LlamaCppBrain("sys", "http://127.0.0.1:9", "")
    b.begin("fix it")
    late = b.current
    b.commit("On it, I'll...", True)  # cut before the delegate field arrived
    b.begin("wait")
    quiet = b.current
    b.commit("", True)  # cut before anything was heard
    b.begin("next")
    late.request, quiet.request = "Fix it", "Wait for it"
    b._amend(late)
    b._amend(quiet)
    contents = [m["content"] for m in b.history]
    check(json.loads(contents[1]) == {"say": "On it, I'll...", "delegate": "Fix it"}
          and json.loads(contents[3]) == {"say": "", "delegate": "Wait for it"}
          and contents[4] == f"{INTERRUPTED} next",
          f"json turn: a delegation landing after its turn was cut joins that turn's place in history {contents}")
    report_units(check)


def report_units(check: Checker) -> None:
    """What of a Claude result the front retells, and how it is framed."""
    short = "Fixed parser.py; all 41 tests pass.\n\nNot pushed. Want me to push it?"
    check(report_brief(f"  {short}\n") == short, "report brief: two paragraphs or fewer stay whole")
    long_ = ("## Summary\n\nI've started a worker on the Windows player.\n\nHow it would work:\n- streams PCM to "
             "Windows\n- which should eliminate the static\n\n## Next\n\n"
             "Nothing is pushed. I'll report when it's done.")
    got = report_brief(long_)
    check(got == "I've started a worker on the Windows player.\n\nNothing is pushed. I'll report when it's done.",
          f"report brief: the lead and the closing paragraph, not the design between them or the headings {got!r}")
    got = report_brief("All 12 pass.\n\nDetails:\n\n- a.py\n- b.py\n\n| file | lines |\n|---|---|\n| a | 3 |")
    check(got == "All 12 pass.\n\nDetails:", f"report brief: a trailing list or table is not the tail {got!r}")
    check(report_brief("Lead.\n\n1. one\n\n2. two") == "Lead.",
          "report brief: only the lead when nothing after it is prose")
    check(len(report_brief("x" * (EVENT_CHARS + 50))) == EVENT_CHARS, "report brief: cut at EVENT_CHARS")
    msg = event_message('Done. Want me to run "make test" now?')
    check(msg == f"{EVENT} \"Done. Want me to run 'make test' now?\"\n{RETELL}" and "do not answer" in RETELL,
          f"event message: the report quoted, then the retell reminder, so its question is not the last word {msg!r}")


def units() -> int:
    check = Checker()

    line = protocol.encode({"type": "log", "text": 'two\nlines "quoted" é'})
    check(line.endswith("\n") and line.count("\n") == 1 and json.loads(line)["text"] == 'two\nlines "quoted" é',
          "encoder: one line per message, newlines escaped, round-trips")

    guard = ("import os, sys; sys.path.insert(0, sys.argv[1]); from sidecar import protocol; protocol.claim_stdout(); "
             "print('python noise'); os.write(1, b'c noise\\n'); protocol.emit(type='log', text='x')")
    r = subprocess.run([sys.executable, "-c", guard, str(MAIN.parent.parent)], capture_output=True, text=True, timeout=30)
    check(r.stdout == '{"type":"log","text":"x"}\n' and "python noise" in r.stderr and "c noise" in r.stderr,
          f"stdout guard: library prints go to stderr (stdout {r.stdout!r})")

    sp = SentenceSplitter()
    got = sp.push("One two. Three four! Five") + sp.push(" six? Seven") + sp.flush()
    check(got == ["One two.", "Three four!", "Five six?", "Seven"], f"splitter: streamed sentences {got}")
    sp = SentenceSplitter(max_chars=60)
    got = sp.push("a run-on clause that keeps going and going, then another clause that keeps going too")
    check(len(got) == 1 and got[0].endswith("going,") and len(sp.buf) > 0, f"splitter: long run cut at a comma {got}")
    got = SentenceSplitter().push("**Bold** `code` here. ")
    check(got == ["Bold code here."], f"splitter: inline markdown stripped {got}")
    s = speakable("See ```py\nx = 1\n``` and https://example.com/x now.")
    check("code omitted" in s and "a link" in s and "x = 1" not in s, f"speakable: fences and URLs {s!r}")

    ev = run_detector([0.0] * 20 + [0.9] * 13 + [0.0] * 30)
    start = [i for i, k, _ in ev if k == "speech_start"]
    utt = [v for _, k, v in ev if k == "utterance"]
    check(start == [22] and len(utt) == 1 and len(utt[0]) == (10 + 10 + 21) * FRAME,
          f"turn detector: starts after 3 frames, pre-roll 320 ms, ends after 700 ms of silence {start}")
    ev = run_detector([0.6] * 30 + [0.9] * 8 + [0.0] * 30, speaking=True)
    check([i for i, k, _ in ev if k == "speech_start"] == [37], "turn detector: while speaking, 0.6 never barges in; 8 frames at 0.9 do")
    ev = run_detector([0.9] * 3 + [0.0] * 30)
    check([k for _, k, _ in ev] == ["speech_start", "discard"], "turn detector: under 250 ms of speech is discarded")

    words = [("▁open", 0.0), ("▁the", 0.0), ("▁file", 0.0)]
    ev = run_kyutai([(None, 0.0)] * 3 + words + [(None, 0.9)] * 8)
    check([k for _, k, _ in ev] == ["speech_start", "utterance"] and ev[0][0] == 3 and ev[1][2] == "open the file"
          and ev[1][0] == 3 + 3 + 5, f"kyutai turns: first word starts, end-of-turn head held 6 steps ends {ev}")
    ev = run_kyutai([("▁mm", 0.0), ("▁hm", 0.0), (None, 0.0), ("▁wait", 0.0)] + [(None, 0.9)] * 8, speaking=True)
    check([k for _, k, _ in ev][:1] == ["speech_start"] and ev[0][0] == 3,
          f"kyutai turns: {BARGE_IN_WORDS} words to barge in while speaking {ev[:1]}")
    ev = run_kyutai([("▁so", 0.0)] + [(None, 0.1)] * 30)
    check([k for _, k, _ in ev] == ["speech_start", "utterance"] and ev[1][0] == 1 + 18,
          "kyutai turns: 1500 ms without a word caps the turn")

    check(spoken_model(SWITCH, "Okay, switch to Sonnet.") == "sonnet" and spoken_model(SWITCH, "Use opus to review this.") is None,
          "spoken switch: the mod's pattern over the mod's normalization")
    check(parse_sse('data: {"choices":[{"delta":{"content":"hi"}}]}') == ("delta", {"content": "hi"})
          and parse_sse("data: [DONE]") == ("done", None) and parse_sse("data: {oops") == ("bad", None)
          and parse_sse('data: {"choices":"x"}')[0] == "bad" and parse_sse(": keep-alive") == ("skip", None)
          and parse_sse('data: {"choices":[]}') == ("delta", {}), "sse: deltas, done, malformed and comment lines")
    check(_NO_EFFORT.search("claude-haiku-4-5") and not _NO_EFFORT.search("claude-opus-5-5")
          and _FALLBACKS.match("claude-opus-5-5") and not _FALLBACKS.match("claude-haiku-4-5"),
          "anthropic front: effort and fallbacks only where the model takes them")

    out: list[dict[str, Any]] = []
    protocol.capture(out)
    try:
        d = Delegator()
        bad = asyncio.run(d.call("delegate", None))
        good = asyncio.run(d.call("delegate", {"request": "Fix it"}))
    finally:
        protocol.capture(None)
    check(bad.startswith("error") and out == [{"type": "delegate", "text": "Fix it"}] and good.startswith("Handed off"),
          "delegate: bad JSON arguments answer the model, a good call reaches Claude")
    out.clear()
    protocol.capture(out)
    try:
        d = Delegator()
        first = asyncio.run(d.call("delegate", {"request": "Play the test sentences and report back."}))
        again = asyncio.run(d.call("delegate", {"request": "play the test sentences, and report back"}))
        other = asyncio.run(d.call("delegate", {"request": "Run the unit tests."}))
        d.announcing = True
        on_result = asyncio.run(d.call("delegate", {"request": "Fix the build."}))
        d.announcing = False
        d.sent = [(t - d.REPEAT_S - 1, r) for t, r in d.sent]
        later = asyncio.run(d.call("delegate", {"request": "Play the test sentences and report back."}))
    finally:
        protocol.capture(None)
    sent = [o["text"] for o in out if o["type"] == "delegate"]
    check(first.startswith("Handed off") and again.startswith("Not handed off") and other.startswith("Handed off")
          and on_result.startswith("Not handed off") and later.startswith("Handed off")
          and sent == ["Play the test sentences and report back.", "Run the unit tests.",
                       "Play the test sentences and report back."],
          f"delegate: no repeat within {Delegator.REPEAT_S:.0f} s, nothing while a result is announced {sent}")
    b = Brain()
    for i in range(HISTORY_MAX):
        b.begin(f"u{i}")
        b.commit(f"a{i}", False)
    b.begin("last")
    check(len(b.history) <= HISTORY_MAX and b.history[0]["role"] == "user" and b.history[-1]["content"] == "last",
          f"front history capped at {HISTORY_MAX}, starting on a user turn")
    front_history(check)

    class NoDevices:
        def query_devices(self, device=None, kind=None):
            raise ValueError("No input device matching -1")

    def fatal(fn: Callable[[], None]) -> str:
        try:
            fn()
        except CannotStart as e:
            return str(e)
        return ""

    msg = fatal(lambda: check_devices(NoDevices(), None, None))
    check("no input device" in msg and "microphone" in msg, f"no mic: one fatal message {msg!r}")
    real, sys.platform = sys.platform, "linux"
    try:
        msg = fatal(lambda: check_tts("say"))
    finally:
        sys.platform = real
    check("macOS only" in msg and "apt install espeak-ng" in msg, f"say off macOS: one fatal message naming the fix {msg!r}")
    kyutai_choice(check)
    gpu_budget(check)
    file_log_units(check)

    front_server_units(check)
    echo_units(check)
    winplayer_units(check)
    print("UNIT: ALL PASS" if check.ok else "UNIT: SOME CHECKS FAILED", flush=True)
    return 0 if check.ok else 1


def file_log_units(check: Checker) -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "sub" / "sidecar.log"
        code = ("import os, sys; sys.path.insert(0, sys.argv[1]); from sidecar import protocol; protocol.claim_stdout(); "
                "protocol.start_file_log(sys.argv[2], 'session start: test'); protocol.log('hello'); protocol.warn('two\\nlines'); "
                "print('lib noise'); os.write(2, b'c noise\\n'); protocol.stop_file_log()")
        r = subprocess.run([sys.executable, "-c", code, str(MAIN.parent.parent), str(path)], capture_output=True, text=True, timeout=30)
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        rec = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}[+-]\d\d:\d\d (\w+) pid=(\d+) (.*)$")
        rows = [m.groups() for m in map(rec.match, text.splitlines()) if m]
        check(len(rows) == len(text.splitlines()) == 6 and len({pid for _, pid, _ in rows}) == 1,
              f"file log: every line is ISO time, level, pid, text ({text!r})")
        check([(lv, t.split(" argv")[0]) for lv, _, t in rows][:3] == [("INFO", "session start: test"), ("INFO", "hello"), ("WARNING", "two")]
              and ("STDERR", "lib noise") in [(lv, t) for lv, _, t in rows] and ("STDERR", "c noise") in [(lv, t) for lv, _, t in rows],
              "file log: header, log, warn (split per line) and library stderr all land in the file")
        check('"type":"log","text":"hello"' in r.stdout and "lib noise" in r.stderr and "c noise" in r.stderr,
              f"file log: the mod channel is unchanged (stdout {r.stdout!r}, stderr {r.stderr!r})")

        small = Path(d) / "rot" / "s.log"
        protocol.stop_file_log()
        h = protocol.start_file_log(str(small), "h")
        try:
            for _ in range(2):
                protocol.file_log("INFO", "x" * 1000 * 1000)
                protocol.file_log("INFO", "y" * 1000 * 1000)
                protocol.file_log("INFO", "z" * 1000 * 1000)
            check(h == small and small.with_name("s.log.1").exists() and small.stat().st_size <= protocol.LOG_MAX_BYTES,
                  "file log: rotates at the size cap, keeping one previous file")
        finally:
            protocol.stop_file_log()
            protocol._file_logger = None

        blocked = Path(d) / "afile"
        blocked.write_text("x")
        warned: list[dict[str, Any]] = []
        protocol.capture(warned)
        try:
            got = protocol.start_file_log(str(blocked / "x.log"), "h")
        finally:
            protocol.capture(None)
            protocol.stop_file_log()
        check(got is None and len(warned) == 1 and warned[0]["type"] == "warn" and protocol._file_logger is None,
              f"file log: an unwritable path warns once and the sidecar goes on {warned}")
        protocol.file_log("INFO", "after failure")  # must not raise


class _Status:
    def __init__(self, underflow: bool = False):
        self.output_underflow = underflow


class _Times:
    def __init__(self, dac: float = 0.0, adc: float = 0.0, now: float = 0.0):
        self.outputBufferDacTime, self.inputBufferAdcTime, self.currentTime = dac, adc, now


def echo_units(check: Checker) -> None:
    """The echo guard's transcript filter and tail, the reference's time alignment, the player's callback, and
    (when livekit loads) the canceller on a synthetic echo."""
    g = EchoGuard(lambda: False)
    g.spoke("I handed the parser fix to the coding agent, and it is running the tests now.")
    got = (g.strip("handed the parser fix to the coding agent", True),
           g.strip("Wait, stop. What about the database migration?", True),
           g.strip("I handed the parser fix. Wait, what about the migration?", True),
           g.strip("handed the parser fix to the coding agent", False),
           g.strip("running", True), g.strip("tests", True))
    check(got == ("", "Wait, stop. What about the database migration?", "Wait, what about the migration?",
                  "handed the parser fix to the coding agent", "", ""),
          f"echo guard: drops echo, keeps the user's words around it, only while overlapping playback {got}")
    g.sound_until = time.monotonic() - 10
    idle = not g.active()
    g.sounding(time.monotonic() + 0.1)
    check(idle and g.active() and EchoGuard(lambda: True).active(), "echo guard: active while speaking and in the tail")

    ref = EchoReference(24_000)
    t0, block = 100.0, 900
    sig = np.sin(2 * np.pi * 220 * np.arange(24_000) / 24_000).astype(np.float32)
    for i in range(0, len(sig), block):
        ref.played(sig[i:i + block], t0 + i / 24_000)
    got_ref = np.concatenate([ref.take(512, t0 + 0.2 + k * 512 / SR, SR) for k in range(10)])
    want = np.sin(2 * np.pi * 220 * (0.2 + np.arange(len(got_ref)) / SR)).astype(np.float32)
    again = ref.take(512, t0 + 0.2, SR)
    err = float(np.abs(got_ref - want).max())
    check(err < 0.02 and not again.any(), f"echo reference: the mic reads what played when it captured, once ({err:.4f})")

    g2 = EchoGuard(lambda: False)
    ref2 = EchoReference(24_000)
    p = Player(None, 24_000, reference=ref2, guard=g2)
    p._clip, p._pos = np.full(1000, 0.5, np.float32), 0
    out = np.ones((600, 1), np.float32)
    p._callback(out, 600, _Times(dac=50.0), _Status())
    first = bool((out[:, 0] == 0.5).all())
    p._callback(out, 600, _Times(dac=50.025), _Status(underflow=True))
    check(first and (out[:400, 0] == 0.5).all() and not out[400:, 0].any() and p._done.is_set()
          and p.underflows == 1 and g2.active(), "player callback: plays the clip, pads silence, marks the tail")
    check(np.abs(ref2.take(320, 50.0 + 0.005, SR) - 0.5).max() < 1e-3, "player callback: feeds the echo reference")

    try:
        canceller = EchoCanceller()
    except Exception as e:  # noqa: BLE001
        print(f"SKIP  echo canceller: livekit unavailable ({type(e).__name__}: {e})", flush=True)
        return
    rng = np.random.default_rng(0)
    far = np.repeat(rng.standard_normal(SR * 6 // 80), 80).astype(np.float32) * 0.2  # blocky, speech-like band
    far *= (np.sin(np.arange(len(far)) / SR * 2 * np.pi * 1.5) > -0.3)  # gaps, like words
    d = int(0.2 * SR)
    echo = np.zeros_like(far)
    echo[d:] = 0.4 * far[:-d]
    out = np.concatenate([canceller.process(echo[i:i + 512], far[i:i + 512]) for i in range(0, len(far), 512)])
    tail = slice(SR * 3, len(out))
    erle = 10 * np.log10(np.mean(echo[tail] ** 2) / (np.mean(out[tail] ** 2) + 1e-12))
    check(erle > 10, f"echo canceller: AEC3 removes a 200 ms echo once converged (ERLE {erle:.1f} dB)")


# =============================================================================================================
# self-test
# =============================================================================================================
def fake_front() -> tuple[str, list[dict[str, Any]]]:
    """An OpenAI-compatible SSE server answering in the front's JSON turn, in small pieces as a model streams, with a
    reply that depends on the last message: delegates 'fix', relays [task finished], fails on 'boom' (500), sends a
    malformed line on 'garbled', a broken JSON turn on 'badargs', plain text on 'plaintext'."""
    seen: list[dict[str, Any]] = []

    def turn(say: str, delegate: str = "", lead: str = "", cut: int = 0) -> list[dict[str, Any]]:
        text = lead + json.dumps({"say": say, "delegate": delegate})
        text = text[:-cut] if cut else text
        return [{"content": text[i:i + 7]} for i in range(0, len(text), 7)]

    class H(BaseHTTPRequestHandler):
        def log_message(self, *_: Any) -> None:
            pass

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            last = body["messages"][-1]
            text = str(last.get("content") or "")
            if "boom" in text and last["role"] == "user":
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"model crashed")
                return
            self.send_response(200)
            if not body.get("stream"):
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"choices":[{"message":{"role":"assistant","content":"Hi"}}]}')
                return
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            raw: list[str] = []
            delay = 0.0
            if text.startswith(EVENT) and "next steps" in text:  # a small model re-delegating a result's to-dos
                deltas = turn("I'll get the test sentences playing.", "Play the test sentences and report back")
            elif text.startswith(EVENT):
                deltas = turn("All the tests pass now.")
            elif "goodbye" in text.lower():
                deltas = turn("Goodbye.")
            elif "thinking leak" in text:
                deltas = turn('Hello "there", été.', lead="<think>\n\n</think>\n\n")  # \\u escapes, quotes
            elif "notecall" in text:
                deltas = turn("Sure. [delegated: Rebuild the index]")
            elif "garbled" in text:
                raw, deltas = ["data: {not json"], turn("Still here.")
            elif "badargs" in text:
                deltas = turn("I'll let you know.", "Never sent", cut=8)
            elif "plaintext" in text:
                deltas = [{"content": "Plain words. "}, {"content": "No JSON here."}]
            elif "slow work" in text:  # a long "on it", then the delegate field: cut while it is spoken
                deltas = turn("".join(f"Working on part {i} of it now. " for i in range(8)), "Rebuild the cache")
                delay = 0.03
            elif "fix" in text:
                deltas = turn("On it.", "Fix the failing test in parser.py")
            else:
                deltas = turn("".join(f"This is sentence number {i} of a long answer. " for i in range(12)))
            for r in raw:
                self.wfile.write(f"{r}\n\n".encode())
            for d in deltas:
                self.wfile.write(f"data: {json.dumps({'choices': [{'delta': d}]})}\n\n".encode())
                time.sleep(delay)
            self.wfile.write(b"data: [DONE]\n\n")

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_port}", seen


def post(port: int, path: str, body: str = "", token: str | None = None) -> int:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=body.encode(), method="POST")
    if token is not None:
        req.add_header(protocol.TOKEN_HEADER, token)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, ConnectionError):  # reset: the server refused the body, or exited first
        return -1


class TimedTTS(SilentTTS):
    """Silent speech, 0.05 s a word, recording when each synthesis starts."""

    def __init__(self) -> None:
        self.started: list[tuple[float, str]] = []

    def synth(self, text: str) -> np.ndarray:
        self.started.append((time.monotonic(), text))
        return super().synth(text)


class TimedPlayer(Player):
    def __init__(self) -> None:
        super().__init__(None, SilentTTS.sample_rate)
        self.ended: list[float] = []

    def play(self, audio, cancel) -> int:
        n = super().play(audio, cancel)
        self.ended.append(time.monotonic())
        return n


async def until(cond: Callable[[], Any], s: float = 10) -> bool:
    end = time.monotonic() + s
    while time.monotonic() < end:
        if cond():
            return True
        await asyncio.sleep(0.02)
    return bool(cond())


async def selftest_sessions(check: Checker) -> None:
    out: list[dict[str, Any]] = []
    protocol.capture(out)

    def emitted(kind: str) -> list[dict[str, Any]]:
        return [o for o in out if o["type"] == kind]

    url, seen = fake_front()
    brain = make_brain("llamacpp", url, "")
    assert isinstance(brain, LlamaCppBrain)
    check(await warm_up(brain), "front: warm-up reaches the server")
    w = seen[-1]
    schema = w["response_format"]["json_schema"]["schema"]
    check("tools" not in w and schema["required"] == ["say", "delegate"]
          and w["chat_template_kwargs"] == {"enable_thinking": False} and w["max_tokens"] == 1 and "one assistant" in w["messages"][0]["content"]
          and '"delegate"' in w["messages"][0]["content"],
          "front: warm-up sends the front prompt with the JSON turn's schema, no tools, thinking off")

    def said(k: int = -1) -> str:
        """What the front said in history message k: its JSON turn's "say", or plain text."""
        content = brain.history[k]["content"] if len(brain.history) >= -k else ""
        try:
            turn = json.loads(content)
        except (TypeError, ValueError):
            return str(content)
        return str(turn.get("say", "")) if isinstance(turn, dict) else str(content)

    def delegated(k: int = -1) -> str:
        try:
            return str(json.loads(brain.history[k]["content"]).get("delegate", ""))
        except (TypeError, ValueError, AttributeError):
            return ""

    tts, player = TimedTTS(), TimedPlayer()
    voice = Voice(tts, player, threading.Event())
    hearing = {"on": False}
    said_goodbye = threading.Event()
    sess = FrontSession(voice, lambda: hearing["on"], brain, SWITCH, said_goodbye.set)
    quit_seen = threading.Event()

    def quit_(_: str) -> bool:
        quit_seen.set()
        sess.stop()
        return True

    token = "t0ken"
    server = protocol.serve({**sess.routes(), "/quit": quit_}, token)
    port = server.server_port
    running = asyncio.create_task(sess.run())

    def utter(text: str) -> None:
        for kind, payload in (("speech_start", 1.0), ("transcribing", None), ("utterance", text)):
            sess.post(kind, payload)

    async def turn_done() -> bool:
        await asyncio.sleep(0.05)
        return await until(lambda: not sess.turn_running())

    utter("Please fix the failing test.")
    await until(lambda: emitted("delegate"))
    await turn_done()
    check(emitted("delegate") == [{"type": "delegate", "text": "Fix the failing test in parser.py"}], f"front: utterance -> delegate {emitted('delegate')}")
    check([m["role"] for m in brain.history] == ["user", "assistant"] and said() == "On it."
          and delegated() == "Fix the failing test in parser.py" and "[" not in said(),
          f"front: history keeps the turn as the model answers it, words and delegation apart {brain.history}")
    check([o["role"] for o in emitted("transcript")] == ["user", "front"], "front: transcript, user then front")

    n = len(brain.history)
    utter("Okay, switch to Sonnet.")
    await until(lambda: emitted("switch_model"))
    check(emitted("switch_model") == [{"type": "switch_model", "model": "sonnet"}] and len(brain.history) == n,
          "front: a spoken model switch goes to the mod, not the front")

    r = await asyncio.to_thread(post, port, "/event", "inject", None)
    r2 = await asyncio.to_thread(post, port, "/event", "inject", "wrong")
    r3 = await asyncio.to_thread(post, port, "/event", "x" * (protocol.MAX_BODY + 1), token)
    r4 = await asyncio.to_thread(post, port, "/nope", "", token)
    check((r, r2, r4) == (403, 403, 404) and r3 in (413, -1) and sess.results.empty(),
          f"http: no token 403, wrong token 403, oversize 413 (or reset), unknown 404; nothing queued {(r, r2, r3, r4)}")

    sess.post("speech_start", 1.0)  # the user is talking: Claude's result must wait
    await until(lambda: sess.user_talking)
    r = await asyncio.to_thread(post, port, "/event", "Fixed parser.py; all 41 tests pass.", token)
    await asyncio.sleep(0.5)
    held = not sess.turn_running() and not said().startswith("All the tests")
    hearing["on"] = True  # still transcribing
    sess.post("discard", None)
    await asyncio.sleep(0.4)
    held = held and not sess.turn_running()
    hearing["on"] = False
    await until(lambda: said() == "All the tests pass now.")
    check(r == 204 and held, "announcer: a result waits while the user talks and while speech is transcribed")
    check(brain.history[-2]["content"].startswith(f'{EVENT} "Fixed parser.py') and said() == "All the tests pass now.",
          "announcer: then the front relays it as a [task finished] turn")

    out.clear()
    r = await asyncio.to_thread(post, port, "/event", "Done. next steps: play the test sentences and report back.", token)
    await until(lambda: said() == "I'll get the test sentences playing.", 5)
    await turn_done()
    check(r == 204 and not emitted("delegate") and delegated() == ""
          and any("announces a result" in o["text"] for o in emitted("log")),
          "announcer: a result listing next steps is reported, never re-delegated (no delegation loop)")

    tts.started.clear()
    player.ended.clear()
    utter("tell me something long")
    await until(lambda: sess.state == "speaking")
    await asyncio.sleep(0.6)
    check(len(tts.started) >= 2 and player.ended and tts.started[1][0] < player.ended[0],
          "voice: the next sentence is synthesized while the current one plays")
    sess.post("speech_start", 1.0)
    await turn_done()
    spoken = said()
    check(brain.interrupted and spoken.endswith("...") and "sentence number 0 of a long answer." in spoken
          and "number 11" not in spoken, f"barge-in: cuts the front, commits only what was heard {spoken!r}")
    sess.post("discard", None)
    await until(lambda: sess.state == "listening")

    utter("tell me something long again")
    await until(lambda: sess.state == "speaking")
    await asyncio.sleep(0.3)
    r = await asyncio.to_thread(post, port, "/stop", "", token)
    await turn_done()
    check(r == 204 and brain.interrupted and said().endswith("..."), "POST /stop cuts the front")

    out.clear()
    utter("boom")
    await until(lambda: emitted("delegate"))
    await turn_done()
    check(seen[-1]["messages"][-1]["content"] == f"{INTERRUPTED} boom", "front: after a cut, the next user message says so")
    check(emitted("delegate") == [{"type": "delegate", "text": "boom"}] and "HTTP 500" in emitted("warn")[0]["text"],
          f"front 5xx: warns, and the request goes straight to Claude {[o['text'][:60] for o in emitted('warn')]}")

    out.clear()
    utter("garbled")
    await turn_done()
    await until(lambda: said() == "Still here.", 3)
    check(said() == "Still here." and not emitted("delegate")
          and any("malformed" in o["text"] for o in emitted("log")), "front: a malformed stream line is skipped, the turn goes on")

    out.clear()
    utter("thinking leak")
    await turn_done()
    await until(lambda: said().startswith("Hello"), 3)
    check(sess.turn_spoken == ['Hello "there", \u00e9t\u00e9.'] and said() == sess.turn_spoken[0],
          f"front: a think block ahead of the JSON turn is not spoken; escapes decode {sess.turn_spoken}")

    out.clear()
    utter("notecall")
    await until(lambda: emitted("delegate"), 3)
    await turn_done()
    check(emitted("delegate") == [{"type": "delegate", "text": "Rebuild the index"}] and sess.turn_spoken[0] == "Sure."
          and not any("[" in t for t in sess.turn_spoken) and delegated() == "Rebuild the index",
          f"front: a written delegation note is silent and still delegates {sess.turn_spoken}")

    out.clear()
    utter("badargs")
    await until(lambda: said() == "I'll let you know.", 5)
    await turn_done()
    check(not emitted("delegate") and said() == "I'll let you know." and not delegated(),
          "front: a JSON turn cut short still speaks, and an unfinished delegate field hands off nothing")

    out.clear()
    utter("plaintext")
    await until(lambda: said() == "Plain words. No JSON here.", 5)
    await turn_done()
    check(said() == "Plain words. No JSON here." and any("plain text" in o["text"] for o in emitted("log")),
          "front: a server that ignores response_format is spoken as plain text, with a log line")

    out.clear()
    utter("slow work please")
    await until(lambda: sess.state == "speaking", 5)
    await asyncio.sleep(0.2)
    sess.post("speech_start", 1.0)  # barge in while "on it" is spoken, before the delegate field has streamed
    await turn_done()
    cut_early = not emitted("delegate")
    await until(lambda: emitted("delegate"), 5)
    asked = next(i for i, m in enumerate(brain.history) if m["content"] == f"{INTERRUPTED} slow work please"
                 or m["content"] == "slow work please")
    entry = json.loads(brain.history[asked + 1]["content"]) if len(brain.history) > asked + 1 else {}
    check(cut_early and emitted("delegate") == [{"type": "delegate", "text": "Rebuild the cache"}]
          and entry.get("delegate") == "Rebuild the cache" and str(entry.get("say", "")).startswith("Working")
          and "part 7" not in str(entry.get("say", "")),
          f"front: a turn cut while 'on it' is spoken still hands off its delegate field, and history shows it {entry}")
    sess.post("discard", None)
    await until(lambda: sess.state == "listening")

    dead = make_brain("llamacpp", "http://127.0.0.1:9", "")
    assert dead is not None
    out.clear()
    check(not await warm_up(dead) and "127.0.0.1:9" in emitted("warn")[-1]["text"], "front down: warm-up warns with the URL")
    sess.brain = dead
    out.clear()
    utter("what time is it")
    await until(lambda: emitted("delegate"))
    await turn_done()
    check(emitted("delegate") == [{"type": "delegate", "text": "what time is it"}] and emitted("warn"),
          "front down: the utterance goes straight to Claude")
    body = "It is noon.\n\n- a\n- b\n\nThe clock is in the corner. More."
    r = await asyncio.to_thread(post, port, "/event", body, token)
    await until(lambda: sess.turn_spoken == ["It is noon.", "The clock is in the corner."], 5)
    check(sess.turn_spoken == ["It is noon.", "The clock is in the corner."],
          f"front down: Claude's answer read out (its brief, not the retell framing) {sess.turn_spoken}")
    await dead.aclose()
    sess.brain = brain

    utter("okay goodbye")
    check(await asyncio.to_thread(said_goodbye.wait, 5), "front: goodbye ends the session")

    r = await asyncio.to_thread(post, port, "/quit", "", token)
    check(r == 204 and await until(running.done, 3) and quit_seen.is_set(), "POST /quit stops the front session")
    server.shutdown()
    server.server_close()

    # -- live mode -------------------------------------------------------------------------------------------
    out.clear()
    tts.started.clear()
    player.ended.clear()
    live = LiveSession(voice, lambda: False)
    server = protocol.serve(live.routes(), token)
    port = server.server_port
    running = asyncio.create_task(live.run())
    r = await asyncio.to_thread(post, port, "/speak", "First sentence here. Second one, a bit longer. Third.", token)
    await until(lambda: emitted("spoken"))
    check(r == 204 and emitted("spoken") == [{"type": "spoken", "text": "First sentence here. Second one, a bit longer. Third.", "cut": False}],
          f"live: /speak is read out in full {emitted('spoken')}")
    check(len(player.ended) == 3 and tts.started[1][0] < player.ended[0] and tts.started[2][0] < player.ended[1],
          "live: synthesis runs one sentence ahead of playback")
    out.clear()
    long = " ".join(f"Sentence {i} goes on for a while here." for i in range(10))
    await asyncio.to_thread(post, port, "/speak", long, token)
    await until(lambda: live.state == "speaking")
    await asyncio.sleep(0.5)
    live.post("speech_start", 1.0)
    await until(lambda: emitted("spoken"))
    sp = emitted("spoken")[0] if emitted("spoken") else {}
    check(emitted("barge_in") and sp.get("cut") is True and sp["text"].endswith("...") and "Sentence 9" not in sp["text"],
          f"live: speech over playback cuts it, spoken reports what was heard {sp.get('text', '')[:70]!r}")
    live.post("transcribing", None)
    live.post("utterance", "hello there")
    await until(lambda: emitted("utterance"))
    check(emitted("utterance") == [{"type": "utterance", "text": "hello there"}], "live: utterance reaches the mod")
    live.stop()
    check(await until(running.done, 3), "live: stop ends the session")
    server.shutdown()
    server.server_close()
    protocol.capture(None)


# -- the process lifecycle, in a child ----------------------------------------------------------------------
def spawn_child(mode: str, front_url: str, shell_parent: bool = False) -> tuple[subprocess.Popen, dict[str, Any], int | None]:
    cmd = [sys.executable, str(MAIN), "--mode", mode, "--fake-audio", "--front-url", front_url]
    child_pid = None
    if shell_parent:  # sh is the parent: kill it outright and the sidecar is orphaned
        p = subprocess.Popen(["sh", "-c", '"$@" & echo "$!"; wait', "sh", *cmd], stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, text=True)
        child_pid = int(p.stdout.readline())
    else:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    lines: list[dict[str, Any]] = []
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        raw = p.stdout.readline()
        if not raw:
            break
        lines.append(json.loads(raw))  # anything but JSON on stdout fails here
        if lines[-1]["type"] == "ready":
            return p, lines[-1], child_pid
    raise RuntimeError(f"child never reported ready: {lines}")


def gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return False
    except ProcessLookupError:
        return True
    except PermissionError:
        return False


def selftest_lifecycle(check: Checker, front_url: str) -> None:
    p, ready, _ = spawn_child("front", front_url)
    check(post(ready["port"], "/event", "x") == 403, "child: a POST without the token is refused")
    t = time.monotonic()
    check(post(ready["port"], "/quit", "", ready["token"]) == 204, "child: POST /quit accepted")
    try:
        code = p.wait(timeout=QUIT_WAIT)
    except subprocess.TimeoutExpired:
        p.kill()
        code = None
    check(code == 0, f"child: /quit exits 0 in {time.monotonic() - t:.2f}s")

    if sys.platform == "win32":
        print("SKIP  SIGTERM, parent death and closed stdout: POSIX only", flush=True)
        return
    p, ready, _ = spawn_child("live", front_url)
    t = time.monotonic()
    p.send_signal(signal.SIGTERM)
    try:
        code = p.wait(timeout=QUIT_WAIT)
    except subprocess.TimeoutExpired:
        p.kill()
        code = None
    check(code == 0, f"child: SIGTERM exits 0 in {time.monotonic() - t:.2f}s")

    p, ready, _ = spawn_child("live", front_url)
    p.stdout.close()  # the mod stopped reading
    t = time.monotonic()
    post(ready["port"], "/stop", "", ready["token"])  # makes the child write (barge_in)
    try:
        code = p.wait(timeout=QUIT_WAIT)
    except subprocess.TimeoutExpired:
        p.kill()
        code = None
    check(code == 0, f"child: a closed stdout ends it in {time.monotonic() - t:.2f}s")

    p, ready, pid = spawn_child("front", front_url, shell_parent=True)
    assert pid is not None
    t = time.monotonic()
    p.kill()  # SIGKILL the parent: no cleanup, no forwarded signal
    p.wait()
    deadline = time.monotonic() + QUIT_WAIT
    while time.monotonic() < deadline and not gone(pid):
        time.sleep(0.1)
    ok = gone(pid)
    if not ok:
        os.kill(pid, signal.SIGKILL)
    check(ok, f"child: orphaned by a killed parent, it exits in {time.monotonic() - t:.2f}s")


QUIT_WAIT = 8.0


# -- the real backends, when their weights are on disk ---------------------------------------------------------
def speech_signal() -> tuple[np.ndarray, float, float] | None:
    """A spoken sentence at 16 kHz in 2 s of lead silence and 5 s of tail, from say or Kokoro; None if neither."""
    from .speech import KokoroTTS, SayTTS

    text = "Please open the parser file and fix the failing test."
    try:
        tts = SayTTS() if sys.platform == "darwin" else KokoroTTS()
        wav = tts.synth(text)
    except Exception as e:  # noqa: BLE001
        print(f"SKIP  real speech: no synthesizer ({type(e).__name__}: {e})", flush=True)
        return None
    speech = np.interp(np.linspace(0, len(wav) - 1, int(len(wav) * SR / tts.sample_rate)), np.arange(len(wav)), wav).astype(np.float32)
    voiced = np.flatnonzero(np.abs(speech) > 0.02)
    onset, offset = 2 + voiced[0] / SR, 2 + voiced[-1] / SR
    noise = (0.002 * np.random.default_rng(0).standard_normal(SR * 7 + len(speech))).astype(np.float32)
    return np.concatenate([np.zeros(SR * 2, np.float32), speech, np.zeros(SR * 5, np.float32)]) + noise, onset, offset


def frames_of(det, signal_: np.ndarray) -> list[tuple[float, str, Any]]:
    got = []
    for i in range(0, len(signal_) - FRAME + 1, FRAME):
        got += [((i + FRAME) / SR, k, v) for k, v in det.feed(signal_[i:i + FRAME])]
    return got


def selftest_backends(check: Checker) -> None:
    from . import speech

    sig = speech_signal()
    if sig is None:
        return
    signal_, onset, offset = sig

    try:
        from huggingface_hub import try_to_load_from_cache

        whisper_cached = isinstance(try_to_load_from_cache("Systran/faster-whisper-base.en", "model.bin"), str)
    except Exception:  # noqa: BLE001
        whisper_cached = False
    if (speech.CACHE / "silero_vad.onnx").exists() and whisper_cached:
        det, transcribe = speech.recognizer("whisper", "base.en", 700, lambda: False)
        ev = frames_of(det, signal_)
        utts = [v for _, k, v in ev if k == "utterance"]
        text = transcribe(utts[0]) if len(utts) == 1 else ""
        check(len(utts) == 1 and "test" in text.lower(), f"whisper: Silero endpoints one utterance, transcribed {text!r}")
    else:
        print("SKIP  whisper backend: Silero or faster-whisper base.en not downloaded", flush=True)

    if (speech.CACHE / "kokoro-v1.0.onnx").exists() and (speech.CACHE / "voices-v1.0.bin").exists():
        t = time.monotonic()
        try:
            k = speech.KokoroTTS()
            wav = k.synth("Kokoro is speaking.")
            check(len(wav) > k.sample_rate * 0.5 and np.abs(wav).max() > 0.01,
                  f"kokoro: synthesizes speech ({len(wav) / k.sample_rate:.1f}s, loaded in {time.monotonic() - t:.1f}s)")
        except Exception as e:  # noqa: BLE001
            check(False, f"kokoro: {type(e).__name__}: {e}")
    else:
        print("SKIP  kokoro: model not downloaded", flush=True)

    from .kyutai_cuda import KyutaiCudaSTT, kyutai_backend

    backend, why = kyutai_backend()
    if backend is None or not speech.kyutai_cached():
        print(f"SKIP  kyutai backend: {why or 'weights not downloaded'}", flush=True)
        return
    t = time.monotonic()
    stt = speech.KyutaiSTT() if backend == "mlx" else KyutaiCudaSTT()
    print(f"      kyutai STT on {backend.upper()} loaded in {time.monotonic() - t:.1f}s", flush=True)
    stt.reset()
    t = time.monotonic()
    ev = frames_of(KyutaiTurns(stt, Tuning(end_silence_ms=1500), lambda: False), signal_)
    took = time.monotonic() - t
    start = next((at for at, k, _ in ev if k == "speech_start"), None)
    utt = next(((at, v) for at, k, v in ev if k == "utterance"), None)
    one = sum(k == "utterance" for _, k, _ in ev) == 1
    check(start is not None and utt is not None and one and "test" in utt[1].lower(),
          f"kyutai: one turn, one utterance {[(round(a, 2), k, v) for a, k, v in ev]}")
    if start is not None and utt is not None:
        print(f"      kyutai: first word {start - onset:.2f}s after speech onset, end of turn {utt[0] - offset:.2f}s after "
              f"speech end, {took / (len(signal_) / SR):.2f}x realtime compute", flush=True)
    stt.reset()
    ev2 = frames_of(KyutaiTurns(stt, Tuning(end_silence_ms=1500), lambda: True), signal_)
    start2 = next((at for at, k, _ in ev2 if k == "speech_start"), None)
    check(start is not None and start2 is not None and start2 > start, f"kyutai: barge-in waits for {BARGE_IN_WORDS} words: {start} -> {start2}")


def selftest() -> int:
    if units():
        return 1
    check = Checker()
    asyncio.run(selftest_sessions(check))
    url, _ = fake_front()
    selftest_lifecycle(check, url)
    selftest_backends(check)
    print("ALL PASS" if check.ok else "SOME CHECKS FAILED", flush=True)
    return 0 if check.ok else 1
