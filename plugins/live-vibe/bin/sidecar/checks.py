"""--unit: the pure pieces (protocol encoder, stdout guard, SentenceSplitter, TurnDetector, KyutaiTurns' turn logic,
SSE parsing, the spoken-switch pattern). --selftest: --unit, then both sessions end to end against a fake front
server with a silent player, the token guard, the process lifecycle in a child, and the real speech backends
when their weights are already downloaded. No mic, no speaker, no network after the first downloads."""
from __future__ import annotations

import asyncio
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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import protocol
from .audio import FRAME, SR, CannotStart, Player, check_devices, SentenceSplitter, Tuning, TurnDetector, Voice, speakable
from .front import (EVENT, HISTORY_MAX, _FALLBACKS, _NO_EFFORT, Brain, Delegator, FrontSession, LlamaCppBrain,
                    make_brain, parse_sse, spoken_model, warm_up)
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


def run_kyutai(script: list[tuple[str | None, float]], speaking: bool = False) -> list[tuple[int, str, Any]]:
    turns = KyutaiTurns(FakeKyutai(script), Tuning(end_silence_ms=1500), lambda: speaking)
    block = KYUTAI_BLOCK * SR // KYUTAI_SR
    return [(i, k, v) for i in range(len(script) + 25) for k, v in turns.feed(np.zeros(block, np.float32))]


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
    b = Brain()
    for i in range(HISTORY_MAX):
        b.begin(f"u{i}")
        b.commit(f"a{i}", [], False)
    b.begin("last")
    check(len(b.history) <= HISTORY_MAX and b.history[0]["role"] == "user" and b.history[-1]["content"] == "last",
          f"front history capped at {HISTORY_MAX}, starting on a user turn")

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

    print("UNIT: ALL PASS" if check.ok else "UNIT: SOME CHECKS FAILED", flush=True)
    return 0 if check.ok else 1


# =============================================================================================================
# self-test
# =============================================================================================================
def fake_front() -> tuple[str, list[dict[str, Any]]]:
    """An OpenAI-compatible SSE server whose reply depends on the last message: delegates 'fix', relays
    [task finished], fails on 'boom' (500), sends a malformed line on 'garbled', bad tool JSON on 'badargs'."""
    seen: list[dict[str, Any]] = []

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
            if last["role"] == "tool":
                deltas = [{"content": "I'll let you know."}]
            elif text.startswith(EVENT):
                deltas = [{"content": "All the tests pass now."}]
            elif "goodbye" in text.lower():
                deltas = [{"content": "Goodbye."}]
            elif "garbled" in text:
                raw, deltas = ["data: {not json"], [{"content": "Still here."}]
            elif "badargs" in text:
                deltas = [{"tool_calls": [{"index": 0, "id": "c9", "function": {"name": "delegate", "arguments": "{not json"}}]}]
            elif "fix" in text:
                args = json.dumps({"request": "Fix the failing test in parser.py"})
                deltas = [{"content": "On it. "},
                          {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "delegate", "arguments": args[:20]}}]},
                          {"tool_calls": [{"index": 0, "function": {"arguments": args[20:]}}]}]
            else:
                deltas = [{"content": f"This is sentence number {i} of a long answer. "} for i in range(12)]
            for r in raw:
                self.wfile.write(f"{r}\n\n".encode())
            for d in deltas:
                self.wfile.write(f"data: {json.dumps({'choices': [{'delta': d}]})}\n\n".encode())
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
    except urllib.error.URLError:  # e.g. reset while still sending a body the server already refused
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
    check([t["function"]["name"] for t in w["tools"]] == ["delegate"] and w["chat_template_kwargs"] == {"enable_thinking": False}
          and w["max_tokens"] == 1 and "one assistant" in w["messages"][0]["content"],
          "front: warm-up sends the front prompt, only delegate, thinking off")

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
    check("[delegated: Fix the failing test" in brain.history[-1]["content"], f"front: spoke and noted the delegation {brain.history[-1]['content']!r}")
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
    held = not sess.turn_running() and not brain.history[-1]["content"].startswith("All the tests")
    hearing["on"] = True  # still transcribing
    sess.post("discard", None)
    await asyncio.sleep(0.4)
    held = held and not sess.turn_running()
    hearing["on"] = False
    await until(lambda: brain.history[-1]["content"] == "All the tests pass now.")
    check(r == 204 and held, "announcer: a result waits while the user talks and while speech is transcribed")
    check(brain.history[-2]["content"].startswith(f"{EVENT} Fixed parser.py") and brain.history[-1]["content"] == "All the tests pass now.",
          "announcer: then the front relays it as a [task finished] turn")

    tts.started.clear()
    player.ended.clear()
    utter("tell me something long")
    await until(lambda: sess.state == "speaking")
    await asyncio.sleep(0.6)
    check(len(tts.started) >= 2 and player.ended and tts.started[1][0] < player.ended[0],
          "voice: the next sentence is synthesized while the current one plays")
    sess.post("speech_start", 1.0)
    await turn_done()
    last = brain.history[-1]["content"]
    spoken = last.removesuffix(" [cut off by the user]")
    check(last.endswith("[cut off by the user]") and spoken.endswith("...") and "sentence number 0 of a long answer." in spoken
          and "number 11" not in spoken, f"barge-in: cuts the front, commits only what was heard {last!r}")
    sess.post("discard", None)
    await until(lambda: sess.state == "listening")

    utter("tell me something long again")
    await until(lambda: sess.state == "speaking")
    await asyncio.sleep(0.3)
    r = await asyncio.to_thread(post, port, "/stop", "", token)
    await turn_done()
    check(r == 204 and brain.history[-1]["content"].endswith("[cut off by the user]"), "POST /stop cuts the front")

    out.clear()
    utter("boom")
    await until(lambda: emitted("delegate"))
    await turn_done()
    check(emitted("delegate") == [{"type": "delegate", "text": "boom"}] and "HTTP 500" in emitted("warn")[0]["text"],
          f"front 5xx: warns, and the request goes straight to Claude {[o['text'][:60] for o in emitted('warn')]}")

    out.clear()
    utter("garbled")
    await turn_done()
    await until(lambda: brain.history[-1]["content"] == "Still here.", 3)
    check(brain.history[-1]["content"] == "Still here." and not emitted("delegate")
          and any("malformed" in o["text"] for o in emitted("log")), "front: a malformed stream line is skipped, the turn goes on")

    out.clear()
    utter("badargs")
    await until(lambda: brain.history[-1]["content"] == "I'll let you know.", 5)
    tool_msg = next(m for m in seen[-1]["messages"] if m["role"] == "tool")
    check(not emitted("delegate") and tool_msg["content"].startswith("error") and brain.history[-1]["content"] == "I'll let you know.",
          "front: a tool call with bad JSON gets an error result, the session goes on")

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
    r = await asyncio.to_thread(post, port, "/event", "It is noon. The clock is in the corner. More.", token)
    await until(lambda: sess.turn_spoken == ["It is noon.", "The clock is in the corner."], 5)
    check(sess.turn_spoken == ["It is noon.", "The clock is in the corner."], f"front down: Claude's answer read out {sess.turn_spoken}")
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

    if not speech.is_apple_silicon() or not speech.kyutai_cached():
        print("SKIP  kyutai backend: weights not downloaded (or not Apple silicon)", flush=True)
        return
    t = time.monotonic()
    stt = speech.KyutaiSTT()
    print(f"      kyutai STT loaded in {time.monotonic() - t:.1f}s", flush=True)
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
