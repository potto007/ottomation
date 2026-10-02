"""/livevibe: a small fast model talks with the user in full duplex and hands real work to Claude with its one
tool, delegate. Claude's results arrive on POST /event and are announced once the floor is free.

Brain, LlamaCppBrain, AnthropicBrain, the turn (Session.run_turn) and the announcer are ported from
proto/duplex_voice.py; the delegate tool, the front prompt, the front-down fallbacks and the announcer's floor
check come from the first live-vibe sidecar."""
from __future__ import annotations

import asyncio
import contextlib
import json
import re
import time
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from .audio import SentenceSplitter, Voice, split_all
from .protocol import Route, emit, log, warn
from .session import Duplex, cancel_and_wait

EVENT = "[task finished]"
EVENT_CHARS = 6000  # a long report is cut for the small front model; Claude's full answer is on screen
HISTORY_MAX = 40  # ponytail: the front keeps its last 40 messages; summarize older ones if long sessions forget
TOOL_ROUNDS = 4
RESULTS_MAX = 32  # Claude results waiting for the floor

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


class Delegator:
    """The front's one tool: hands work to Claude over stdout."""

    def __init__(self) -> None:
        self.notes: list[str] = []

    async def call(self, name: str, args: Any) -> str:
        if not isinstance(args, dict):
            return 'error: the arguments were not valid JSON. Call delegate again with {"request": "..."}.'
        request = str(args.get("request") or "").strip()
        if name != "delegate" or not request:
            return f"error: use delegate with a request (got {name})"
        emit(type="delegate", text=request)
        self.notes.append(f"delegated: {request[:100]}")
        return f"Handed off; it runs in the background. Keep talking. The result arrives as a {EVENT} message."


# -- brains -----------------------------------------------------------------------------------------------
class Brain:
    """Streams text deltas and runs its own tool loop. History is plain text turns, so a cut turn is easy to commit."""

    name = "base"
    model = ""

    def __init__(self, system: str = ""):
        self.system = system
        self.history: list[dict[str, str]] = []

    @property
    def where(self) -> str:
        return self.name

    def begin(self, user_text: str) -> None:
        self.history.append({"role": "user", "content": user_text})
        if len(self.history) > HISTORY_MAX:
            del self.history[: len(self.history) - HISTORY_MAX]
            while self.history and self.history[0]["role"] != "user":
                del self.history[0]

    def commit(self, spoken: str, notes: list[str], interrupted: bool) -> None:
        text = spoken.strip()
        if notes:
            text += " [" + "; ".join(notes) + "]"
        if interrupted:
            text += " [cut off by the user]"
        self.history.append({"role": "assistant", "content": text or "(said nothing)"})

    def respond(self, user_text: str, tools: Delegator) -> AsyncIterator[str]:
        raise NotImplementedError

    async def warm_up(self) -> None:
        """One tiny request with the real system prompt, so the first spoken turn is not a cold one."""

    async def aclose(self) -> None:
        pass


class Echo(Brain):
    """Says its input verbatim: Claude's answer read out while the front model is down."""

    name = "echo"

    async def respond(self, user_text: str, tools: Delegator) -> AsyncIterator[str]:
        self.begin(user_text)
        yield user_text


def parse_sse(line: str) -> tuple[str, dict[str, Any] | None]:
    """One line of an OpenAI-compatible stream -> ('delta', delta) | ('done', None) | ('skip', None) | ('bad', None)."""
    if not line.startswith("data:"):
        return "skip", None
    payload = line[5:].strip()
    if payload == "[DONE]":
        return "done", None
    try:
        choices = json.loads(payload)["choices"]
        delta = (choices[0].get("delta") or {}) if choices else {}
    except (ValueError, KeyError, TypeError, IndexError, AttributeError):
        return "bad", None
    return ("delta", delta) if isinstance(delta, dict) else ("bad", None)


class LlamaCppBrain(Brain):
    """llama-server (llama.cpp) or any OpenAI-compatible endpoint; llama-server needs --jinja for tools.

    Sampling matches the FTL spec-decoding bench on the M5 Pro (Qwen3.6-35B-A3B Q4_0, DFlash2 n=3). Thinking is off
    per request through chat_template_kwargs: a spoken reply cannot wait for a thinking block."""

    name = "llamacpp"
    SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}

    def __init__(self, system: str, url: str, model: str):
        super().__init__(system)
        import httpx

        self.url = url.rstrip("/")
        self.model = model or "default"
        # read: the longest gap between two streamed chunks; a model that stalls that long has failed this turn
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0, write=10.0, pool=5.0))
        self.tools = [{"type": "function", "function": DELEGATE}]

    @property
    def where(self) -> str:
        return f"{self.url} model {self.model}"

    async def warm_up(self) -> None:
        body = {"model": self.model, "max_tokens": 1, "stream": False, **self.SAMPLING,
                "messages": [{"role": "system", "content": self.system}, {"role": "user", "content": "Hi."}],
                "tools": self.tools, "chat_template_kwargs": {"enable_thinking": False}}
        r = await self.client.post(f"{self.url}/v1/chat/completions", json=body, timeout=120.0)  # a router may load the model now
        r.raise_for_status()

    async def respond(self, user_text: str, tools: Delegator) -> AsyncIterator[str]:
        self.begin(user_text)
        msgs: list[dict[str, Any]] = [{"role": "system", "content": self.system}, *self.history]
        for _round in range(TOOL_ROUNDS):
            content, calls, bad = "", {}, 0
            body = {"model": self.model, "messages": msgs, "tools": self.tools, "tool_choice": "auto",
                    "stream": True, "max_tokens": 400, **self.SAMPLING,
                    "chat_template_kwargs": {"enable_thinking": False}}
            async with self.client.stream("POST", f"{self.url}/v1/chat/completions", json=body) as r:
                if r.status_code >= 400:
                    await r.aread()
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:160]}")
                async for line in r.aiter_lines():
                    kind, delta = parse_sse(line)
                    if kind == "done":
                        break
                    if kind == "bad":
                        bad += 1
                    if delta is None:
                        continue
                    if isinstance(delta.get("content"), str) and delta["content"]:
                        content += delta["content"]
                        yield delta["content"]
                    for tc in delta.get("tool_calls") or []:
                        if not isinstance(tc, dict):
                            continue
                        e = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "args": ""})
                        e["id"] = tc.get("id") or e["id"]
                        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                        e["name"] += str(fn.get("name") or "")
                        e["args"] += str(fn.get("arguments") or "")
            if bad:
                log(f"front: skipped {bad} malformed stream line(s) from {self.url}")
            if not calls:
                return
            msgs.append({"role": "assistant", "content": content or None, "tool_calls": [
                {"id": e["id"] or f"call_{i}", "type": "function", "function": {"name": e["name"], "arguments": e["args"]}}
                for i, e in calls.items()]})
            for i, e in calls.items():
                try:
                    args = json.loads(e["args"] or "{}")
                except ValueError:
                    args = None
                msgs.append({"role": "tool", "tool_call_id": e["id"] or f"call_{i}", "content": await tools.call(e["name"], args)})

    async def aclose(self) -> None:
        await self.client.aclose()


# Haiku 4.5 and Sonnet 4.5 reject output_config.effort; the server-side fallback takes only the newest models.
_NO_EFFORT = re.compile(r"haiku-4-5|sonnet-4-5")
_FALLBACKS = re.compile(r"^claude-(fable-5-1|opus-5-5|opus-5|sonnet-5-5)$")


class AnthropicBrain(Brain):
    """Claude through the Anthropic SDK: streaming, a manual tool loop."""

    name = "anthropic"

    def __init__(self, system: str, model: str):
        super().__init__(system)
        import anthropic

        self.client = anthropic.AsyncAnthropic(timeout=60.0, max_retries=1)
        self.model = model or "claude-haiku-4-5"
        self.tools = [{"name": DELEGATE["name"], "description": DELEGATE["description"],
                       "input_schema": DELEGATE["parameters"], "strict": True, "eager_input_streaming": True}]

    @property
    def where(self) -> str:
        return f"Anthropic {self.model}"

    async def warm_up(self) -> None:
        await self.client.messages.create(model=self.model, max_tokens=1, messages=[{"role": "user", "content": "Hi."}])

    async def respond(self, user_text: str, tools: Delegator) -> AsyncIterator[str]:
        self.begin(user_text)
        msgs: list[dict[str, Any]] = [dict(m) for m in self.history]
        kw: dict[str, Any] = {"model": self.model, "max_tokens": 1024, "system": self.system, "tools": self.tools}
        if not _NO_EFFORT.search(self.model):
            kw["output_config"] = {"effort": "low"}  # voice: latency over depth
        if _FALLBACKS.match(self.model):
            kw["betas"], kw["fallbacks"] = ["server-side-fallback-2026-07-01"], "default"
        for _round in range(TOOL_ROUNDS):
            async with self.client.beta.messages.stream(messages=msgs, **kw) as stream:
                async for ev in stream:
                    if ev.type == "text":
                        yield ev.text
                msg = await stream.get_final_message()
            if msg.stop_reason == "refusal":
                yield " I can't help with that one."
                return
            uses = [b for b in msg.content if b.type == "tool_use"]
            if not uses or msg.stop_reason == "max_tokens":
                return
            msgs.append({"role": "assistant", "content": msg.content})
            # eager input streaming leaves validation to us: Delegator checks the input's shape
            msgs.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": b.id, "content": await tools.call(b.name, b.input)} for b in uses]})

    async def aclose(self) -> None:
        await self.client.close()


def make_brain(backend: str, url: str, model: str) -> Brain | None:
    system = FRONT_PROMPT.format(workspace=Path.cwd())
    try:
        if backend == "anthropic":
            return AnthropicBrain(system, model)
        return LlamaCppBrain(system, url, model)
    except Exception as e:  # noqa: BLE001
        warn(f"front backend {backend} cannot start ({type(e).__name__}: {str(e)[:160]}). Anthropic needs "
             "ANTHROPIC_API_KEY or `ant auth login`; or set frontBackend to llamacpp. Speech goes straight to Claude.")
        return None


async def warm_up(brain: Brain) -> bool:
    t = time.monotonic()
    try:
        await brain.warm_up()
    except Exception as e:  # noqa: BLE001
        warn(f"front model unreachable at {brain.where} ({type(e).__name__}: {str(e)[:160]}). "
             "Until it answers, speech goes straight to Claude.")
        return False
    log(f"front: {brain.name} at {brain.where}, warm in {time.monotonic() - t:.1f}s")
    return True


def spoken_model(pattern: re.Pattern[str] | None, text: str) -> str | None:
    """The mod's own pattern (--switch-pattern) over the mod's own normalization (spokenModel in register.tsx)."""
    if pattern is None:
        return None
    m = pattern.search(re.sub(r"\s+", " ", re.sub(r"[^a-z ]+", " ", text.lower())).strip())
    return m.group(1) if m else None


_GOODBYE = re.compile(r"\b(goodbye|bye bye)\b")


# -- the session ------------------------------------------------------------------------------------------
class FrontSession(Duplex):
    def __init__(self, voice: Voice, hearing: Callable[[], bool], brain: Brain | None,
                 switch: re.Pattern[str] | None, goodbye: Callable[[], None]):
        super().__init__(voice, hearing)
        self.brain, self.switch, self.goodbye = brain, switch, goodbye
        self.tools = Delegator()
        self.results: asyncio.Queue[str] = asyncio.Queue(maxsize=RESULTS_MAX)
        self.turn: asyncio.Task | None = None
        self.turn_spoken: list[str] = []

    def routes(self) -> dict[str, Route]:
        def event(body: str) -> bool:
            try:
                self.results.put_nowait(body)
                return True
            except asyncio.QueueFull:
                return False

        def stop(_: str) -> bool:
            self.spawn(self.interrupt())
            return True

        return {"/event": self.on_loop(event), "/stop": self.on_loop(stop)}

    def turn_running(self) -> bool:
        return self.turn is not None and not self.turn.done()

    def floor_free(self) -> bool:
        return not (self.user_talking or self.hearing() or self.turn_running())

    async def handle(self, kind: str, payload: Any) -> None:
        if kind == "speech_start":
            self.user_talking = True
            await self.interrupt()
            self.set_state("user_speaking")
        elif kind == "transcribing":
            self.set_state("transcribing")
        elif kind == "discard":
            self.user_talking = False
            self.settle()
        elif kind == "utterance":
            self.user_talking = False
            await self.on_utterance(payload)

    async def on_utterance(self, text: str) -> None:
        emit(type="transcript", role="user", text=text)
        model = spoken_model(self.switch, text)
        if model:
            emit(type="switch_model", model=model)  # the mod runs /model and posts the result as an event
            self.settle()
            return
        await self.interrupt()
        self.turn = asyncio.create_task(self.run_turn(text))

    async def interrupt(self) -> None:
        """Barge-in and POST /stop: cancel the running turn; it commits what was heard before it ends."""
        if self.turn_running():
            await cancel_and_wait(self.turn)
            self.settle()

    async def run_turn(self, user_text: str, is_event: bool = False, brain: Brain | None = None) -> None:
        """Front model -> sentences -> Voice, then the history gets exactly what was heard."""
        brain = brain or self.brain
        if brain is None:
            return await self.front_down(user_text, is_event)
        self.turn_spoken, self.tools.notes = [], []
        failed: list[Exception] = []

        async def sentences() -> AsyncIterator[str]:
            splitter = SentenceSplitter()
            try:
                async with contextlib.aclosing(brain.respond(user_text, self.tools)) as deltas:
                    async for delta in deltas:
                        for s in splitter.push(delta):
                            yield s
            except Exception as e:  # noqa: BLE001 - connect error, 5xx, a broken stream: the turn ends cleanly
                failed.append(e)
            for s in splitter.flush():
                yield s

        self.set_state("thinking")
        interrupted = True
        try:
            interrupted = await self.voice.speak(sentences(), self.turn_spoken, lambda: self.set_state("speaking"))
        finally:
            brain.commit(" ".join(self.turn_spoken), list(self.tools.notes), interrupted)
            said = brain.history[-1]["content"] if brain.history else ""
            if brain is self.brain and said != "(said nothing)":
                emit(type="transcript", role="front", text=said.strip())
        if failed:
            e = failed[0]
            warn(f"front model failed at {brain.where} ({type(e).__name__}: {str(e)[:160]})")
            return await self.front_down(user_text, is_event)
        self.settle()
        if not is_event and _GOODBYE.search(user_text.lower()):
            self.goodbye()

    async def front_down(self, text: str, is_event: bool) -> None:
        """Without a front model the request still reaches Claude, and Claude's answer is read out (its first two
        sentences)."""
        if not is_event:
            emit(type="delegate", text=text)
            self.settle()
            return
        short = " ".join(split_all(text.removeprefix(EVENT))[:2])
        await self.run_turn(short, is_event=True, brain=Echo())

    async def background(self) -> None:
        """The announcer: Claude's results reach the front only when the floor is free (nobody talking, nothing
        being transcribed, no turn running)."""
        while True:
            text = await self.results.get()
            while not self.floor_free():
                await asyncio.sleep(0.1)
            self.turn = asyncio.create_task(self.run_turn(f"{EVENT} {text.strip()[:EVENT_CHARS]}", is_event=True))
            await asyncio.wait({self.turn})

    async def shutdown(self) -> None:
        await cancel_and_wait(self.turn)
        if self.brain is not None:
            with contextlib.suppress(Exception):
                await self.brain.aclose()
