"""/livevibe: a small fast model talks with the user in full duplex and hands real work to Claude with its one
tool, delegate. Claude's results arrive on POST /event and are announced once the floor is free.

Brain, LlamaCppBrain, AnthropicBrain, the turn (Session.run_turn) and the announcer are ported from
proto/duplex_voice.py; the delegate tool, the front prompt, the front-down fallbacks and the announcer's floor
check come from the first live-vibe sidecar."""
from __future__ import annotations

import asyncio
import contextlib
import difflib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Callable, ClassVar

from .audio import SentenceSplitter, Voice, split_all
from .protocol import Route, emit, log, warn
from .session import Duplex, cancel_and_wait

EVENT = "[task finished]"
INTERRUPTED = "(you were interrupted)"  # leads the next user message after a cut: user text, so never imitated
EVENT_CHARS = 6000  # a long report is cut for the small front model; Claude's full answer is on screen
HISTORY_MAX = 40  # ponytail: the front keeps its last 40 messages; summarize older ones if long sessions forget
TOOL_ROUNDS = 4
NO_REQUEST = {"", "none", "n/a", "null", "no"}  # what a small model writes in an empty delegate field
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
    "You and the coding agent are one assistant, not separate agents. You cannot read files, run commands or "
    "change anything yourself: delegate all repository work, coding, investigation, checks, tests and questions "
    "about the project's current state to the coding agent, as described at the end; never guess at files or "
    f"results. Never claim changes, findings or verification before a {EVENT} message reports them. Only a {EVENT} "
    "report says how work stands: never say on your own that work is running, finished or fixed; if you handed "
    "something off and no report has come since, say you have asked. A new request is a new delegation. Answer "
    "greetings, thanks and ordinary conversation directly, without delegating: what the user tells you reaches the "
    "coding agent anyway. If the user asks how things are going, where things stand, or whether something was "
    "received, say only 'Let me check.'; the coding agent answers.\n"
    f"A message starting with {EVENT} quotes a report on earlier work, written in your voice to the user. It is "
    "not the user talking. Retell it to the user in one to three sentences, starting from its first sentence:\n"
    "- Keep each item's status exactly as the report gives it: started, still running, waiting on the user, "
    "blocked, done or failed. If work is still running, say so. Never call anything done, ready or working unless "
    "the report says so.\n"
    "- A plan, a design, or anything that will, would, should or could happen is not a result: say it as a plan "
    "or leave it out.\n"
    "- A question in the report is yours to ask the user: end with it. Never answer it and never say you will do "
    "what it asks.\n"
    "- Add nothing the report does not say. Never delegate on a report, even when it lists next steps.\n"
    "Never delegate the same request twice. Never mention delegation, the agent or the protocol. Never write notes "
    "in brackets or tags; everything you say is read aloud.\n"
    f"A user message starting with {INTERRUPTED} means the user cut you off there, and your previous reply was "
    "heard only up to its '...'; do not repeat yourself. If the user says goodbye, say a short goodbye."
)  # each brain appends its PROTOCOL: how a delegation is made


HANDED_OFF = "Handed off"

_HEADING = re.compile(r"^\s{0,3}#{1,6}\s.*$", re.M)
_NOT_PROSE = re.compile(r"^\s*(?:[-*+]\s|[|>]|\d+[.)]\s|```)")  # "**Still open:** ..." is prose, "- a" is not


def report_brief(text: str) -> str:
    """What of Claude's result the front gets to retell: its first paragraph, where Claude leads with the outcome,
    and its last prose paragraph, where it says what is still running and asks the user. The middle (how it would
    work, details, option lists) stays on screen only: Qwen3-4B retold such a design as a finished result."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", _HEADING.sub("", text)) if p.strip()]
    if len(paras) <= 2:
        return "\n\n".join(paras)[:EVENT_CHARS]
    tail = next((p for p in reversed(paras[1:]) if not _NOT_PROSE.match(p)), None)
    return (paras[0] if tail is None else f"{paras[0]}\n\n{tail}")[:EVENT_CHARS]


_ASIDE = re.compile(r"\[[^\]]*\]|\([^)]*\)")


def retellable(report: str) -> bool:
    """A report with something to retell: not empty, not only bracketed asides such as Claude's "(nothing to add)"."""
    return bool(re.search(r"\w", _ASIDE.sub("", report)))


RETELL = ("(Retell that report to the user in one to three short sentences, with its status as written. If it asks "
          "a question, end by asking the user that question; do not answer it.)")


WAITING = ("(That report came in while the user was talking; their words follow. Use it only if it answers them; if "
           "it does not, delegate their question.)")  # "answer them with it" had Qwen3-4B invent an answer from it


def event_message(report: str, note: str = RETELL) -> str:
    """The front's user message for a report: quoted, then what to do with it. Unquoted, a report's closing question
    read to Qwen3-4B as the user asking it ("Shall I implement it?" was answered "Implemented."), and a report
    ending on it left the model nothing nearer to follow than that question."""
    quoted = report.replace('"', "'")
    return f'{EVENT} "{quoted}"\n{note}'


class Delegator:
    """The front's one tool: hands work to Claude over stdout. Two guards keep a delegation loop from forming (seen
    live with Qwen3-4B: a result that listed next steps was re-delegated, Claude's short reply to that came back as a
    result, and so on, eight times): nothing is delegated while a result is being announced (`announcing`), and a
    request close to one handed off in the last REPEAT_S is not sent again. A delegation carries the user's own words
    (`said`) beside the model's reading of them: a 4B's rewrite once turned "what was the fix?" into "wait for the
    worker", and Claude saw only that."""

    REPEAT_S = 300.0
    SIMILAR = 0.85  # difflib ratio over the normalized words

    def __init__(self) -> None:
        self.announcing = False  # a [task finished] turn: results are reported, not acted on
        self.sent: list[tuple[float, str]] = []
        self.handed = 0  # delegations sent so far: a user turn that adds none is passed on as a note
        self.results_seen = False  # a result was announced or taken into a user turn

    @property
    def work_in_session(self) -> bool:
        """Something was handed off or reported this session: a user's question may be about it, so the front never
        answers one itself (see HOLD)."""
        return self.handed > 0 or self.results_seen

    @staticmethod
    def _norm(text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", text.lower()))

    def repeat_of(self, request: str) -> str | None:
        now, norm = time.monotonic(), self._norm(request)
        self.sent = [(t, r) for t, r in self.sent if now - t < self.REPEAT_S]
        for _, earlier in self.sent:
            if difflib.SequenceMatcher(None, norm, self._norm(earlier), autojunk=False).ratio() >= self.SIMILAR:
                return earlier
        return None

    async def call(self, name: str, args: Any, said: str = "") -> str:
        if not isinstance(args, dict):
            return 'error: the arguments were not valid JSON. Call delegate again with {"request": "..."}.'
        request = str(args.get("request") or "").strip()
        if name != "delegate" or not request:
            return f"error: use delegate with a request (got {name})"
        if self.announcing:
            log(f"front: not delegated, the turn announces a result: {request[:120]!r}")
            return f"Not handed off: a {EVENT} message is reported to the user, never acted on. Just tell the user."
        said = said.strip()
        key = f"{said} {request}" if said else request
        if (earlier := self.repeat_of(key)) is not None:
            log(f"front: not delegated again, it repeats {earlier[:80]!r}: {key[:120]!r}")
            return "Not handed off: the same request is already running. Tell the user it is in progress."
        emit(type="delegate", text=request, **({"said": said} if said else {}))
        self.sent.append((time.monotonic(), key))
        self.handed += 1
        return f"{HANDED_OFF}; it runs in the background. Keep talking. The result arrives as a {EVENT} message."


_NOTE_DELEGATE = re.compile(r"^\s*delegat\w*\s*[:-]\s*(.+)$", re.I | re.S)


class SpeechFilter:
    """What the model writes, minus what must never be read aloud: <think> blocks (a template can leak an empty one
    even with thinking off) and bracketed notes like "[delegated: ...]", which a model copies from anything
    bracket-shaped. Tags and notes may be split across stream deltas, so a possible start is held back until it
    resolves. The notes are kept: a "[delegated: X]" written instead of a tool call is still a request for X."""

    TAGS = ("<think>", "</think>")
    NOTE_MAX = 400  # an unclosed "[" this far back was ordinary text after all

    def __init__(self, notes: bool = True) -> None:
        self.buf = ""
        self.thinking = False
        self.catch = "<[" if notes else "<"  # notes=False strips only think blocks (ahead of the JSON turn)
        self.notes: list[str] = []

    def push(self, delta: str) -> str:
        self.buf += delta
        out: list[str] = []
        while self.buf:
            if self.thinking:
                i = self.buf.find("</think>")
                if i < 0:
                    self.buf = self.buf[-(len("</think>") - 1):]  # a closing tag may be arriving in pieces
                    break
                self.buf, self.thinking = self.buf[i + len("</think>"):], False
                continue
            starts = [i for i in map(self.buf.find, self.catch) if i >= 0]
            if not starts:
                out.append(self.buf)
                self.buf = ""
                break
            i = min(starts)
            out.append(self.buf[:i])
            self.buf = self.buf[i:]
            if self.buf[0] == "<":
                tag = next((t for t in self.TAGS if self.buf.startswith(t)), None)
                if tag:
                    self.buf, self.thinking = self.buf[len(tag):], tag == "<think>"
                elif any(t.startswith(self.buf) for t in self.TAGS):
                    break  # wait for the rest of the tag
                else:
                    out.append("<")
                    self.buf = self.buf[1:]
                continue
            j = self.buf.find("]")
            if j >= 0:
                self.notes.append(self.buf[1:j].strip())
                self.buf = self.buf[j + 1:]
            elif len(self.buf) > self.NOTE_MAX:
                out.append("[")
                self.buf = self.buf[1:]
            else:
                break  # wait for the "]"
        return "".join(out)

    def flush(self) -> str:
        rest, self.buf = self.buf, ""
        if self.thinking or rest.startswith("<"):
            self.thinking = False
            return ""
        if rest.startswith("["):  # a note cut off by the end of the reply is still a note
            self.notes.append(rest[1:].strip())
            return ""
        return rest

    def delegations(self) -> list[str]:
        """The requests in "[delegated: ...]" notes."""
        return [m.group(1).strip() for n in self.notes if (m := _NOTE_DELEGATE.match(n)) and m.group(1).strip()]


# -- brains -----------------------------------------------------------------------------------------------
def is_turn_start(m: dict[str, Any]) -> bool:
    """A user's own message, where a turn begins: tool results (role tool, or for Anthropic a user message of
    tool_result blocks) belong to the turn before them."""
    return m.get("role") == "user" and isinstance(m.get("content"), str)


class Brain:
    """Streams text deltas and runs its own tool loop. Per turn the history holds the user's message, the turn's tool
    calls and results in the backend's own structured form, then only what was heard: no annotations among the
    assistant's words, since a model imitates whatever its own past turns look like."""

    name = "base"
    model = ""
    trimmed = False  # the last reply was held and cut: the user did not hear what it said

    def __init__(self, system: str = ""):
        self.system = system
        self.history: list[dict[str, Any]] = []
        self.pending: list[dict[str, Any]] = []  # this turn's tool calls and results, committed with the words
        self.interrupted = False  # the last turn was cut: the next user message says so
        self.user_said = ""  # the user's own words this turn, as heard

    @property
    def where(self) -> str:
        return self.name

    def begin(self, user_text: str) -> None:
        self.user_said = user_text
        if self.interrupted and not user_text.startswith(EVENT):  # a report is not the user: the flag waits for them
            user_text = f"{INTERRUPTED} {user_text}"
            self.interrupted = False
        self.pending = []
        self.history.append({"role": "user", "content": user_text})
        if len(self.history) > HISTORY_MAX:  # whole turns go, so no tool result outlives its call
            del self.history[: len(self.history) - HISTORY_MAX]
            while self.history and not is_turn_start(self.history[0]):
                del self.history[0]

    def record(self, *messages: dict[str, Any]) -> None:
        """Tool calls and their results, as the turn makes them."""
        self.pending.extend(messages)

    def commit(self, spoken: str, interrupted: bool) -> str:
        """Ends the turn: its tool exchange, then what was heard (nothing when nothing was). Returns that text."""
        text = spoken.strip()
        self.history.extend(self.pending)
        self.pending = []
        if (content := self.said(text)) is not None:
            self.history.append({"role": "assistant", "content": content})
        self.interrupted = interrupted
        return text

    def said(self, text: str) -> str | None:
        """The assistant message for what was heard, or None for no message."""
        return text or None

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


ACK = "On it, I've asked."
ACK_WORDS = 12
_CLAIM = re.compile(r"\b(done|finished|complete[sd]?|ready|fixed|running|works|working|passed|failed|recorded|"
                    r"information|don't|haven't|hasn't|isn't|wasn't|no one|nobody|"
                    r"was in|likely|probably|because|due to)\b", re.I)  # the last: a short guessed cause


def is_request(text: Any) -> bool:
    """A delegate field that asks for something: not empty, not the "none" a small model writes instead."""
    return isinstance(text, str) and text.strip().lower().strip(".") not in NO_REQUEST


HOLD = "Let me check."  # all a question the front does not delegate hears from it; Claude gets it as `ask`


def short_first(say: str) -> str:
    """say's first sentence when it is short and makes no claim about the work, else ""."""
    first = re.split(r"(?<=[.!?])\s+", say.strip(), maxsplit=1)[0].strip()
    return "" if len(first.split()) > ACK_WORDS or _CLAIM.search(first) else first


def acknowledgement(say: str) -> str:
    """What a delegating turn speaks: its first sentence when that is a short "on it" with no claim about the work,
    otherwise ACK. The answer is Claude's to give; the front's memory of the work is stale by then."""
    return short_first(say) or ACK


def held_answer(say: str, worked: bool) -> str:
    """What a user's question the front does not delegate speaks; Claude gets the question as `ask` either way. With
    work in session, only HOLD: the question may be about the work, and the front's memory of it is stale. Otherwise
    the reply's first sentence when it is short and makes no claim ("I'm doing well, thanks!"), else HOLD. A guess was
    never that: every recorded one was long, made a claim, or sat in a later sentence."""
    return HOLD if worked else short_first(say) or HOLD


class TurnStream:
    """Reads the front's JSON turn, {"say": "...", "delegate": "..."}, as it streams: push() returns the newly decoded
    characters of "say", so speech starts before the object is complete; `fields` holds each finished value. A reply
    that does not start with "{" (a server that ignores response_format) is plain text, passed through whole."""

    _ESCAPES: ClassVar[dict[str, str]] = {"n": "\n", "t": "\t", "r": "", "b": "", "f": "", "/": "/", "\\": "\\", '"': '"'}

    def __init__(self, speak: str = "say") -> None:
        self.speak = speak
        self.plain: bool | None = None  # unknown until the first non-blank character
        self.fields: dict[str, str] = {}
        self.in_str = self.esc = self.value = False
        self.key = ""
        self.cur: list[str] = []
        self.hexits: str | None = None  # the digits of a \\u escape so far

    def push(self, delta: str) -> str:
        if self.plain is None:
            stripped = delta.lstrip()
            if not stripped:
                return ""
            self.plain = not stripped.startswith("{")
        if self.plain:
            return delta
        out: list[str] = []
        for ch in delta:
            if not self.in_str:
                if ch == '"':
                    self.in_str, self.cur = True, []
                elif ch == ":":
                    self.value = True
                elif ch in ",{}":
                    self.value = False
                continue
            if self.hexits is not None:
                self.hexits += ch
                if len(self.hexits) == 4:
                    code = int(self.hexits, 16) if all(c in "0123456789abcdefABCDEF" for c in self.hexits) else 0
                    self.hexits = None
                    if code and not 0xD800 <= code < 0xE000:  # a lone surrogate cannot be spoken
                        self._add(chr(code), out)
                continue
            if self.esc:
                self.esc = False
                if ch == "u":
                    self.hexits = ""
                else:
                    self._add(self._ESCAPES.get(ch, ch), out)
                continue
            if ch == "\\":
                self.esc = True
            elif ch == '"':
                self.in_str = False
                if self.value:
                    self.fields[self.key], self.value = "".join(self.cur), False
                else:
                    self.key = "".join(self.cur)
            else:
                self._add(ch, out)
        return "".join(out)

    def _add(self, text: str, out: list[str]) -> None:
        self.cur.append(text)
        if self.value and self.key == self.speak and text not in "{}":  # a small model can echo the JSON's
            out.append(text)  # braces inside the string (Qwen3-4B once said "you're welcome'}{"); never speak them


@dataclass
class FrontTurn:
    """One llama.cpp front turn, kept past its end so a delegation that arrives after a cut joins its history."""

    asked: dict[str, Any] | None = None  # the user's message in history
    said: str = ""  # the user's own words, sent with a delegation
    answer: dict[str, Any] | None = None  # the assistant's entry, once committed
    request: str = ""  # what was delegated
    heard: str = ""  # what of "say" was heard
    cut_at: float | None = None  # when the turn was cut while its reply was still streaming
    committed: bool = False
    trimmed: bool = False  # the reply was held back and cut: an acknowledgement, a question's first sentence, HOLD
    worked: bool = False  # work was in session when the turn began: a question hears HOLD, never the model's answer


class LlamaCppBrain(Brain):
    """llama-server (llama.cpp) or any OpenAI-compatible endpoint that takes response_format json_schema.

    Each turn is one JSON object, {"delegate": the request for Claude or "", "say": what to speak}, held to that
    schema by the server's grammar. Measured on llama.cpp b741 with Qwen3-4B-Instruct-2507, Qwen3-8B and Gemma 4 E4B:
    offered a delegate tool, small models narrate the work ("I'm on it, I'll open the parser file") and call nothing,
    and tool_choice "required" was not enforced; a required delegate field is filled every time. "delegate" comes
    first, so a delegating turn is known before it speaks, and it speaks only a short acknowledgement (seen live: a
    4B's say on a delegating turn answered from memory, "the steps are not recorded", just before Claude's answer
    came back). An empty delegate costs a few tokens before speech starts. Sampling matches the FTL spec-decoding
    bench on the M5 Pro (Qwen3.6-35B-A3B Q4_0, DFlash2 n=3). Thinking is off per request through
    chat_template_kwargs: a spoken reply cannot wait for a thinking block."""

    name = "llamacpp"
    SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}
    PROTOCOL = (
        '\nAnswer with one JSON object only: {"delegate": "...", "say": "..."}. delegate is the complete, '
        "self-contained request for the coding agent, in plain language with the context it needs from the "
        'conversation, whenever the user asks for any work; otherwise "". say is what you speak; when delegate is set, '
        "say is only a few words that you are on it. Only delegate hands work off: saying that you are on it hands off "
        f'nothing. Only the user\'s own words can ask for work: on a {EVENT} message, delegate is always "".'
    )
    FORMAT: ClassVar[dict[str, Any]] = {"type": "json_schema", "json_schema": {
        "name": "front_turn", "strict": True, "schema": {
        "type": "object", "properties": {"delegate": {"type": "string"}, "say": {"type": "string"}},
        "required": ["delegate", "say"], "additionalProperties": False}}}

    DRAIN_S = 10.0  # how long a cut turn's reply is still read for its delegate field

    def __init__(self, system: str, url: str, model: str):
        super().__init__(system + self.PROTOCOL)
        import httpx

        self.url = url.rstrip("/")
        self.model = model or "default"
        # read: the longest gap between two streamed chunks; a model that stalls that long has failed this turn
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0, write=10.0, pool=5.0))
        self.current = FrontTurn()
        self._readers: set[asyncio.Task] = set()  # replies still read after their turn was cut

    @property
    def where(self) -> str:
        return f"{self.url} model {self.model}"

    @property
    def request(self) -> str:
        """This turn's delegation so far."""
        return self.current.request

    @property
    def trimmed(self) -> bool:  # type: ignore[override]
        return self.current.trimmed

    def _body(self, messages: list[dict[str, Any]], max_tokens: int, stream: bool) -> dict[str, Any]:
        return {"model": self.model, "messages": messages, "response_format": self.FORMAT, "stream": stream,
                "max_tokens": max_tokens, **self.SAMPLING, "chat_template_kwargs": {"enable_thinking": False}}

    async def warm_up(self) -> None:
        msgs = [{"role": "system", "content": self.system}, {"role": "user", "content": "Hi."}]
        r = await self.client.post(f"{self.url}/v1/chat/completions", json=self._body(msgs, 1, False),
                                   timeout=120.0)  # a router may load the model now
        r.raise_for_status()

    def begin(self, user_text: str) -> None:
        super().begin(user_text)
        self.current = FrontTurn(asked=self.history[-1], said=user_text)

    def said(self, text: str) -> str | None:
        """In the shape the model answers in, so its own past turns teach the format; a delegation stays even when
        nothing of the reply was heard."""
        self.current.heard = text
        if not (text or self.current.request):
            return None
        return json.dumps({"delegate": self.current.request, "say": text}, ensure_ascii=False)

    def commit(self, spoken: str, interrupted: bool) -> str:
        text = super().commit(spoken, interrupted)
        last = self.history[-1] if self.history else None
        self.current.answer = last if last is not None and last["role"] == "assistant" else None
        self.current.committed = True
        return text

    def _amend(self, turn: FrontTurn) -> None:
        """A delegation that arrived after its turn was committed (cut while "on it" was spoken) joins that turn's
        history entry, or becomes one right after the user's message."""
        content = json.dumps({"delegate": turn.request, "say": turn.heard}, ensure_ascii=False)
        if turn.answer is not None:
            turn.answer["content"] = content
            return
        at = next((i for i, m in enumerate(self.history) if m is turn.asked), None)
        if at is not None:
            turn.answer = {"role": "assistant", "content": content}
            self.history.insert(at + 1, turn.answer)

    async def respond(self, user_text: str, tools: Delegator) -> AsyncIterator[str]:
        """Speaks the reply's "say" as it streams. The reply is read by its own task, which hands off the delegate
        field even when this turn is cut first: the user may already have heard "on it"."""
        self.begin(user_text)
        msgs: list[dict[str, Any]] = [{"role": "system", "content": self.system}, *self.history]
        turn = self.current
        words: asyncio.Queue[str | None] = asyncio.Queue()
        announcing = bool(getattr(tools, "announcing", False))
        turn.worked = bool(getattr(tools, "work_in_session", False))
        reader = asyncio.create_task(self._read(msgs, turn, words, tools, announcing))
        finished = False
        try:
            while (said := await words.get()) is not None:
                yield said
            await reader  # its delegation is made before it ends the words; raises the turn's failure
            finished = True
        finally:
            if not finished and not reader.done():
                turn.cut_at = time.monotonic()
                self._readers.add(reader)
                reader.add_done_callback(self._reader_done)

    def _reader_done(self, task: asyncio.Task) -> None:
        self._readers.discard(task)
        if not task.cancelled() and task.exception() is not None:
            e = task.exception()
            log(f"front: reading a cut turn's reply failed ({type(e).__name__}: {str(e)[:120]})")

    async def _read(self, msgs: list[dict[str, Any]], turn: FrontTurn, words: asyncio.Queue[str | None],
                    tools: Delegator, announcing: bool = False) -> None:
        # raw -> no think block (one can lead the JSON) -> the "say" field -> no notes or think blocks -> speech
        think, json_turn, speech, bad = SpeechFilter(notes=False), TurnStream(), SpeechFilter(), 0
        held: list[str] | None = None  # a delegating turn's say, kept back whole and cut to an acknowledgement
        spoke = handed = False
        try:
            async with self.client.stream("POST", f"{self.url}/v1/chat/completions",
                                          json=self._body(msgs, 400, True)) as r:
                if r.status_code >= 400:
                    await r.aread()
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:160]}")
                async for line in r.aiter_lines():
                    kind, delta = parse_sse(line)
                    if kind == "done":
                        break
                    if kind == "bad":
                        bad += 1
                    text = (delta or {}).get("content")
                    if isinstance(text, str) and text:
                        said = speech.push(json_turn.push(think.push(text)))
                        if not handed and "delegate" in json_turn.fields:  # handed off before a word is spoken,
                            handed = True  # so a turn cut while it speaks has already delegated, or never will
                            await self._hand_off(json_turn.fields["delegate"].strip(), turn, tools, announcing)
                        if held is None and not (spoke or announcing) and "delegate" in json_turn.fields and (
                                is_request(json_turn.fields["delegate"]) or "?" in turn.said):
                            held = []  # a delegation or a user's question, known before any of say was spoken
                        if said and held is not None:
                            held.append(said)
                        elif said:
                            words.put_nowait(said)
                            spoke = True
                    if turn.cut_at is not None and ("delegate" in json_turn.fields
                                                    or time.monotonic() - turn.cut_at > self.DRAIN_S):
                        break  # cut: only the delegate field was still wanted
            said = speech.push(json_turn.push(think.flush())) + speech.flush()
            if bad:
                log(f"front: skipped {bad} malformed stream line(s) from {self.url}")
            if json_turn.plain:
                log(f"front: {self.url} answered in plain text, not the JSON turn; does it take response_format?")
            field = json_turn.fields.get("delegate", "").strip()
            for request in speech.delegations() if not field else [] if handed else [field]:  # or the notes instead
                await self._hand_off(request, turn, tools, announcing)
            if held is not None:  # handed off: an acknowledgement; a question kept: see held_answer
                whole = "".join(held) + said
                spoken = acknowledgement(whole) if turn.request else held_answer(whole, turn.worked)
                # a question's held reply, even a whole "Let me check.", told the user nothing of a waiting result
                turn.trimmed = not turn.request or spoken != whole.strip()
                words.put_nowait(spoken)
            elif said:
                words.put_nowait(said)
        finally:
            words.put_nowait(None)

    async def _hand_off(self, request: str, turn: FrontTurn, tools: Delegator, announcing: bool) -> None:
        if not is_request(request):
            return
        if announcing:  # snapshot: a cut result turn's reader may finish after the next turn began
            log(f"front: not delegated, the turn announces a result: {request[:120]!r}")
            return
        if not (await tools.call("delegate", {"request": request}, said=turn.said)).startswith(HANDED_OFF):
            return
        turn.request = f"{turn.request} {request}".strip()
        if turn.committed:  # cut, and its words already in history
            log(f"front: the turn was cut, but its delegation went out: {request!r}")
            self._amend(turn)

    async def aclose(self) -> None:
        for task in list(self._readers):
            task.cancel()
        await self.client.aclose()


# Haiku 4.5 and Sonnet 4.5 reject output_config.effort; the server-side fallback takes only the newest models.
_NO_EFFORT = re.compile(r"haiku-4-5|sonnet-4-5")
_FALLBACKS = re.compile(r"^claude-(fable-5-1|opus-5-5|opus-5|sonnet-5-5)$")


class AnthropicBrain(Brain):
    """Claude through the Anthropic SDK: streaming, a manual tool loop."""

    name = "anthropic"
    PROTOCOL = (
        "\nDelegate by calling the delegate tool with the complete request, and say in a few words that you are on "
        "it. Only the tool call hands work off: saying that you are on it, or writing a note about it, hands off "
        "nothing."
    )

    def __init__(self, system: str, model: str):
        super().__init__(system + self.PROTOCOL)
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
        for n in range(TOOL_ROUNDS):
            speech = SpeechFilter()
            async with self.client.beta.messages.stream(messages=msgs, **kw) as stream:
                async for ev in stream:
                    if ev.type == "text" and (said := speech.push(ev.text)):
                        yield said
                msg = await stream.get_final_message()
            if said := speech.flush():
                yield said
            if msg.stop_reason == "refusal":
                yield " I can't help with that one."
                return
            if msg.stop_reason == "max_tokens":
                return
            uses = [{"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
                    for b in msg.content if b.type == "tool_use"]
            if not uses:  # the note instead of the call: make the call
                for i, request in enumerate(speech.delegations()):
                    log(f"front: the model wrote a delegation note instead of calling delegate; delegating {request!r}")
                    uses.append({"type": "tool_use", "id": f"toolu_note_{n}_{i}", "name": "delegate",
                                 "input": {"request": request}})
            if not uses:
                return
            asked = {"role": "assistant", "content": uses}
            # eager input streaming leaves validation to us: Delegator checks the input's shape
            results = {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": u["id"],
                 "content": await tools.call(u["name"], u["input"], said=self.user_said)}
                for u in uses]}
            text = [{"type": "text", "text": b.text} for b in msg.content if b.type == "text" and b.text.strip()]
            msgs += [{"role": "assistant", "content": text + uses}, results]
            self.record(asked, results)

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
        self.question_open = False  # the last announced report asked the user something

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
        waiting = self.take_results() if self.brain is not None else []
        self.turn = asyncio.create_task(self.run_turn(text, waiting=waiting))

    def take_results(self) -> list[str]:
        """Every result waiting for the floor, oldest first. Once one is taken, work is in session."""
        texts: list[str] = []
        while not self.results.empty():
            texts.append(self.results.get_nowait())
        self.tools.results_seen = self.tools.results_seen or bool(texts)
        return texts

    async def interrupt(self) -> None:
        """Barge-in and POST /stop: cancel the running turn; it commits what was heard before it ends."""
        if self.turn_running():
            await cancel_and_wait(self.turn)
            self.settle()

    async def run_turn(self, user_text: str, is_event: bool = False, brain: Brain | None = None,
                       report: str = "", waiting: list[str] | None = None) -> None:
        """Front model -> sentences -> Voice, then the history gets exactly what was heard. An event turn's `report`
        is the result itself, read out when the front is down. A user turn's `waiting` results arrived while the user
        was talking: they go into the front's history just ahead of the user's words, so the reply rests on them and
        not on what the front last heard (seen live: "the worker is still running" while its result sat queued)."""
        brain = brain or self.brain
        if brain is None:
            return await self.front_down(report or user_text, is_event)
        self.turn_spoken = []
        failed: list[Exception] = []
        handed = self.tools.handed
        if waiting:
            log(f"front: {len(waiting)} waiting result(s) go into the user's turn, not a separate announcement")
            brain.history.extend({"role": "user", "content": event_message(report_brief(t), WAITING)} for t in waiting)

        async def sentences() -> AsyncIterator[str]:
            self.tools.announcing = is_event  # a result is reported, never acted on (set when the turn starts)
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
        cancelled = False
        try:
            interrupted = await self.voice.speak(sentences(), self.turn_spoken, lambda: self.set_state("speaking"))
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:
            said = brain.commit(" ".join(self.turn_spoken), interrupted)
            if brain is self.brain and said:
                emit(type="transcript", role="front", text=said)
            if not (is_event or failed) and brain is self.brain:  # also when a barge-in cut the reply: the user's
                self.pass_on(user_text, said, handed)  # words were heard whole, the reply maybe not
            # A barge-in cut the reply: unless two whole sentences of it were heard (most of a retelling), the
            # waiting results it took go back to the announcer, or to the user's next turn.
            whole = [s for s in self.turn_spoken if not s.endswith("...")]
            if cancelled and waiting and (brain.trimmed or len(whole) < 2):
                log(f"front: the reply was cut early; {len(waiting)} waiting result(s) go back to the announcer")
                self.requeue(waiting)
        if failed:
            e = failed[0]
            warn(f"front model failed at {brain.where} ({type(e).__name__}: {str(e)[:160]})")
            self.requeue(waiting or [])  # never heard: the announcer reads them out
            return await self.front_down(report or user_text, is_event)
        # A held reply (cut to HOLD or an acknowledgement) told the user nothing of the waiting results
        # (eval: a waiting report, then "what time is it in Tokyo?", then only "I've asked"): the announcer reads them.
        if waiting and brain.trimmed:
            log(f"front: the reply was held short; {len(waiting)} waiting result(s) go back to the announcer")
            self.requeue(waiting)
        self.settle()
        if not is_event and _GOODBYE.search(user_text.lower()):
            self.goodbye()

    def requeue(self, texts: list[str]) -> None:
        """Results back to the announcer, ahead of any that arrived since."""
        for t in texts + self.take_results():
            with contextlib.suppress(asyncio.QueueFull):
                self.results.put_nowait(t)

    def pass_on(self, user_text: str, said: str, handed: int) -> None:
        """A user turn that delegated nothing still reaches Claude, as a note: a confirmation ("it sounds perfect, the
        static is gone"), a correction or a decision is information Claude needs (seen live: said three times, never
        passed on, while Claude kept saying nobody had confirmed by ear). The mod hands two kinds to Claude as a
        prompt rather than a note: `answer`, a reply to a question the last report asked, and `ask`, a question the
        front did not delegate (Qwen3-4B, asked "so what was the fix?" with only a log report to go on, invented a
        cause and delegated nothing, even when told to delegate what the report does not answer); the front said
        only HOLD or, with no work in session, a short first sentence (held_answer). Claude may answer that with
        "(nothing to add)", which the mod does not pass back."""
        answer, self.question_open = self.question_open, False
        if self.tools.handed != handed:
            return
        kind = {"answer": True} if answer else {"ask": True} if "?" in user_text else {}
        emit(type="note", said=user_text, reply=said, **kind)

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
        being transcribed, no turn running). They are taken only then, so a user turn that starts first takes them
        instead (run_turn's `waiting`). Results that waited together are one announcement, oldest first. A result is
        taken once: an announcement cut by a barge-in is not announced again."""
        while True:
            while self.results.empty() or not self.floor_free():
                await asyncio.sleep(0.1)
            texts = self.take_results()
            if len(texts) > 1:
                log(f"front: {len(texts)} results waited together; announcing them as one")
            report = "\n\n".join(b for t in texts if retellable(b := report_brief(t)))[:EVENT_CHARS]
            if not report:  # "(nothing to add)", or nothing at all: a turn on it would only invent something to say
                log(f"front: {len(texts)} result(s) with nothing to say; not announced")
                continue
            self.question_open = "?" in report
            self.turn = asyncio.create_task(self.run_turn(event_message(report), is_event=True, report=report))
            await asyncio.wait({self.turn})

    async def shutdown(self) -> None:
        await cancel_and_wait(self.turn)
        if self.brain is not None:
            with contextlib.suppress(Exception):
                await self.brain.aclose()
