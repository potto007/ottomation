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
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Callable, ClassVar

from .audio import SentenceSplitter, Tuning, Voice, split_all
from .protocol import Route, emit, file_log, log, warn
from .session import Duplex, cancel_and_wait

EVENT = "[task finished]"
INTERRUPTED = "(you were interrupted)"  # leads the next user message after a cut: user text, so never imitated
EVENT_CHARS = 6000  # a long report is cut for the small front model; Claude's full answer is on screen
HISTORY_MAX = 40  # ponytail: the front keeps its last 40 messages; summarize older ones if long sessions forget
HISTORY_KEEP = 20  # experimental: past HISTORY_MAX, one trim after a turn keeps about this many (a stable prefix)
HISTORY_HARD = 2 * HISTORY_MAX  # experimental: the per-turn trim's ceiling, should the block trim never run
TOKENS_MAX = 500  # experimental: the most of Claude's answer retold, and of its notes kept as silent context
CHARS_PER_TOKEN = 4  # the token estimate where no tokenizer answers
NOTES = "[agent notes]"  # experimental: a user message of the coding agent's background notes, never read out
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
    "not the user talking. Retell it to the user in one or two short sentences, starting from its first sentence; "
    "the user can read the full report on screen:\n"
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


# Experimental (L4): the latest report's Status line lets the front answer a status question itself.
_STATUS_ASK = ("If the user asks how things are going, where things stand, or whether something was received, say only "
               "'Let me check.'; the coding agent answers.\n")
FRONT_PROMPT_EXPERIMENTAL = FRONT_PROMPT.replace(_STATUS_ASK, (
    "If the user asks how things are going or where things stand, answer in one or two short sentences from the "
    f"latest {EVENT} report's status and the {NOTES} before it, keeping that status exactly; if you handed something "
    "off after that report, or no report has come, say only 'Let me check.'; the coding agent answers.\n")) + (
    f"\nA {EVENT} report starts with its status, 'Status: working', 'done', 'failed' or 'cancelled': say it in plain "
    "words (still running, done, failed, cancelled), never the word 'Status'. A user message starting with "
    f"{NOTES} holds the coding agent's background notes on its work: never read them out or retell them on their "
    "own; use them only to answer the user's questions.")


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


_STATUS_LINE = re.compile(r"^[ \t>*_]*status[*_]*\s*:\s*[*_`]*\s*(working|done|failed|cancell?ed)\b[*_`]*[.!]?[ \t]*(.*)$",
                          re.I | re.M)
RELAY_SPOKEN = 3  # the sentences after a Status line that are retold


@dataclass
class Relay:
    """Experimental (L4): Claude's answers split as GPT-Live splits commentary from thinking. `retold` is what the
    front retells: the last `Status:` line and up to RELAY_SPOKEN sentences after it, or, without one, report_brief.
    `spoken` is those sentences alone (read out when the front is down). `notes` is the rest of the answer, which
    goes into the front's history as silent context, so a later status question needs no Claude turn."""

    status: str
    retold: str
    spoken: str
    notes: str

    def messages(self, note: str) -> list[dict[str, Any]]:
        """The front's history entries for this relay: the notes, then the report to retell, with `note`."""
        out = [{"role": "user", "content": notes_message(self.notes)}] if self.notes else []
        return out + [{"role": "user", "content": event_message(self.retold, note)}]


def notes_message(notes: str) -> str:
    quoted = notes.replace('"', "'")
    return (f'{NOTES} "{quoted}"\n(Background from the coding agent, not said to the user. Never read it out; use it '
            "only to answer the user's later questions.)")


def split_status(text: str) -> tuple[str, str, str]:
    """(status, the sentences after the last Status line, the answer before it); status '' when there is none."""
    found = list(_STATUS_LINE.finditer(text))
    if not found:
        return "", "", text
    m = found[-1]
    status = m.group(1).lower().replace("canceled", "cancelled")
    spoken = " ".join(split_all(f"{m.group(2)}\n{text[m.end():]}".strip())[:RELAY_SPOKEN])
    return status, spoken, text[:m.start()].strip()


def relay_of(texts: list[str]) -> Relay:
    """Results announced together, oldest first, as one Relay (uncapped: see cap_tokens). The status is the newest
    one given."""
    statuses, retold, spoken, notes = [], [], [], []
    for t in texts:
        status, said, rest = split_status(t)
        if status:
            statuses.append(status)
            retold.append(f"Status: {status}. {said}".strip())
            spoken.append(said)
        else:
            brief = report_brief(t)
            retold.append(brief)
            spoken.append(brief)
            kept = set(re.split(r"\n\s*\n", brief))
            rest = "\n\n".join(p for p in (q.strip() for q in re.split(r"\n\s*\n", _HEADING.sub("", t)))
                               if p and p not in kept)
        if rest.strip():
            notes.append(rest.strip())
    return Relay(statuses[-1] if statuses else "", "\n\n".join(r for r in retold if r),
                 "\n\n".join(s for s in spoken if s), "\n\n".join(notes))


def estimate_tokens(text: str) -> int:
    return -(-len(text) // CHARS_PER_TOKEN)


def cap_tokens(text: str, tokens: int, limit: int = TOKENS_MAX) -> str:
    """`text` cut to about `limit` of its `tokens` tokens, back to a sentence end (or a word) inside the cut."""
    if tokens <= limit:
        return text
    cut = text[:int(len(text) * limit / tokens)]
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "), cut.rfind("\n"))
    if end > len(cut) // 2:
        return cut[:end + 1].rstrip()
    return cut.rsplit(None, 1)[0] if " " in cut else cut


_ASIDE = re.compile(r"\[[^\]]*\]|\([^)]*\)")


def retellable(report: str) -> bool:
    """A report with something to retell: not empty, not only bracketed asides such as Claude's "(nothing to add)"."""
    return bool(re.search(r"\w", _ASIDE.sub("", report)))


RETELL = ("(Retell that report to the user in one or two short sentences, with its status as written; never read it "
          "out whole. If it asks a question, end by asking the user that question; do not answer it.)")

RETELL_STATUS = ("(Retell that report to the user in one or two short sentences, saying its status in plain words, never "
                 "the word 'Status'; never read it out whole. If it asks a question, end by asking the user that "
                 "question; do not answer it.)")

RELAY_SENTENCES = 2  # what a retelling speaks at most, whatever the model writes, plus a closing question
RELAY_CHARS = 280


class RelayCap:
    """Holds a retelling to RELAY_SENTENCES sentences and RELAY_CHARS characters (seen live: Qwen3-4B read Claude's
    whole answer back, which the screen then showed twice). A question past the cap still ends it: a report's
    question is the user's to answer. A retelling with nothing speakable falls back to the report's first sentences."""

    def __init__(self, report: str) -> None:
        self.report = report
        self.kept: list[str] = []
        self.chars = 0
        self.dropped = 0
        self.question = ""

    def push(self, sentence: str) -> list[str]:
        """What of this sentence to speak: itself, or nothing once the cap is reached."""
        if not self.kept or (not self.dropped and len(self.kept) < RELAY_SENTENCES
                             and self.chars + len(sentence) <= RELAY_CHARS):
            self.kept.append(sentence)
            self.chars += len(sentence) + 1
            return [sentence]
        self.dropped += 1
        if sentence.rstrip().endswith("?"):
            self.question = sentence
        return []

    def finish(self) -> list[str]:
        """What to speak after the last sentence: the dropped closing question, or the fallback."""
        if self.dropped:
            log(f"front: the retelling ran past {RELAY_SENTENCES} sentences; {self.dropped} not spoken")
        if not self.kept:
            return self.fallback()
        return [self.question] if self.question else []

    def fallback(self) -> list[str]:
        out: list[str] = []
        chars = 0
        for s in split_all(self.report):
            if out and (len(out) >= RELAY_SENTENCES or chars + len(s) > RELAY_CHARS):
                break
            out.append(s)
            chars += len(s) + 1
        if out:
            log("front: the retelling had nothing to say; the report's first sentences are read instead")
        return out


FRAGMENT_HOLD_S = 2.5  # how long an unfinished utterance waits for the rest of the sentence
FRAGMENT_PIECES = 4  # at most this many pieces are merged into one utterance
_OPEN_END = re.compile(r"(?:[,;:\-\u2013\u2014]|\.\.\.|\u2026)\s*$")
_FUNCTION_END = re.compile(
    r"\b(?:a|an|the|and|or|but|so|to|of|in|on|at|for|with|from|by|about|into|than|then|which|who|whose|when|where|"
    r"if|because|as|is|are|was|were|be|been|do|does|did|have|has|had|will|would|could|should|can|need|needs|want|"
    r"wants|i|we|they|he|she|my|our|your|their|its|like|just|also|really|maybe|not|very)$", re.I)


def looks_unfinished(text: str) -> bool:
    """An utterance cut mid-sentence (the recognizer's pause cap ended it): it ends open ("so," "and...") or, with no
    closing punctuation, on a function word ("Also, we need"). Its rest is usually the next utterance."""
    t = text.strip()
    if not t:
        return False
    if _OPEN_END.search(t):
        return True
    return t[-1] not in ".!?\"')" and bool(_FUNCTION_END.search(t))


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
        self.status = ""  # experimental: the latest report's Status (working, done, failed, cancelled)
        self.status_handed = -1  # `handed` when that report came

    @property
    def status_fresh(self) -> bool:
        """Experimental: a Status report came and nothing was handed off since, so it says how the work stands."""
        return bool(self.status) and self.handed == self.status_handed

    def saw_status(self, status: str) -> None:
        if status:
            self.status, self.status_handed = status, self.handed

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
    experimental = False  # the experimental voice path: block trims (compact) and status answers

    def __init__(self, system: str = ""):
        self.system = system
        self.history: list[dict[str, Any]] = []
        self.pending: list[dict[str, Any]] = []  # this turn's tool calls and results, committed with the words
        self.interrupted = False  # the last turn was cut: the next user message says so
        self.user_said = ""  # the user's own words this turn, as heard

    @property
    def where(self) -> str:
        return self.name

    def framed(self, user_text: str) -> str:
        """The user message begin() would add for `user_text`, with no side effect."""
        if self.interrupted and not user_text.startswith(EVENT):  # a report is not the user: the flag waits for them
            return f"{INTERRUPTED} {user_text}"
        return user_text

    def begin(self, user_text: str) -> None:
        self.user_said = user_text
        framed = self.framed(user_text)
        if framed != user_text:
            self.interrupted = False
        self.pending = []
        self.history.append({"role": "user", "content": framed})
        # Experimental: compact() trims in one block after a turn; this per-turn trim is only its safety net.
        cap = HISTORY_HARD if self.experimental else HISTORY_MAX
        if len(self.history) > cap:  # whole turns go, so no tool result outlives its call
            del self.history[: len(self.history) - cap]
            while self.history and not is_turn_start(self.history[0]):
                del self.history[0]

    def compact(self) -> bool:
        """Experimental (L1): past HISTORY_MAX messages, drop the oldest in one block down to about HISTORY_KEEP,
        cutting only where a turn starts, so no tool result loses its call and no delegation loses its user message.
        Trimming a little every turn shifted everything after the system prompt and cost the server's prompt cache
        on every request (f_keep 0.13 to 0.16, 3200 to 4300 tokens prefilled again); once per ~20 messages it costs
        one prefill, which a warm request then pays while nobody talks. True when it trimmed."""
        if len(self.history) <= HISTORY_MAX:
            return False
        cut = len(self.history) - HISTORY_KEEP
        while cut < len(self.history) and not is_turn_start(self.history[cut]):
            cut += 1
        del self.history[:cut]
        return cut > 0

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

    async def warm_prefix(self) -> None:
        """Experimental: one 1-token request over the whole history, so the server caches the new prefix."""

    async def count_tokens(self, text: str) -> int:
        return estimate_tokens(text)

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


_STATUS_QUESTION = re.compile(
    r"\b(?:how(?:'s| is| are)? (?:it|things|we|that|everything|the \w+) (?:going|doing|coming along)|where (?:are we|do "
    r"things stand|things stand|is it at)|status|progress|any (?:news|update)|is it (?:done|finished|ready|still running)"
    r"|done yet|finished yet|still running)\b", re.I)


def is_status_question(text: str) -> bool:
    """A question about how the work stands ("how's it going?", "is it done yet?"), which a fresh Status report
    answers (experimental)."""
    return "?" in text and bool(_STATUS_QUESTION.search(text))


_GUESS = re.compile(r"\b(likely|probably|because|due to|seems?|must have)\b", re.I)


def status_reply(say: str) -> str:
    """Experimental: what a question speaks when the latest Status report is fresh (no delegation since): the
    reply's first two sentences, which rest on that report and the notes; HOLD for an empty reply or a guess."""
    first = " ".join(re.split(r"(?<=[.!?])\s+", say.strip())[:2]).strip()
    return HOLD if not first or _GUESS.search(first) else first


def timings_of(line: str) -> dict[str, Any] | None:
    """llama-server's `timings` object (prompt_n, cache_n, prompt_ms...) from a stream line that carries one."""
    try:
        t = json.loads(line[5:].strip()).get("timings")
    except (ValueError, AttributeError):
        return None
    return t if isinstance(t, dict) else None


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
    fresh_status: bool = False  # experimental: a Status report came after the last delegation (status_reply)
    status_answer: bool = False  # experimental: a question was answered from that status, not held
    first_token_at: float | None = None  # when the reply's first content arrived (a speculation's: maybe early)
    timings: dict[str, Any] | None = None  # llama-server's `timings` from the stream's last chunk


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

    def request_messages(self, user_text: str) -> list[dict[str, Any]]:
        """What respond() would send for `user_text` now, with no side effect: a speculation's request."""
        return [{"role": "system", "content": self.system}, *self.history,
                {"role": "user", "content": self.framed(user_text)}]

    def speculate(self, user_text: str) -> Prefetch:
        """Experimental (L6): starts this request on a provisional user text, reading the reply into a buffer only.
        Nothing is spoken and nothing is handed off until respond() takes it (see Prefetch)."""
        return Prefetch(self, self.request_messages(user_text), user_text)

    async def respond(self, user_text: str, tools: Delegator, prefetch: Prefetch | None = None) -> AsyncIterator[str]:
        """Speaks the reply's "say" as it streams. The reply is read by its own task, which hands off the delegate
        field even when this turn is cut first: the user may already have heard "on it". `prefetch` is a speculation
        on this same request, whose buffered reply is read in place of a new one."""
        self.begin(user_text)
        msgs: list[dict[str, Any]] = [{"role": "system", "content": self.system}, *self.history]
        turn = self.current
        words: asyncio.Queue[str | None] = asyncio.Queue()
        announcing = bool(getattr(tools, "announcing", False))
        turn.worked = bool(getattr(tools, "work_in_session", False))
        turn.fresh_status = (self.experimental and bool(getattr(tools, "status_fresh", False))
                             and is_status_question(user_text))
        reader = asyncio.create_task(self._read(msgs, turn, words, tools, announcing, prefetch))
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

    async def _lines(self, msgs: list[dict[str, Any]], max_tokens: int = 400) -> AsyncIterator[str]:
        """The reply's stream lines, from a new request."""
        async with self.client.stream("POST", f"{self.url}/v1/chat/completions",
                                      json=self._body(msgs, max_tokens, True)) as r:
            if r.status_code >= 400:
                await r.aread()
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:160]}")
            async for line in r.aiter_lines():
                yield line

    async def _read(self, msgs: list[dict[str, Any]], turn: FrontTurn, words: asyncio.Queue[str | None],
                    tools: Delegator, announcing: bool = False, prefetch: Prefetch | None = None) -> None:
        # raw -> no think block (one can lead the JSON) -> the "say" field -> no notes or think blocks -> speech
        think, json_turn, speech, bad = SpeechFilter(notes=False), TurnStream(), SpeechFilter(), 0
        held: list[str] | None = None  # a delegating turn's say, kept back whole and cut to an acknowledgement
        spoke = handed = False
        source = prefetch.lines() if prefetch is not None else self._lines(msgs)
        try:
            async with contextlib.aclosing(source) as lines:
                async for line in lines:
                    if '"timings"' in line:
                        turn.timings = timings_of(line) or turn.timings
                    kind, delta = parse_sse(line)
                    if kind == "done":
                        break
                    if kind == "bad":
                        bad += 1
                    text = (delta or {}).get("content")
                    if isinstance(text, str) and text:
                        if turn.first_token_at is None:
                            turn.first_token_at = (prefetch.first_token_at if prefetch is not None else None) \
                                or time.monotonic()
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
                if turn.request:
                    spoken = acknowledgement(whole)
                elif turn.fresh_status:  # experimental: the latest Status report answers it
                    spoken = status_reply(whole)
                    turn.status_answer = spoken != HOLD
                else:
                    spoken = held_answer(whole, turn.worked)
                # a question's held reply, even a whole "Let me check.", told the user nothing of a waiting result
                turn.trimmed = (spoken != whole.strip() if turn.status_answer else
                                not turn.request or spoken != whole.strip())
                words.put_nowait(spoken)
            elif said:
                words.put_nowait(said)
        finally:
            if prefetch is not None:
                prefetch.cancel()
            words.put_nowait(None)

    async def warm_prefix(self) -> None:
        msgs = [{"role": "system", "content": self.system}, *self.history]
        async with contextlib.aclosing(self._lines(msgs, 1)) as lines:  # streamed: a cancel reaches the server
            async for line in lines:
                if parse_sse(line)[0] == "done":
                    break

    async def count_tokens(self, text: str) -> int:
        """llama-server's own count (POST /tokenize); the estimate when it does not answer."""
        try:
            r = await self.client.post(f"{self.url}/tokenize", json={"content": text}, timeout=5.0)
            r.raise_for_status()
            return len(r.json()["tokens"])
        except Exception:  # noqa: BLE001 - another OpenAI-compatible server may have no /tokenize
            return estimate_tokens(text)

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


class Prefetch:
    """Experimental (L6): a front request started before the end of the user's turn, on its provisional text. Its
    stream lines go into a buffer and nothing reads them: no speech, no delegation. respond() takes it when the
    final text and the history match (the request is then exactly the one it would make) and reads the buffered
    lines, then the rest as they come; otherwise it is cancelled, which closes the request so the server stops."""

    def __init__(self, brain: LlamaCppBrain, msgs: list[dict[str, Any]], text: str):
        self.text = text
        self.key = json.dumps(msgs, sort_keys=True)  # the request: respond() takes it only when its own matches
        self.buf: list[str] = []
        self.done = False
        self.error: BaseException | None = None
        self.started = time.monotonic()
        self.first_token_at: float | None = None
        self._more = asyncio.Event()
        self._task = asyncio.ensure_future(self._fetch(brain, msgs))

    async def _fetch(self, brain: LlamaCppBrain, msgs: list[dict[str, Any]]) -> None:
        try:
            async with contextlib.aclosing(brain._lines(msgs)) as lines:
                async for line in lines:
                    if self.first_token_at is None and '"content"' in line:
                        self.first_token_at = time.monotonic()
                    self.buf.append(line)
                    self._more.set()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - raised to the turn that takes it
            self.error = e
        finally:
            self.done = True
            self._more.set()

    def matches(self, brain: LlamaCppBrain, text: str) -> bool:
        return text == self.text and json.dumps(brain.request_messages(text), sort_keys=True) == self.key

    async def lines(self) -> AsyncIterator[str]:
        i = 0
        while True:
            while i < len(self.buf):
                i += 1
                yield self.buf[i - 1]
            if self.done:
                if self.error is not None:
                    raise self.error
                return
            self._more.clear()
            if i == len(self.buf) and not self.done:
                await self._more.wait()

    def cancel(self) -> None:
        if not self._task.done():
            self._task.cancel()

    def ran_ms(self) -> float:
        return 1000 * (time.monotonic() - self.started)


class _HeadTap:
    """KyutaiSTT as KyutaiTurns uses it, keeping each step's word piece and pause heads for TurnWatch."""

    def __init__(self, stt: Any):
        self._stt = stt
        self.last: tuple[str | None, Any] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stt, name)

    def step(self, block: Any) -> tuple[str | None, Any]:
        piece, pauses = self._stt.step(block)
        self.last = (piece, pauses)
        return piece, pauses


class TurnWatch:
    """Experimental: the recognizer's detector, watched on the listener thread for what the front needs and speech.py
    does not send. Kyutai only (a detector with `stt` and `text`); anything else passes through unchanged.
    - ("eot_likely", text): the first step of a turn on which the 2 s pause head, or its EMA (p, kept as KyutaiTurns
      keeps it), is above LIKELY with no new word piece: the end of the turn is likely, so the front may speculate
      (L6). Once per text; ("eot_retract", None) when a word piece arrives after it. The 2 s head alone, not the
      0.5 s head that confirms a semantic end: with the STT flush on, that confirmation ends the turn within the
      same step, so only the 2 s head's earlier rise (and the whole continuous wait of a non-semantic end) leaves
      time to hide the front's reply in.
    - ("listening_pause", seconds into the turn): a pause with more speech coming during long dictation (over
      BACKCHANNEL_AFTER_S into the turn, the 0.5 s head above 0.8 and p below 0.4), at most once a second; the
      session decides whether to say "mm-hm" (L7).
    It reads only KyutaiTurns' public state (`active`, `text`) and the heads of each step, through _HeadTap."""

    LIKELY = 0.6  # speech.END_OF_TURN
    EMA_KEEP = 0.5  # speech.S2_EMA_KEEP
    IGNORE_STEPS = 12  # speech.HEADS_IGNORE_STEPS: the heads jitter after a model reset
    MS_PER_STEP = 80.0
    PAUSE_S05 = 0.8
    PAUSE_P = 0.4
    BACKCHANNEL_AFTER_S = 6.0

    def __init__(self, inner: Any, backchannels: bool = False):
        self.inner, self.backchannels = inner, backchannels
        self.tap: _HeadTap | None = None
        if getattr(inner, "stt", None) is not None and hasattr(inner, "text"):
            self.tap = _HeadTap(inner.stt)
            inner.stt = self.tap
        self._reset()

    def _reset(self) -> None:
        self.likely: str | None = None
        self.p: float | None = None
        self.steps = 0  # steps since this turn's speech_start
        self.cue_at = -1_000

    @property
    def active(self) -> bool:
        return bool(self.inner.active)

    def feed(self, frame: Any) -> list[tuple[str, Any]]:
        events = self.inner.feed(frame)
        if self.tap is None:
            return events
        last, self.tap.last = self.tap.last, None
        if any(k == "speech_start" for k, _ in events):
            self._reset()
        if any(k in ("utterance", "discard") for k, _ in events):
            self._reset()
            return events
        if last is None or not self.inner.active:
            return events
        piece, pauses = last
        self.steps += 1
        if getattr(self.tap, "steps", 0) <= self.IGNORE_STEPS:
            return events
        s2, s05 = float(getattr(pauses, "s2", 0.0)), float(getattr(pauses, "s05", 0.0))
        self.p = s2 if self.p is None else self.EMA_KEEP * self.p + (1 - self.EMA_KEEP) * s2
        if piece is not None:
            if self.likely is not None:
                self.likely = None
                events.append(("eot_retract", None))
        elif self.likely is None and max(s2, self.p) > self.LIKELY and (text := str(self.inner.text).strip()):
            self.likely = text
            events.append(("eot_likely", text))
        seconds = self.steps * self.MS_PER_STEP / 1000
        if (self.backchannels and seconds > self.BACKCHANNEL_AFTER_S and s05 > self.PAUSE_S05
                and self.p < self.PAUSE_P and self.steps - self.cue_at >= 1000 / self.MS_PER_STEP):
            self.cue_at = self.steps
            events.append(("listening_pause", round(seconds, 2)))
        return events


def make_brain(backend: str, url: str, model: str, experimental: bool = False) -> Brain | None:
    system = (FRONT_PROMPT_EXPERIMENTAL if experimental else FRONT_PROMPT).format(workspace=Path.cwd())
    try:
        brain: Brain = AnthropicBrain(system, model) if backend == "anthropic" else LlamaCppBrain(system, url, model)
        brain.experimental = experimental
        return brain
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
UPTO_WORDS = 12  # the most of a cut retelling Claude is told: where it stopped, not what it said


def spoken_upto(heard: str) -> str:
    """The last UPTO_WORDS words heard of a cut reply, without its '...': '' when nothing was heard."""
    words = heard.strip().removesuffix("...").split()
    return " ".join(words[-UPTO_WORDS:])


BACKCHANNEL_TEXT = "Mm-hmm."  # the front's listening sound, rendered once by the synthesizer (L7)
BACKCHANNEL_EVERY_S = 8.0  # at most one backchannel this often
BACKCHANNEL_AFTER_S = 6.0  # never in a turn's first seconds: only during long dictation


def _ms(start: float | None, end: float | None) -> str:
    return "n/a" if start is None or end is None else f"{max(0.0, 1000 * (end - start)):.0f}"


# -- the session ------------------------------------------------------------------------------------------
class FrontSession(Duplex):
    def __init__(self, voice: Voice, hearing: Callable[[], bool], brain: Brain | None,
                 switch: re.Pattern[str] | None, goodbye: Callable[[], None], tune: Tuning | None = None):
        super().__init__(voice, hearing)
        self.brain, self.switch, self.goodbye = brain, switch, goodbye
        tune = tune or Tuning()
        self.experimental = tune.experimental  # off: every experimental path below stays off (0.6.4)
        self.backchannels = tune.experimental and tune.backchannels
        self.tools = Delegator()
        self.results: asyncio.Queue[str] = asyncio.Queue(maxsize=RESULTS_MAX)
        self.turn: asyncio.Task | None = None
        self.turn_spoken: list[str] = []
        self.question_open = False  # the last announced report asked the user something
        self.fragment: list[str] = []  # an unfinished utterance held for its rest (looks_unfinished)
        self.fragment_due: asyncio.Task | None = None
        self.fragment_gen = 0
        self.eot_at: float | None = None  # when the last utterance arrived: the end of the user's turn
        self.spec: Prefetch | None = None  # experimental: a front reply started on the provisional text (L6)
        self.warm: asyncio.Task | None = None  # experimental: the warm request after a trim (L1)
        self.asked = False  # the front's last reply ended on a question: no backchannel over its answer
        self.backchannel_clip: Any = None  # the rendered BACKCHANNEL_TEXT
        self.backchannel_at = -1e9  # loop time of the last one
        self.backchannel_stop = threading.Event()

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
        return not (self.user_talking or self.hearing() or self.turn_running() or self.fragment)

    async def handle(self, kind: str, payload: Any) -> None:
        await self._handle(kind, payload)
        if kind in ("utterance", "discard") and self.spec is not None:  # the turn ended, and nothing took it
            self.drop_spec("discarded")

    async def _handle(self, kind: str, payload: Any) -> None:
        if kind == "speech_start":
            self.user_talking = True
            if self.warm is not None and not self.warm.done():  # the floor is the user's: the server too
                self.warm.cancel()
                file_log("INFO", "front warm: cancelled, the user spoke")
            if self.fragment_due is not None:  # the rest of a held utterance is coming: wait for it
                self.fragment_due.cancel()
            if not self.barge_in():  # over the voice: paused until the words say backchannel or cut
                await self.interrupt()
            self.set_state("user_speaking")
        elif kind == "barge_due":
            await self.barge_due(payload)
        elif kind == "transcribing":
            self.set_state("transcribing")
        elif kind == "discard":
            self.user_talking = False
            await self.barge_verdict(payload, discarded=True)
            if self.fragment:
                self.arm_fragment()
            self.settle()
        elif kind == "eot_likely":
            self.speculate(payload)
        elif kind == "eot_retract":
            if self.spec is not None:
                self.drop_spec("cancelled")
        elif kind == "listening_pause":
            await self.backchannel_cue(payload)
        elif kind == "utterance":
            self.user_talking = False
            self.eot_at = time.monotonic()
            if await self.barge_verdict(payload):
                emit(type="transcript", role="user", text=payload)
                self.backchannel(payload)
                self.settle()
                return
            await self.on_utterance(payload)
        elif kind == "fragment_due" and payload == self.fragment_gen and self.fragment and not self.user_talking:
            text, self.fragment = " ".join(self.fragment), []
            log(f"front: nothing followed the unfinished utterance; sending it as it is: {text[:80]!r}")
            await self.dispatch(text)

    def arm_fragment(self) -> None:
        """(Re)starts the wait for a held utterance's rest; when it runs out, the held words go on as they are."""
        if self.fragment_due is not None:
            self.fragment_due.cancel()
        self.fragment_gen += 1
        gen = self.fragment_gen

        async def due() -> None:
            await asyncio.sleep(FRAGMENT_HOLD_S)
            self._put(("fragment_due", gen))

        self.fragment_due = asyncio.ensure_future(due())

    async def on_utterance(self, text: str) -> None:
        """An utterance that stops mid-sentence (the recognizer's pause cap cut it: "Also, we need") is held up to
        FRAGMENT_HOLD_S for its rest, and the pieces go on as one: the front never reads half a sentence as a task,
        and the screen shows the user's words once, whole."""
        pieces = [*self.fragment, text]
        self.fragment = []
        if self.fragment_due is not None:
            self.fragment_due.cancel()
            self.fragment_due = None
        text = " ".join(p.strip() for p in pieces if p.strip())
        model = spoken_model(self.switch, text)
        if model:
            emit(type="transcript", role="user", text=text)
            emit(type="switch_model", model=model)  # the mod runs /model and posts the result as an event
            self.settle()
            return
        if len(pieces) < FRAGMENT_PIECES and looks_unfinished(text) and not self.turn_running():
            self.fragment = [p for p in pieces if p.strip()]
            self.arm_fragment()
            log(f"front: holding an unfinished utterance up to {FRAGMENT_HOLD_S:.1f} s for its rest: {text[:80]!r}")
            self.settle()
            return
        if len(pieces) > 1:
            log(f"front: {len(pieces)} pieces of one sentence go on as one utterance")
        await self.dispatch(text)

    async def dispatch(self, text: str) -> None:
        """The user's whole utterance: on screen, then a front turn."""
        emit(type="transcript", role="user", text=text)
        await self.interrupt()
        self.backchannel_stop.set()
        self.asked = False
        spec, self.spec = self.spec, None
        eot, self.eot_at = self.eot_at, None
        waiting = self.take_results() if self.brain is not None else []
        self.turn = asyncio.create_task(self.run_turn(text, waiting=waiting, prefetch=spec, eot=eot))

    # -- experimental: speculation (L6), the warm prefix (L1), backchannels (L7) ------------------------------------
    def speculate(self, text: str) -> None:
        """The end of the user's turn is likely (TurnWatch): start the front's reply on the words so far, as the
        whole utterance would go (any held fragment first). Not while anything else holds the floor or the server,
        nor for what would not become a front turn (a model switch, an unfinished sentence). Never a Claude turn:
        only the llama.cpp front speculates, and a speculation hands nothing off until the final text takes it."""
        if not self.experimental or not isinstance(self.brain, LlamaCppBrain):
            return
        if self.spec is not None:
            self.drop_spec("cancelled")
        full = " ".join(p.strip() for p in [*self.fragment, text] if p.strip())
        if (not full or self.turn_running() or self.pause is not None or not self.results.empty()
                or spoken_model(self.switch, full) or looks_unfinished(full)):
            return
        self.spec = self.brain.speculate(full)

    def drop_spec(self, why: str) -> None:
        spec, self.spec = self.spec, None
        if spec is None:
            return
        spec.cancel()
        file_log("INFO", f"front speculate: {why} saved_ms=0 ran_ms={spec.ran_ms():.0f}")

    def warm_prefix(self, brain: Brain) -> None:
        """After a block trim (Brain.compact), one 1-token request over the new history while nobody talks, so the
        next turn's prefill is cached. Skipped while the user speaks; cancelled when they start."""
        if self.user_talking or self.hearing():
            file_log("INFO", "front warm: skipped, the user is speaking")
            return

        async def warm() -> None:
            t = time.monotonic()
            try:
                await brain.warm_prefix()
                file_log("INFO", f"front warm: {len(brain.history)} messages in {1000 * (time.monotonic() - t):.0f} ms")
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - only the next turn's prefill is at stake
                file_log("INFO", f"front warm: failed ({type(e).__name__}: {str(e)[:120]})")

        self.warm = asyncio.ensure_future(warm())

    async def render_backchannel(self) -> None:
        try:
            self.backchannel_clip = await asyncio.to_thread(self.voice.tts.synth, BACKCHANNEL_TEXT)
        except Exception as e:  # noqa: BLE001 - no backchannels, nothing else lost
            log(f"front: the backchannel clip did not render ({type(e).__name__}: {str(e)[:120]}); none are played")

    async def backchannel_cue(self, seconds: Any) -> None:
        """TurnWatch heard a pause with more speech coming, `seconds` into the user's turn: a short "mm-hm", at most
        one per BACKCHANNEL_EVERY_S, never in a turn's first seconds, never over the front's own voice or a turn in
        flight, and never while the user answers a question the front just asked."""
        now = self.loop.time()
        if not (self.backchannels and self.backchannel_clip is not None and self.user_talking):
            return
        if (float(seconds) <= BACKCHANNEL_AFTER_S or now - self.backchannel_at < BACKCHANNEL_EVERY_S or self.asked
                or self.turn_running() or self.voice.speaking.is_set() or self.pause is not None):
            return
        self.backchannel_at = now
        self.backchannel_stop = threading.Event()
        played = await self.voice.blip(BACKCHANNEL_TEXT, self.backchannel_clip, self.backchannel_stop)
        file_log("INFO", f"front backchannel: {'played' if played else 'skipped'} at_s={float(seconds):.1f}")

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

    async def cut(self) -> None:
        await self.interrupt()

    async def relay(self, texts: list[str], brain: Brain | None) -> Relay:
        """Experimental (L4): results as one Relay, its retold part and notes each held to TOKENS_MAX tokens (the
        front server's own count when it gives one; a token is at least a character, so short text needs none)."""
        r = relay_of(texts)
        for field in ("retold", "notes"):
            text = getattr(r, field)
            if len(text) > TOKENS_MAX:
                n = await brain.count_tokens(text) if brain is not None else estimate_tokens(text)
                setattr(r, field, cap_tokens(text, n))
        self.tools.saw_status(r.status)
        return r

    async def run_turn(self, user_text: str, is_event: bool = False, brain: Brain | None = None,
                       report: str = "", waiting: list[str] | None = None, prefetch: Prefetch | None = None,
                       eot: float | None = None) -> None:
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
            if self.experimental:
                brain.history.extend((await self.relay(waiting, brain)).messages(WAITING))
            else:
                brain.history.extend({"role": "user", "content": event_message(report_brief(t), WAITING)}
                                     for t in waiting)
        if prefetch is not None:
            if isinstance(brain, LlamaCppBrain) and prefetch.matches(brain, user_text):
                file_log("INFO", f"front speculate: used saved_ms={prefetch.ran_ms():.0f}")
            else:
                prefetch.cancel()
                file_log("INFO", f"front speculate: discarded saved_ms=0 ran_ms={prefetch.ran_ms():.0f}")
                prefetch = None
        started = time.monotonic()
        audio_at: list[float] = []

        def on_play() -> None:
            if not audio_at:
                audio_at.append(time.monotonic())
            self.set_state("speaking")

        words: asyncio.Queue[str | None] = asyncio.Queue()
        written: list[str] = []
        shown = False

        def show(text: str) -> None:
            nonlocal shown
            shown = True
            if brain is self.brain and text:
                emit(type="transcript", role="front", text=text, kind=self.front_kind(text, is_event, handed))

        async def produce() -> None:
            """Reads the model to the end of its reply on a task of its own, so the reply is on screen as soon as it
            is written while the voice still reads it out (speak() pulls one sentence ahead of playback)."""
            self.tools.announcing = is_event  # a result is reported, never acted on (set when the turn starts)
            splitter = SentenceSplitter()
            cap = RelayCap(report) if is_event else None

            def put(sentences: list[str]) -> None:
                for s in sentences:
                    for t in cap.push(s) if cap else [s]:
                        written.append(t)
                        words.put_nowait(t)

            try:
                try:
                    extra = {"prefetch": prefetch} if prefetch is not None else {}
                    async with contextlib.aclosing(brain.respond(user_text, self.tools, **extra)) as deltas:
                        async for delta in deltas:
                            put(splitter.push(delta))
                except Exception as e:  # noqa: BLE001 - connect error, 5xx, a broken stream: the turn ends cleanly
                    failed.append(e)
                put(splitter.flush())
                if cap is not None and not failed:
                    for t in cap.finish():
                        written.append(t)
                        words.put_nowait(t)
                if not failed:
                    show(" ".join(written))
            finally:
                words.put_nowait(None)

        async def sentences() -> AsyncIterator[str]:
            while (s := await words.get()) is not None:
                yield s

        self.set_state("thinking")
        producer = asyncio.create_task(produce())
        interrupted = True
        cancelled = False
        try:
            interrupted = await self.voice.speak(sentences(), self.turn_spoken, on_play)
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:
            if not producer.done():  # cut while the model still writes: closing its reply hands a cut turn on
                await cancel_and_wait(producer)
            said = brain.commit(" ".join(self.turn_spoken), interrupted)
            if not shown and said:  # cut before the reply was whole, or failed: what was heard
                show(said)
            if isinstance(brain, LlamaCppBrain):
                self.log_turn(brain.current, eot if not is_event and eot is not None else started,
                              audio_at[0] if audio_at else None, is_event)
            if brain is self.brain:
                self.asked = said.rstrip().endswith("?")
            if self.experimental and brain is self.brain and brain.compact():
                file_log("INFO", f"front history: trimmed in one block to {len(brain.history)} messages")
                self.warm_prefix(brain)
            if is_event and interrupted:  # Claude's answer was retold only in part: Claude hears where it stopped
                emit(type="cut", heard=spoken_upto(said))
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

    @staticmethod
    def log_turn(turn: FrontTurn, eot: float, audio_at: float | None, is_event: bool) -> None:
        """One sidecar-log line per front turn: from the end of the user's turn (or an announcement's start) to the
        first token and the first audio, and what llama-server's timings say of its prompt cache: prompt_n is what
        it prefilled, cache_n what it reused. f_keep is only in the server's own log; n/a unless a server sends it."""
        t = turn.timings or {}
        f_keep = f"{float(t['f_keep']):.2f}" if isinstance(t.get("f_keep"), (int, float)) else "n/a"
        file_log("INFO", f"front turn: eot_to_first_token_ms={_ms(eot, turn.first_token_at)} "
                         f"first_audio_ms={_ms(eot, audio_at)} f_keep={f_keep} "
                         f"prefill_tokens={t.get('prompt_n', 'n/a')} cached_tokens={t.get('cache_n', 'n/a')} "
                         f"kind={'event' if is_event else 'user'}")

    def front_kind(self, text: str, is_event: bool, handed: int) -> str:
        """What a front line is, for the mod's display: `relay` retells a report of Claude's (shown above it), `filler`
        is an acknowledgement or a holding line (Claude answers next), `reply` is the front's own answer."""
        if is_event:
            return "relay"
        if self.tools.handed != handed or text.strip() in (HOLD, ACK):
            return "filler"
        return "reply"

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
        from_status = bool(getattr(getattr(self.brain, "current", None), "status_answer", False))
        # experimental: a status question the latest Status report answered needs no Claude turn
        kind = {"answer": True} if answer else {"ask": True} if "?" in user_text and not from_status else {}
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
        if self.backchannels:
            self.spawn(self.render_backchannel())
        while True:
            while self.results.empty() or not self.floor_free():
                await asyncio.sleep(0.1)
            texts = self.take_results()
            if len(texts) > 1:
                log(f"front: {len(texts)} results waited together; announcing them as one")
            if self.experimental:
                kept = [t for t in texts if retellable(report_brief(t))]
                r = await self.relay(kept, self.brain) if kept else None
                if r is None or not r.retold:
                    log(f"front: {len(texts)} result(s) with nothing to say; not announced")
                    continue
                if self.brain is not None and r.notes:
                    self.brain.history.append({"role": "user", "content": notes_message(r.notes)})
                self.question_open = "?" in r.spoken
                message = event_message(r.retold, RETELL_STATUS if r.status else RETELL)
                self.turn = asyncio.create_task(self.run_turn(message, is_event=True, report=r.spoken or r.retold))
                await asyncio.wait({self.turn})
                continue
            report = "\n\n".join(b for t in texts if retellable(b := report_brief(t)))[:EVENT_CHARS]
            if not report:  # "(nothing to add)", or nothing at all: a turn on it would only invent something to say
                log(f"front: {len(texts)} result(s) with nothing to say; not announced")
                continue
            self.question_open = "?" in report
            self.turn = asyncio.create_task(self.run_turn(event_message(report), is_event=True, report=report))
            await asyncio.wait({self.turn})

    async def shutdown(self) -> None:
        if self.fragment_due is not None:
            self.fragment_due.cancel()
        self.drop_spec("cancelled")
        if self.warm is not None:
            self.warm.cancel()
        await cancel_and_wait(self.turn)
        if self.brain is not None:
            with contextlib.suppress(Exception):
                await self.brain.aclose()
