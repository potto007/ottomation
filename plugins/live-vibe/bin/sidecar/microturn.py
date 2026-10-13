"""The micro-turn front: the text front takes a turn every few hundred milliseconds and answers with words or with
`listen`, so it reads the user between words and decides itself when to speak (docs/design/microturn-front.md).

Increment 1: the window message, the two-request decision step with the listen-probability rule, the history commit
rule, and the per-step measurements the bench reports. The loop in the sidecar (increment 2) drives `MicroStepper`.

The rule (Codacus, from DuplexCascade): `listen` is one token, "keep talking" is many first words each with a small
share, so `listen` wins the argmax while most of the probability wants speech. Read P(listen) from the top logprobs
at the action slot; at or above THRESHOLD listen, below it block the `listen` token and the model must speak.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar

LISTEN = "listen"
SPEAK = "speak"
THRESHOLD = 0.30  # P(listen) at or above this listens; 0.50 made his model finish the user's sentences
TOP_N = 10  # logprobs read at the action slot; a token outside them counts as probability 0
PREFIX = '{"action": "'  # the assistant's reply so far when the action is decided
NO_VOICE = "[no voice]"
VOICE = "[voice]"
SPEAKING = "[speaking]"

MICRO_PROTOCOL = (
    f"\nYou are on a live call. Every few hundred milliseconds you get one message with what the microphone heard in "
    f"that window: {VOICE} followed by the words, or {NO_VOICE} when nothing new was heard. [p=0.8] is how likely it is "
    f"that the user has finished speaking; {SPEAKING} means your own voice is still playing. Answer with one JSON "
    f'object only: {{"action": "{LISTEN}" or "{SPEAK}", "delegate": "...", "say": "..."}}. {LISTEN} means you say '
    f"nothing now: the user is still talking, or you have nothing to add. {SPEAK} means say is spoken at once: a few "
    f"words that continue what you were saying, or start an answer, or a short listening sound (mm-hm) while the user "
    f"talks on. If the user talks over you, answer {LISTEN} and stop. delegate is the complete request for the coding "
    f'agent whenever the user asks for any work, otherwise "". When you hand work off, say so in a few words that name '
    f'the task ("Checking the repo status."); never hand off in silence.'
)
HANDOFF_SAY = "On it."  # spoken when a delegation goes out with no say of its own

FORMAT: dict[str, Any] = {"type": "json_schema", "json_schema": {
    "name": "micro_turn", "strict": True, "schema": {
        "type": "object",
        "properties": {"action": {"type": "string", "enum": [LISTEN, SPEAK]}, "delegate": {"type": "string"},
                       "say": {"type": "string"}},
        "required": ["action", "delegate", "say"], "additionalProperties": False}}}


def window_message(words: str, p: float | None, speaking: bool) -> str:
    """The user message for one window: the words heard in it (or NO_VOICE), the end-of-turn probability when the
    recognizer has one, and SPEAKING while the front's own voice plays."""
    words = " ".join(words.split())
    parts = [f"{VOICE} {words}" if words else NO_VOICE]
    if p is not None:
        parts.append(f"[p={min(1.0, max(0.0, p)):.1f}]")
    if speaking:
        parts.append(SPEAKING)
    return " ".join(parts)


def listen_probability(top: list[dict[str, Any]], listen_ids: set[int], speak_ids: set[int]) -> float | None:
    """P(listen) among the two actions, from llama-server's top logprobs at the action slot: p_l / (p_l + p_s) with a
    missing token at 0. None when neither token is in the list (the model wanted something the grammar forbids)."""
    p_l = p_s = 0.0
    seen: set[int] = set()  # the chosen token may be listed twice (in the top list and as the result)
    for entry in top:
        try:
            tid, lp = int(entry["id"]), float(entry["logprob"])
        except (KeyError, TypeError, ValueError):
            continue
        if tid in seen:
            continue
        seen.add(tid)
        if tid in listen_ids:
            p_l += math.exp(lp)
        elif tid in speak_ids:
            p_s += math.exp(lp)
    if p_l + p_s <= 0.0:
        return None
    return p_l / (p_l + p_s)


def decide(p_listen: float | None, threshold: float = THRESHOLD) -> str:
    """The rule: listen at or above the threshold, else speak. With no reading (None) the front stays quiet."""
    if p_listen is None:
        return LISTEN
    return LISTEN if p_listen >= threshold else SPEAK


def keep_in_history(window: str, action: str, say: str = "", delegate: str = "") -> bool:
    """Silent windows (nothing heard, nothing said, nothing handed off) are never committed: the history grows with
    the conversation, not the clock, the prompt cache keeps its prefix, and a run of `listen` replies never teaches
    the model to stay silent (the video's failure at 30 turns)."""
    heard = not window.startswith(NO_VOICE)
    return heard or action == SPEAK and bool(say.strip() or delegate.strip())


@dataclass
class MicroStep:
    window: str
    action: str = LISTEN
    p_listen: float | None = None
    decide_ms: float = 0.0  # the decision request, prompt template included
    speak_ms: float = 0.0  # the full turn when it spoke (first byte to last)
    say: str = ""
    delegate: str = ""
    spoken: str = ""  # what the floor rules let through to the voice (say after the session's trim rule)
    timings: dict[str, Any] = field(default_factory=dict)  # llama-server's timings of the decision request

    @property
    def total_ms(self) -> float:
        return self.decide_ms + self.speak_ms


class MicroStepper:
    """One micro-turn against llama-server: `/apply-template` renders the chat prompt, `/completion` with the PREFIX
    and `n_predict` 1 gives the action slot's logprobs, the rule decides, and a speaking step runs the JSON turn on
    `/v1/chat/completions` with the `listen` token blocked. The brain's system prompt and history are read, not
    written: the caller commits what keep_in_history says."""

    TOKENIZE_WORDS: ClassVar[tuple[str, str]] = (LISTEN, SPEAK)

    def __init__(self, client: Any, url: str, model: str, system: str, threshold: float = THRESHOLD,
                 sampling: dict[str, Any] | None = None):
        self.client, self.url, self.model = client, url.rstrip("/"), model or "default"
        self.system = system + MICRO_PROTOCOL
        self.threshold = threshold
        self.sampling = sampling or {}
        self.listen_ids: set[int] = set()
        self.speak_ids: set[int] = set()
        self.lora: list[dict[str, Any]] = []  # the server's adapters, applied to this loop's requests alone

    async def resolve_adapters(self) -> None:
        """The timing LoRA is for the loop's two requests only: on the full front turns (relay, retell) it regressed
        the relay eval from 45/51 to 39/51. The managed server sets every adapter's server-wide scale to 0 once
        healthy (FrontServer._scope_lora), and the loop names them at scale 1 per request. A server without the
        endpoint has none."""
        get = getattr(self.client, "get", None)
        if get is None:
            return
        try:
            r = await get(f"{self.url}/lora-adapters", timeout=10.0)
            adapters = r.json() if r.status_code == 200 else []
        except Exception:  # noqa: BLE001 - no endpoint, no adapters
            adapters = []
        self.lora = [{"id": int(a["id"]), "scale": 1.0} for a in adapters if isinstance(a, dict) and "id" in a]

    async def resolve_tokens(self) -> None:
        """The first token of each action word, as the model tokenizes it right after the opening quote."""
        await self.resolve_adapters()
        for word, ids in ((LISTEN, self.listen_ids), (SPEAK, self.speak_ids)):
            r = await self.client.post(f"{self.url}/tokenize", json={"content": word, "add_special": False},
                                       timeout=10.0)
            r.raise_for_status()
            toks = r.json()["tokens"]
            if toks:
                ids.add(int(toks[0]))
        if not self.listen_ids or not self.speak_ids or self.listen_ids & self.speak_ids:
            raise RuntimeError(f"action tokens not distinct: listen={self.listen_ids} speak={self.speak_ids}")

    def messages(self, history: list[dict[str, Any]], window: str) -> list[dict[str, Any]]:
        return [{"role": "system", "content": self.system}, *history, {"role": "user", "content": window}]

    async def prompt(self, messages: list[dict[str, Any]]) -> str:
        r = await self.client.post(f"{self.url}/apply-template", json={"messages": messages}, timeout=10.0)
        r.raise_for_status()
        return str(r.json()["prompt"])

    async def decide(self, history: list[dict[str, Any]], window: str) -> MicroStep:
        """The decision request alone: PREFIX appended to the templated prompt, one token, TOP_N logprobs."""
        if not self.listen_ids or not self.speak_ids:  # first step: the action tokens, from the server's tokenizer
            await self.resolve_tokens()
        step = MicroStep(window=window)
        t = time.monotonic()
        prompt = await self.prompt(self.messages(history, window))
        body = {"prompt": prompt + PREFIX, "n_predict": 1, "n_probs": TOP_N, "cache_prompt": True, "stream": False,
                "temperature": 0.0, "samplers": []}  # raw logits: no top_k/top_p before the probabilities
        if self.lora:
            body["lora"] = self.lora
        r = await self.client.post(f"{self.url}/completion", json=body, timeout=30.0)
        r.raise_for_status()
        data = r.json()
        step.decide_ms = 1000 * (time.monotonic() - t)
        step.timings = data.get("timings") or {}
        probs = data.get("completion_probabilities") or []
        top = (probs[0].get("top_logprobs") or []) if probs else []
        if probs and all("logprob" in e for e in top):
            top = [*top, {"id": probs[0].get("id"), "logprob": probs[0].get("logprob")}]  # the chosen token too
        step.p_listen = listen_probability(top, self.listen_ids, self.speak_ids)
        step.action = decide(step.p_listen, self.threshold)
        return step

    async def step(self, history: list[dict[str, Any]], window: str) -> MicroStep:
        """Decide, then speak if the rule says so: the full JSON turn with `listen` blocked, not streamed here (the
        sidecar loop streams it through the brain's reader in increment 2)."""
        step = await self.decide(history, window)
        if step.action != SPEAK:
            return step
        t = time.monotonic()
        body = {"model": self.model, "messages": self.messages(history, window), "response_format": FORMAT,
                "stream": False, "max_tokens": 80, **self.sampling,
                "chat_template_kwargs": {"enable_thinking": False},
                "logit_bias": {str(i): -100 for i in self.listen_ids}}
        if self.lora:
            body["lora"] = self.lora
        r = await self.client.post(f"{self.url}/v1/chat/completions", json=body, timeout=30.0)
        r.raise_for_status()
        step.speak_ms = 1000 * (time.monotonic() - t)
        content = r.json()["choices"][0]["message"]["content"] or ""
        step.say, step.delegate = fields_of(content)
        return step


EOT = 0.6  # the recognizer's end-of-turn probability at or above which the user is taken to have finished
IDLE_SKIP = 3  # silent windows after the last words or speech that still ask the model; then the loop idles
ECHO_RATIO = 0.8  # a say whose words are this much inside the user's utterance is an echo, not a reply


def is_echo(say: str, utterance: str) -> bool:
    """The front repeating the user's words back ("add a unit test for the parser." to "add a unit test for the
    parser"): most of say's words are in the utterance."""
    s = [w for w in _words(say)]
    if not s:
        return False
    u = set(_words(utterance))
    return sum(1 for w in s if w in u) / len(s) >= ECHO_RATIO


def _words(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9']+", text.lower())


@dataclass
class LoopDeps:
    """What the loop does to the world, injected so the loop runs in a unit check with none of it."""

    speak: Any  # (text) -> None: words to the voice; starts a speaking run or extends the one playing
    end_run: Any  # async () -> None: the run ends after what is queued has played
    cut: Any  # async () -> None: the voice stops now
    delegate: Any  # async (request, said) -> bool: handed off
    speaking: Any  # () -> bool: the voice is playing
    busy: Any  # () -> bool: an utterance-level turn holds the floor (an announcement): no step now
    show: Any  # (text, kind) -> None: a front line on screen
    note: Any  # (text, said) -> None: the user's utterance to Claude as a note, with what the front said to it, when
    # the loop handed nothing off for it
    is_backchannel: Any  # (text) -> bool
    log: Any  # (text) -> None
    trim: Any = staticmethod(lambda say, handing: say)  # (say, handing off) -> what is spoken: the session's
    # acknowledgement rule for a hand-off (a short "on it" with no claim about the work) and its reply rule otherwise


class MicroLoop:
    """The loop (design note, "The loop"), one tick per step_ms, driven by the session:

    - `on_piece(piece, p)` on every recognizer step: the user's words so far in this utterance and the latest
      end-of-turn probability. `on_utterance_end(text)` when the recognizer ends the utterance (its text replaces the
      pieces). The utterance is held as a pending note for Claude until the step after it resolves: a delegation
      drops the note (Claude gets the words with the request), anything else sends it (deps.note), as the
      utterance-level session does after a turn that handed nothing off.
    - `tick()` builds the window and asks the stepper. The window carries the whole utterance so far ([voice] and the
      words), not this step's fragment, so the model reads a sentence, not a word, and the history takes the
      utterance once when it ends. Floor rules before the model's choice stands: while the user is still talking (p
      below EOT and the utterance not ended), only a backchannel may be spoken; a say that echoes the user's words is
      dropped; a delegation is taken once the user has finished. A `listen` while the voice plays and the user has
      spoken is a cut; a `listen` with nothing heard lets the run finish.
    - A speaking run (consecutive speak steps) is one assistant message in the history, written when it ends with
      what was heard, so the history never fills with one-word speak turns (the bench's speak bias)."""

    def __init__(self, stepper: MicroStepper, history: list[dict[str, Any]], deps: LoopDeps, step_ms: int,
                 eot: float = EOT):
        self.stepper, self.history, self.deps, self.step_ms, self.eot = stepper, history, deps, step_ms, eot
        self.utterance: list[str] = []  # the user's words since the last utterance end
        self.last_utterance = ""  # the recognizer's text of the utterance that just ended (the echo check's reference)
        self.final: tuple[str, float] | None = None  # an ended utterance and its p, asked once as a whole before it joins the history
        self.holding = False  # the session holds an unfinished utterance for its rest: the user is still talking
        self.replied = False  # a line was spoken for the utterance that ended; no second run without new words
        self.pending_note: str | None = None  # an ended utterance not yet handed off or passed on
        self.new_words = False  # words arrived since the last tick
        self.p: float | None = None
        self.user_done = True  # the recognizer ended the utterance (or none is open)
        self.run: list[str] = []  # the say of the speaking run in progress
        self.run_delegate = ""
        self.handed_this_utterance = False
        self.in_flight = False
        self.silent = 0  # ticks since the last words or speech
        self.ticks = self.skipped = self.late = 0
        self.last: MicroStep | None = None

    # -- from the session ----------------------------------------------------------------------------------
    def on_piece(self, piece: str | None, p: float | None) -> None:
        if piece and piece.strip():
            if self.user_done:  # a new utterance begins: the last one's note, if still pending, goes now
                self.flush_note()
                self.commit_final()
                self.replied = False  # the one-reply rule is per utterance (sixth session: the first step on a new
                # question was dropped as the last one's second reply)
            self.utterance.append(piece.strip())
            self.new_words = True
            self.user_done = False
            self.silent = 0
        if p is not None:
            self.p = p

    def on_utterance_end(self, text: str) -> None:
        """The recognizer's utterance: the next window asks the model on its whole text (not the pieces) with the
        turn taken as ended, then that window joins the history as the user's message; it becomes the pending note,
        and the loop starts a fresh utterance."""
        text = " ".join(text.split())
        self.flush_note()
        self.commit_final()
        if text:
            self.final = (text, self.p if self.p is not None else 0.9)
            self.last_utterance = text
            self.pending_note = text
        self.utterance = []
        self.new_words = False
        self.user_done = True
        self.holding = False
        self.replied = False
        self.handed_this_utterance = False
        self.silent = 0

    def hold_open(self) -> None:
        """The recognizer ended an utterance that stops mid-sentence and the session waits for its rest: the words
        stay in the window and the user counts as still talking (decision only, no speak) until the whole sentence
        ends or the wait runs out."""
        self.holding = True
        self.user_done = False

    def commit_final(self) -> None:
        """The ended utterance joins the history once: after the window that asked on it, or when the user moves on
        before one did."""
        if self.final is not None:
            text, p = self.final
            self.history.append({"role": "user", "content": window_message(text, p, False)})
            self.final = None

    def flush_note(self, said: str = "") -> None:
        note, self.pending_note = self.pending_note, None
        if note:
            self.deps.note(note, said)

    def resolve_note(self, handed: bool, said: str = "") -> None:
        """The step after an utterance settled it: handed off (the words went with the request) or not (a note, with
        what the front said to it, so Claude knows whether the user already heard an answer)."""
        if handed:
            self.pending_note = None
        else:
            self.flush_note(said)

    def run_ended(self, heard: list[str], cut: bool) -> None:
        """The voice finished (or was cut): the run is one assistant message of what was heard."""
        said = " ".join(h for h in heard if h.strip()).strip()
        if said or self.run_delegate:
            self.history.append({"role": "assistant", "content": json_turn(SPEAK, self.run_delegate, said)})
            self.deps.show(said, "reply")
        self.run = []
        self.run_delegate = ""
        if cut:
            self.silent = 0

    # -- the tick ------------------------------------------------------------------------------------------
    @property
    def utterance_text(self) -> str:
        return " ".join(self.utterance)

    def user_talking(self) -> bool:
        if self.holding:
            return True
        return bool(self.utterance) and not self.user_done and (self.p is None or self.p < self.eot)

    def window(self) -> str:
        speaking = bool(self.deps.speaking())
        if self.utterance and not self.user_done:
            return window_message(self.utterance_text, self.p, speaking)
        if self.final is not None:  # the ended utterance, whole, with the turn taken as over
            return window_message(self.final[0], max(self.final[1], self.eot), speaking)
        return window_message("", self.p if self.utterance else None, speaking)

    def idle(self) -> bool:
        """Nothing to ask: no new words, the user is not mid-utterance, nothing plays, and the model has had
        IDLE_SKIP silent windows since."""
        return (not self.new_words and self.final is None and (self.user_done or not self.utterance)
                and not self.deps.speaking() and not self.run and self.silent >= IDLE_SKIP)

    def context(self) -> list[dict[str, Any]]:
        """The history the model sees: the committed messages, plus the speaking run in progress as a provisional
        assistant message, so a step continues what is being said instead of starting the answer again (the first
        live session looped on one sentence 192 times for want of this)."""
        if not self.run:
            return self.history
        return [*self.history, {"role": "assistant", "content": json_turn(SPEAK, self.run_delegate, " ".join(self.run))}]

    def floor(self, step: MicroStep) -> tuple[str, str]:
        """The action after the floor rules, and why."""
        if step.action != SPEAK:
            return LISTEN, "model"
        if self.handed_this_utterance:  # one line per hand-off; what follows is Claude's to say
            return LISTEN, "handed"
        if self.user_talking():
            if step.say and self.deps.is_backchannel(step.say):
                step.spoken = step.say.strip()
                return SPEAK, "backchannel"
            return LISTEN, "floor"
        if self.replied and not self.run:  # one reply per utterance: a new run needs new words from the user
            return LISTEN, "said"
        handing = bool(step.delegate.strip())
        say = str(self.deps.trim(step.say.strip(), handing)).strip() if step.say.strip() else ""
        step.spoken = say
        reference = self.utterance_text or self.last_utterance
        if say and reference and not handing and is_echo(say, reference):  # a hand-off names the task
            return LISTEN, "echo"
        if say and self.run and is_echo(say, " ".join(self.run)):  # the run said this already: it is over
            return LISTEN, "repeat"
        if not say and not handing:
            return LISTEN, "empty"
        return SPEAK, "model"

    async def tick(self) -> MicroStep | None:
        """One window. None when skipped (a step still in flight, the floor held by a turn, or idle)."""
        self.ticks += 1
        if self.in_flight or self.deps.busy():
            self.skipped += 1
            return None
        if self.idle():
            return None
        window = self.window()
        asked_final = self.final is not None and not (self.utterance and not self.user_done)
        history = self.context()
        self.new_words = False
        self.in_flight = True
        t = time.monotonic()
        try:
            if self.user_talking():  # the decision alone: a speak here could only be a backchannel, and the model's
                step = await self.stepper.decide(history, window)  # say is not known without the full turn
                if step.action == SPEAK:  # the model wants in while the user talks: hold it (no backchannel yet)
                    step.action = LISTEN
                    reason = "floor"
                else:
                    reason = "model"
            else:
                step = await self.stepper.step(history, window)
                step.action, reason = self.floor(step)
        finally:
            self.in_flight = False
        if asked_final:
            self.commit_final()
        ms = 1000 * (time.monotonic() - t)
        if ms > self.step_ms:
            self.late += 1
        self.last = step
        await self.act(step, reason)
        self.deps.log(f"micro step: {step.action} p_listen={'n/a' if step.p_listen is None else f'{step.p_listen:.2f}'} "
                      f"why={reason} ms={ms:.0f} window={window[:60]!r}" + (f" say={step.say[:60]!r}" if step.say else "")
                      + (f" spoken={step.spoken[:60]!r}" if step.spoken != step.say.strip() else ""))
        return step

    async def act(self, step: MicroStep, reason: str) -> None:
        if step.action == SPEAK:
            self.silent = 0
            handed = False
            if step.delegate.strip() and not self.handed_this_utterance and not self.user_talking():
                said = self.utterance_text or self.last_utterance
                self.handed_this_utterance = True  # one attempt per utterance, whatever the delegator says (a
                # refused duplicate had the loop re-delegate and re-announce on every step of the fifth session)
                if await self.deps.delegate(step.delegate.strip(), said):
                    handed = True
                    self.run_delegate = self.run_delegate or step.delegate.strip()
            say = step.spoken or (HANDOFF_SAY if handed else "")  # a hand-off is never silent
            if say:
                self.run.append(say)
                self.deps.speak(say)
                if self.user_done:
                    self.replied = True
            if self.user_done:
                self.resolve_note(handed, say)
            return
        self.silent += 1
        if self.user_done and reason != "floor":  # a listen after the user finished: nothing was handed off
            self.resolve_note(False, " ".join(self.run))
        if self.deps.speaking() and self.utterance and not self.user_done:  # the user talks over the voice
            await self.deps.cut()
        elif self.run and (not self.deps.speaking() or reason in ("model", "repeat", "handed")):  # over, or ends here
            await self.deps.end_run()


def json_turn(action: str, delegate: str, say: str) -> str:
    import json

    return json.dumps({"action": action, "delegate": delegate, "say": say}, ensure_ascii=False)


def fields_of(content: str) -> tuple[str, str]:
    """`say` and `delegate` from a finished JSON turn; empty when the server answered in plain text."""
    import json

    try:
        obj = json.loads(content)
    except ValueError:
        return "", ""
    if not isinstance(obj, dict):
        return "", ""
    return str(obj.get("say") or ""), str(obj.get("delegate") or "")
