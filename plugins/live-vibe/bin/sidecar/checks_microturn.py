"""--unit checks for the micro-turn front's pure parts (microturn.py): the window message, the listen-probability
rule, the history commit rule, and MicroStepper against a scripted server (no model)."""
from __future__ import annotations

import asyncio
import json
import math
from typing import Any

from .microturn import (
    FORMAT,
    HANDOFF_SAY,
    IDLE_SKIP,
    LISTEN,
    NO_VOICE,
    PREFIX,
    SPEAK,
    SPEAKING,
    THRESHOLD,
    LoopDeps,
    MicroLoop,
    MicroStep,
    MicroStepper,
    decide,
    fields_of,
    is_echo,
    keep_in_history,
    listen_probability,
    window_message,
)


class _Reply:
    def __init__(self, data: Any, status: int = 200):
        self._data, self.status_code = data, status

    def json(self) -> Any:
        return self._data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class ScriptedServer:
    """Answers /tokenize, /apply-template, /completion and /v1/chat/completions like llama-server, from a script of
    action-slot logprobs; records every request body."""

    LISTEN_ID, SPEAK_ID = 701, 702

    def __init__(self, p_listen: list[float], say: str = "Sure, on it.", delegate: str = ""):
        self.p = list(p_listen)
        self.say, self.delegate = say, delegate
        self.posts: list[tuple[str, dict[str, Any]]] = []

    async def post(self, url: str, json: dict[str, Any], timeout: float = 0.0) -> _Reply:
        path = url.split("/", 3)[-1]
        self.posts.append((path, json))
        if path == "tokenize":
            word = json["content"]
            return _Reply({"tokens": [self.LISTEN_ID if word == LISTEN else self.SPEAK_ID]})
        if path == "apply-template":
            return _Reply({"prompt": "<rendered>" + json["messages"][-1]["content"]})
        if path == "completion":
            p = self.p.pop(0) if self.p else 0.5
            chosen = self.LISTEN_ID if p >= 0.5 else self.SPEAK_ID
            top = [{"id": self.LISTEN_ID, "logprob": math.log(max(p, 1e-9)) - 0.2},  # 0.2: other mass exists
                   {"id": self.SPEAK_ID, "logprob": math.log(max(1 - p, 1e-9)) - 0.2}]
            return _Reply({"completion_probabilities": [{"id": chosen, "logprob": top[0 if chosen == self.LISTEN_ID else 1]["logprob"],
                                                         "top_logprobs": top}],
                           "timings": {"prompt_n": 7, "cache_n": 900}})
        if path == "v1/chat/completions":
            content = __import__("json").dumps({"action": SPEAK, "delegate": self.delegate, "say": self.say})
            return _Reply({"choices": [{"message": {"content": content}}]})
        return _Reply({}, 404)


def microturn_units(check: Any) -> None:
    m = window_message("  hey can   you ", 0.83, False)
    check(m == "[voice] hey can you [p=0.8]", f"microturn: window message with words {m!r}")
    m = window_message("", None, True)
    check(m == f"{NO_VOICE} {SPEAKING}", f"microturn: silent window while speaking {m!r}")
    m = window_message("", 1.7, False)
    check(m == f"{NO_VOICE} [p=1.0]", f"microturn: p clamped {m!r}")

    top = [{"id": 1, "logprob": math.log(0.27)}, {"id": 2, "logprob": math.log(0.20)}, {"id": 3, "logprob": math.log(0.18)}]
    # the video's case: listen 27% wins the argmax while the first words of speech hold 73% in many pieces. The
    # rule sees only the two action tokens: listen 0.27 against speak = the rest of the two-way mass
    p = listen_probability(top, {1}, {2, 3})
    check(p is not None and abs(p - 0.27 / 0.65) < 1e-9, f"microturn: P(listen) over the two actions {p}")
    check(listen_probability(top, {9}, {8}) is None, "microturn: neither action in the top list -> None")
    check(listen_probability([{"id": "x"}, {"logprob": 1}], {1}, {2}) is None, "microturn: malformed entries skipped")
    check(decide(0.30) == LISTEN and decide(0.29) == SPEAK and decide(None) == LISTEN,
          "microturn: rule at the 0.30 threshold, None listens")
    check(decide(0.45, 0.5) == SPEAK, "microturn: threshold is a parameter")

    check(not keep_in_history(NO_VOICE, LISTEN), "microturn: a silent window is not committed")
    check(not keep_in_history(f"{NO_VOICE} [p=0.1] {SPEAKING}", LISTEN), "microturn: silent while speaking, not committed")
    check(keep_in_history("[voice] hey", LISTEN), "microturn: the user's words are committed even on listen")
    check(keep_in_history(NO_VOICE, SPEAK, say="mm-hm"), "microturn: a spoken window is committed")
    check(keep_in_history(NO_VOICE, SPEAK, delegate="run the tests"), "microturn: a delegating window is committed")
    check(not keep_in_history(NO_VOICE, SPEAK, say="  "), "microturn: speak with nothing said is not committed")

    check(fields_of(json.dumps({"action": SPEAK, "delegate": "x", "say": "y"})) == ("y", "x"), "microturn: fields of a turn")
    check(fields_of("plain words") == ("", ""), "microturn: plain text has no fields")
    order = list(FORMAT["json_schema"]["schema"]["properties"])
    check(order[0] == "action" and FORMAT["json_schema"]["schema"]["properties"]["action"]["enum"] == [LISTEN, SPEAK],
          f"microturn: action is the first field and an enum {order}")

    async def scripted() -> None:
        srv = ScriptedServer([0.9, 0.1, 0.30])
        st = MicroStepper(srv, "http://s", "m", "SYS", threshold=THRESHOLD)
        await st.resolve_tokens()
        check(st.listen_ids == {srv.LISTEN_ID} and st.speak_ids == {srv.SPEAK_ID}, "microturn: action token ids resolved")
        hist = [{"role": "user", "content": "[voice] hi"}, {"role": "assistant", "content": '{"action": "speak", "delegate": "", "say": "Hey."}'}]
        s1 = await st.step(hist, NO_VOICE)
        check(s1.action == LISTEN and s1.p_listen is not None and s1.p_listen > 0.8 and s1.speak_ms == 0,
              f"microturn: listen step ends after the decision (p={s1.p_listen})")
        body = srv.posts[-1][1]
        check(body["prompt"].endswith(PREFIX) and body["n_predict"] == 1 and body["n_probs"] == 10 and body["cache_prompt"],
              "microturn: decision request is 1 token on the templated prompt plus the action prefix")
        check(s1.timings.get("cache_n") == 900, "microturn: server timings kept on the step")
        s2 = await st.step(hist, "[voice] can you run the tests [p=0.9]")
        check(s2.action == SPEAK and s2.say == "Sure, on it." and s2.speak_ms >= 0, f"microturn: speak step reads say {s2.say!r}")
        spoken = srv.posts[-1][1]
        check(spoken.get("logit_bias") == {str(srv.LISTEN_ID): -100} and spoken["response_format"] == FORMAT
              and spoken["messages"][-1]["content"].startswith("[voice] can you"),
              "microturn: speaking turn blocks the listen token and sends the window")
        s3 = await st.step(hist, NO_VOICE)
        check(s3.action == LISTEN and s3.p_listen is not None and abs(s3.p_listen - 0.30) < 1e-6,
              f"microturn: exactly the threshold listens (p={s3.p_listen})")
        lazy = MicroStepper(ScriptedServer([0.9]), "http://s", "m", "SYS")
        s4 = await lazy.decide(hist, NO_VOICE)
        check(lazy.listen_ids and lazy.speak_ids and s4.p_listen is not None,
              f"microturn: the first decision resolves the action tokens itself (p={s4.p_listen})")

    asyncio.run(scripted())

    check(is_echo("add a unit test for the parser.", "can you add a unit test for the parser and run it"),
          "microturn: say made of the user's words is an echo")
    check(not is_echo("Sure, on it.", "can you add a unit test for the parser"), "microturn: a reply is not an echo")
    check(not is_echo("", "anything"), "microturn: empty say is not an echo")

    asyncio.run(loop_units(check))


class ScriptedStepper:
    """Stands in for MicroStepper: a script of (action, say, delegate) per call; records the windows asked."""

    def __init__(self, script: list[tuple[str, str, str]]):
        self.script = list(script)
        self.windows: list[str] = []

    def _next(self, window: str) -> MicroStep:
        self.windows.append(window)
        action, say, delegate = self.script.pop(0) if self.script else (LISTEN, "", "")
        return MicroStep(window=window, action=action, p_listen=0.9 if action == LISTEN else 0.1, say=say,
                         delegate=delegate)

    async def decide(self, history: list, window: str) -> MicroStep:
        s = self._next(window)
        s.say, s.delegate = "", ""  # the decision alone knows no fields
        return s

    async def step(self, history: list, window: str) -> MicroStep:
        return self._next(window)


class World:
    """The loop's hands, recording what it did; `playing` is what the fake voice reports."""

    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.ended = self.cuts = 0
        self.delegated: list[tuple[str, str]] = []
        self.shown: list[tuple[str, str]] = []
        self.playing = False
        self.turn_busy = False
        self.logs: list[str] = []
        self.notes: list[str] = []

    def deps(self) -> LoopDeps:
        async def end_run() -> None:
            self.ended += 1
            self.playing = False

        async def cut() -> None:
            self.cuts += 1
            self.playing = False

        async def delegate(request: str, said: str) -> bool:
            self.delegated.append((request, said))
            return True

        def speak(text: str) -> None:
            self.spoken.append(text)
            self.playing = True

        return LoopDeps(speak=speak, end_run=end_run, cut=cut, delegate=delegate, speaking=lambda: self.playing,
                        busy=lambda: self.turn_busy, show=lambda t, k: self.shown.append((t, k)), note=self.notes.append,
                        is_backchannel=lambda t: t.strip(" .").lower() in ("mm-hm", "uh-huh", "okay", "yeah"),
                        log=self.logs.append)


async def loop_units(check: Any) -> None:
    # 1. A request: the user talks (p low), the model wants in mid-sentence, the floor holds it; at the end the
    #    model speaks and delegates once; the run ends as one assistant message of what was heard.
    st = ScriptedStepper([(SPEAK, "add a unit test.", "add a unit test"),  # mid-sentence: decision only, held
                          (SPEAK, "Sure, on it.", "add a unit test for the parser"),  # after the utterance
                          (SPEAK, "Sure, on it.", "add a unit test for the parser"),  # a repeat: one delegation
                          (LISTEN, "", "")])
    w = World()
    hist: list[dict] = []
    loop = MicroLoop(st, hist, w.deps(), step_ms=300)
    loop.on_piece("can", 0.1)
    loop.on_piece("you add", 0.1)
    s = await loop.tick()
    check(s is not None and s.action == LISTEN and not w.spoken and not w.delegated,
          f"microloop: the floor holds a speak while the user talks ({s and s.action})")
    check(st.windows[-1].startswith("[voice] can you add [p=0.1]"), f"microloop: the window is the utterance so far {st.windows[-1]!r}")
    loop.on_piece("a unit test for the parser", 0.9)
    loop.on_utterance_end("Can you add a unit test for the parser?")
    check(loop.final is not None and not hist and not w.notes,
          "microloop: the ended utterance waits for its window; its note waits for the next step")
    s = await loop.tick()
    check(st.windows[-1] == "[voice] Can you add a unit test for the parser? [p=0.9]",
          f"microloop: the window after the end is the whole utterance with the turn over {st.windows[-1]!r}")
    check(hist[-1]["role"] == "user" and "Can you add a unit test" in hist[-1]["content"],
          "microloop: after that window the utterance is one user message in the history")
    check(s is not None and s.action == SPEAK and w.spoken == ["Sure, on it."]
          and w.delegated == [("add a unit test for the parser", "Can you add a unit test for the parser?")] and not w.notes,
          f"microloop: after the utterance the front speaks and delegates with the user's words; no note ({w.delegated})")
    s = await loop.tick()
    check(len(w.delegated) == 1 and w.spoken == ["Sure, on it.", "Sure, on it."], "microloop: one delegation per utterance; the run goes on")
    w.playing = False  # the voice finished the queued words
    s = await loop.tick()
    check(s is not None and s.action == LISTEN and w.ended == 1, "microloop: listen after the voice finished ends the run")
    loop.run_ended(["Sure, on it.", "Sure, on it."], cut=False)
    check(hist[-1]["role"] == "assistant" and '"action": "speak"' in hist[-1]["content"]
          and '"delegate": "add a unit test for the parser"' in hist[-1]["content"] and w.shown == [("Sure, on it. Sure, on it.", "reply")],
          f"microloop: the run is one assistant message with its delegation {hist[-1]['content'][:80]!r}")
    check(loop.pending_note is None and loop.utterance == [] and not w.notes, "microloop: a handed-off utterance leaves no note")
    # a confirmation the front only listens to reaches Claude as a note on the next step
    loop.on_piece("sounds good", 0.9)
    loop.on_utterance_end("Sounds good.")
    st.script.append((LISTEN, "", ""))
    await loop.tick()
    check(w.notes == ["Sounds good."], f"microloop: an utterance the front listened to is a note {w.notes}")

    # 2. Idle: silent windows ask the model IDLE_SKIP times, then stop asking until something happens.
    st = ScriptedStepper([])
    w = World()
    loop = MicroLoop(st, [], w.deps(), step_ms=300)
    asked = 0
    for _ in range(IDLE_SKIP + 4):
        asked += (await loop.tick()) is not None
    check(asked == IDLE_SKIP and len(st.windows) == IDLE_SKIP, f"microloop: idle after {asked} silent asks")
    loop.on_piece("hey", 0.2)
    check((await loop.tick()) is not None, "microloop: a word wakes the loop")

    # 3. Echo and empty says are dropped; a backchannel may be spoken while the user talks is held for now (decision only).
    st = ScriptedStepper([(SPEAK, "run the whole suite.", ""), (SPEAK, "   ", ""), (SPEAK, "Okay, running it.", "")])
    w = World()
    loop = MicroLoop(st, [], w.deps(), step_ms=300)
    loop.on_piece("run the whole suite", 0.9)
    loop.on_utterance_end("run the whole suite")
    s = await loop.tick()
    check(s is not None and s.action == LISTEN and not w.spoken, "microloop: an echo of the user's words is not spoken")
    s = await loop.tick()
    check(s is not None and s.action == LISTEN and not w.spoken, "microloop: an empty say is not spoken")
    s = await loop.tick()
    check(s is not None and s.action == SPEAK and w.spoken == ["Okay, running it."], "microloop: a reply is spoken")

    # 4. Barge-in by the model: the user speaks over the voice and the model answers listen: the voice is cut.
    w.playing = True
    loop.on_piece("wait", 0.1)
    st.script.append((LISTEN, "", ""))
    s = await loop.tick()
    check(s is not None and s.action == LISTEN and w.cuts == 1, "microloop: listen over the user's words cuts the voice")

    # 5. A turn holding the floor (an announcement) skips the tick; an in-flight step skips too.
    w.turn_busy = True
    check((await loop.tick()) is None and loop.skipped == 1, "microloop: no step while a turn holds the floor")
    w.turn_busy = False
    loop.in_flight = True
    check((await loop.tick()) is None and loop.skipped == 2, "microloop: no step while one is in flight")
    loop.in_flight = False

    # 6. A backchannel heard and played through is not a message
    before = len(loop.history)
    loop.on_utterance_end("")
    check(len(loop.history) == before, "microloop: an empty utterance end adds nothing")

    # 7. A hand-off is never silent, and its say may name the task in the user's words.
    st = ScriptedStepper([(SPEAK, "", "check the status of this repo"),
                          (SPEAK, "Checking the status of this repo.", "look at the failing test")])
    w = World()
    hist = []
    loop = MicroLoop(st, hist, w.deps(), step_ms=200)
    loop.on_piece("can you check the status of this repo", 0.7)
    loop.on_utterance_end("can you check the status of this repo")
    await loop.tick()
    check(w.spoken == [HANDOFF_SAY] and len(w.delegated) == 1, f"microloop: a hand-off with no say speaks {HANDOFF_SAY!r} ({w.spoken})")
    loop.run_ended(w.spoken, cut=False)
    w.playing = False
    loop.on_piece("now look at the failing test", 0.8)
    loop.on_utterance_end("now look at the failing test")
    await loop.tick()
    check(w.spoken[-1] == "Checking the status of this repo." and len(w.delegated) == 2,
          f"microloop: a hand-off's say is not dropped as an echo ({w.spoken[-1]!r})")
    check(st.windows[-1].endswith("[p=0.8]") and not st.windows[-1].startswith(NO_VOICE),
          f"microloop: an ended utterance is asked with its own p when above EOT {st.windows[-1]!r}")

    # 8. The user moves on before the ended utterance got its window: it still joins the history, in order.
    st = ScriptedStepper([])
    loop = MicroLoop(st, hist := [], World().deps(), step_ms=200)
    loop.on_piece("first", 0.9)
    loop.on_utterance_end("first thing")
    loop.on_piece("second", 0.1)
    check(len(hist) == 1 and "first thing" in hist[0]["content"], f"microloop: an unasked utterance is committed when the next begins {hist}")
    check(st.windows == [] and (await loop.tick()) is not None and st.windows[-1].startswith("[voice] second"),
          f"microloop: the new utterance is the window {st.windows}")
