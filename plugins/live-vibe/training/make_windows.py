#!/usr/bin/env -S uv run --script --python 3.12 --with numpy --with httpx
"""Synthetic calls for the timing LoRA (docs/design/microturn-front.md, increment 3).

A call is a few beats: the user asks for work, the front hands it off with a few words, a report comes back (status,
sometimes a question), the user confirms, asks how things stand, asks about the work, or chats. Each utterance is
realized the way Kyutai delivers it (a word or two per 200 ms step, splits such as "statu s", punctuation only at the
end) and cut into the loop's windows with the end-of-turn probability the loop sees. Every window gets the answer the
loop should give: listen while the user talks or the voice plays, speak with a short hand-off on a finished request,
speak a status answer from a fresh report, "Let me check." and a delegation for a question about the work, a short
line for small talk, nothing for a "no".

The system prompt, the window message, the event and notes messages and the JSON turn come from the sidecar itself,
so an example is exactly what the loop sends. Output: train.jsonl and heldout.jsonl (prompt/completion, conversational)
plus heldout_windows.jsonl (window, history, label) for score_heldout.py. Run from anywhere:

  uv run --python 3.12 --with numpy --with httpx python plugins/live-vibe/training/make_windows.py --out <dir>
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "bin"))  # plugins/live-vibe/bin: `import sidecar`

from sidecar.front import (  # noqa: E402
    FRONT_PROMPT_EXPERIMENTAL,
    HOLD,
    RETELL_STATUS,
    event_message,
    notes_message,
)
from sidecar.microturn import LISTEN, SPEAK, MICRO_PROTOCOL, json_turn, window_message  # noqa: E402
from sidecar.timing_lora import fingerprint  # noqa: E402

WORKSPACES = ["/home/dev/src/local-ai", "/Users/ana/code/shop-api", "/home/potto/src/knowledge-base", "/srv/work/billing",
              "/home/dev/projects/relay", "C:\\Users\\sam\\dev\\inventory"]

# -- the world: tasks, questions, reports ----------------------------------------------------------------------
FILES = ["the parser", "the config loader", "the login page", "the export job", "the retry logic", "the CSV import",
         "the rate limiter", "the search index", "the deploy script", "the date formatter", "the cache", "the webhook handler"]
TARGETS = ["src/parser.py", "app/config.ts", "web/login.tsx", "jobs/export.py", "lib/retry.go", "import/csv.rs",
           "api/limits.py", "search/index.ts", "scripts/deploy.sh", "util/dates.py", "core/cache.py", "hooks/webhook.py"]
ISSUES = [4, 5, 8, 12, 17, 23, 31]
BRANCHES = ["feat/retry", "fix/csv-dates", "docs/tetris", "chore/deps", "backup/testing"]

TASKS: list[tuple[list[str], str, str]] = [
    # (ways the user says it, the delegation, the hand-off said)
    (["add a unit test for {f}", "can you add a unit test for {f}", "write a test for {f}", "we need a unit test for {f}, can you do that"],
     "Add a unit test for {f} ({t})", "Adding a unit test for {f}."),
    (["run the tests", "run the test suite", "can you run the tests again", "kick off the tests"],
     "Run the test suite and report the result", "Running the tests."),
    (["check the status of this repo", "what's the status of the repo", "give me the current status", "check the repo status"],
     "Check the repository status: branch, uncommitted changes, recent commits", "Checking the repo status."),
    (["check for open PRs", "are there any open pull requests", "list the open PRs", "what pull requests are open"],
     "List the open pull requests", "Checking the open pull requests."),
    (["check the open issues", "list the open issues", "what issues are open", "pull up the open issues"],
     "List the open GitHub issues", "Checking the open issues."),
    (["fix the failing test in {f}", "the test for {f} is failing, can you fix it", "fix {f}'s failing test"],
     "Fix the failing test for {f} ({t})", "Fixing the failing test for {f}."),
    (["refactor {f} to use the new client", "move {f} over to the new client", "can you port {f} to the new client"],
     "Refactor {f} ({t}) to use the new client", "Refactoring {f}."),
    (["delete the old branches", "clean up the stale branches", "remove the merged branches"],
     "Delete the merged branches locally and on origin, after checking main has their content", "Cleaning up the old branches."),
    (["look at issue {i}", "tell me about issue {i}", "what does issue {i} involve", "what's issue {i} about"],
     "Read issue #{i} and summarize what it involves and its current state", HOLD),
    (["close issue {i}", "go ahead and close issue {i}", "can you close issue {i}"],
     "Close issue #{i} with a comment explaining why", "Closing issue {i}."),
    (["bump the version", "bump the version to the next patch", "cut a patch release"],
     "Bump the version to the next patch and tag it", "Bumping the version."),
    (["update the readme for {f}", "the readme is out of date on {f}, fix it", "document {f} in the readme"],
     "Update the README section on {f}", "Updating the README."),
    (["check whether the server on 8080 is up", "is the inference server running", "see if 8080 is up"],
     "Check whether the inference server on port 8080 is running", "Checking the server on 8080."),
    (["rename {f} to something clearer", "give {f} a better name"],
     "Rename {f} ({t}) to a clearer name and update its callers", "Renaming {f}."),
]

STATUS_QUESTIONS = ["how's it going", "is it done yet", "where are we", "what's the status", "any update", "is it still running",
                    "how are things going", "did that finish", "are you done"]
CONTENT_QUESTIONS = [("what did it find", "Tell the user what the last check found"),
                     ("why did the test fail", "Explain why the test failed"),
                     ("what would that involve", "Explain what the proposed change would involve"),
                     ("tell me more about that", "Give the user more detail on the last report"),
                     ("what about the other one", "Tell the user about the other item from the last report"),
                     ("can you give me more detail", "Give the user more detail on the last report"),
                     ("what changed", "Tell the user what changed"),
                     ("which files did it touch", "List the files the last change touched")]
SMALL_TALK = [("hello", "Hello."), ("hi there", "Hi."), ("thanks", "You're welcome."), ("thank you", "Any time."),
              ("that's great", "Glad to hear it."), ("nice", "Good."), ("okay great", "Great."), ("perfect", "Good."),
              ("good morning", "Good morning."), ("talk to you later", "Talk soon."), ("goodbye", "Bye for now.")]
YES = ["yes", "yeah", "yes please", "go ahead", "yes go ahead", "do it", "sure", "okay go ahead with that", "yep do that",
       "all right go ahead"]
NO = ["no", "no thanks", "no leave it", "not now", "no I don't", "leave it for now", "no that's fine"]
FILLERS = ["um", "uh", "okay", "so", "all right", "hmm", "let's see"]
BACKCHANNEL_ONLY = ["mm-hm", "okay", "uh-huh", "yeah", "right"]
STOPS = ["wait", "stop", "hold on", "never mind", "no wait", "hang on"]
DICTATION = ["so the thing is the export job runs every night at two and most nights it is fine but about once a week it "
             "stalls on the last batch and I think it is the retry logic because the log shows the same request going "
             "out three times",
             "I was looking at the login page yesterday and the error only shows up when the session has expired and "
             "you come back from the email link so I want you to reproduce that first before changing anything",
             "the CSV import has two problems one is the date format which is sometimes day first and sometimes month "
             "first and the other is that empty rows at the end make it throw instead of stopping"]

REPORTS: dict[str, list[tuple[str, str, str | None]]] = {
    # status, the spoken sentences (what the front retold), an optional question Claude asks
    "working": [
        ("working", "I've sent a worker to {what}. It's running now and only reading, nothing is changed yet.", None),
        ("working", "The worker is on it. I'll report when it finishes.", None),
    ],
    "done": [
        ("done", "{what_done}. Nothing else is running.", None),
        ("done", "{what_done}.", "Do you want me to {next}?"),
        ("done", "{what_done}. Two things are still open: {open1} and {open2}.", "Should I look into either of them?"),
    ],
    "failed": [
        ("failed", "{what_failed}. I have not changed anything.", "Do you want me to try {alt}?"),
    ],
}
WHAT = {"test": ("add the test", "The unit test for {f} is in and passes", "The new test for {f} fails on the empty input case"),
        "run": ("run the suite", "The suite passed, 212 tests in 40 seconds", "Three tests failed, all in {f}"),
        "status": ("check the repo", "The repo is on main with no uncommitted changes", "The status check could not reach origin"),
        "prs": ("list the pull requests", "There are two open pull requests, both from last week", "The pull request listing timed out"),
        "issues": ("list the issues", "There are three open issues and no milestones", "The issue listing hit a rate limit"),
        "fix": ("fix the test", "The failing test for {f} is fixed and the suite is green", "The fix did not hold, two tests still fail"),
        "refactor": ("refactor {f}", "{f} now uses the new client and the tests pass", "The refactor broke two callers I could not resolve"),
        "branches": ("clean up the branches", "Both old branches are deleted, locally and on origin", "One branch had unmerged work, so I left it"),
        "issue": ("read issue {i}", "Issue {i} is about {f} citing too much and is effectively fixed", "I could not read issue {i}"),
        "close": ("close issue {i}", "Issue {i} is closed with the comment posted", "Closing issue {i} failed, the token lacks rights"),
        "bump": ("bump the version", "Version bumped to the next patch and tagged", "The bump failed, the tag already exists"),
        "readme": ("update the readme", "The README section on {f} is updated", "The README edit conflicted with a newer change"),
        "server": ("check the server", "The server on 8080 is up and serving the agent model", "Nothing is listening on 8080"),
        "rename": ("rename {f}", "{f} is renamed and all callers updated", "The rename left two callers I could not find")}
TASK_KIND = ["test", "run", "status", "prs", "issues", "fix", "refactor", "branches", "issue", "close", "bump", "readme", "server", "rename"]
NEXT = ["check the open issues", "run the full suite", "open a pull request", "close issue {i}", "read the two related commits"]
OPEN = ["issue {i} about {f}", "the flaky test in {f}", "the stale branch {b}", "the README for {f}"]
ALT = ["the other approach", "a smaller change", "running it again", "checking the logs first"]
NOTES_POOL = ["The worker read the issue and its comments. The cause was ranking, not the librarian: raw term counts let a "
              "huge table outrank the right document.",
              "Three tests failed with the same timeout; all call the retry helper with a two second budget.",
              "The branch exists only locally; its two commits are both in main as squashed content.",
              "Origin is in sync; the last commit is a docs merge from this morning."]

STEP_MS = 200
WORDS_PER_STEP = (0, 0, 1, 1, 1, 1, 2)  # Kyutai at about 2.5 words a second, 200 ms steps


@dataclass
class Example:
    history: list[dict[str, Any]]
    window: str
    action: str
    delegate: str = ""
    say: str = ""
    kind: str = ""  # what the window is, for balancing and the held-out report
    speaking: bool = False

    def completion(self) -> str:
        return json_turn(self.action, self.delegate, self.say)


@dataclass
class Call:
    rng: random.Random
    workspace: str
    history: list[dict[str, Any]] = field(default_factory=list)
    examples: list[Example] = field(default_factory=list)
    report: tuple[str, str, str | None] | None = None  # the latest report, if nothing was handed off since
    handed_since: bool = False
    question_open: bool = False
    voice_playing: bool = False  # the front's last line is still playing when the next utterance starts
    fills: dict[str, str] = field(default_factory=dict)

    # -- realization -------------------------------------------------------------------------------------------
    def stt(self, text: str) -> list[str]:
        """Kyutai-like pieces: words with occasional splits; punctuation only on the last piece."""
        words = text.split()
        out: list[str] = []
        for i, w in enumerate(words):
            core = w.rstrip(".?!,")
            if len(core) > 5 and self.rng.random() < 0.12:
                cut = self.rng.randint(2, len(core) - 2)
                out += [core[:cut], core[cut:]]
            else:
                out.append(core)
            if i < len(words) - 1 and w.endswith(",") and self.rng.random() < 0.7:
                out[-1] += " ,"
        return out

    def p_mid(self) -> float:
        r = self.rng.random()
        return round(self.rng.uniform(0.4, 0.6), 1) if r < 0.08 else round(self.rng.uniform(0.0, 0.3), 1)

    def add(self, window: str, action: str, kind: str, delegate: str = "", say: str = "", speaking: bool = False,
            run: list[str] | None = None, run_delegate: str = "") -> None:
        history = list(self.history)
        if run:
            history.append({"role": "assistant", "content": json_turn(SPEAK, run_delegate, " ".join(run))})
        self.examples.append(Example(history, window, action, delegate, say, kind, speaking))

    def utterance(self, text: str, answer: tuple[str, str, str], final_text: str | None = None,
                  over_voice: bool | None = None, backchannel_ok: bool = False) -> None:
        """One user utterance cut into windows: mid-sentence windows (listen), the window where p crosses the
        threshold on raw text (the answer when the sentence is complete), the final clean window, then the run's
        silent windows. `answer` is (action, delegate, say) for the finished utterance."""
        pieces = self.stt(text)
        if over_voice is None:  # the voice is often still playing when the user comes back in
            over_voice = self.voice_playing and self.rng.random() < 0.3
        final = final_text if final_text is not None else text
        if final[-1:] not in ".?!":
            words = [w.strip(",").lower() for w in text.split()]
            lead = next((w for w in words if w not in ("um", "uh", "okay", "so", "all", "right", "hmm", "let's", "see", "and")), "")
            final += "?" if lead in ("what", "what's", "how", "how's", "is", "are", "did", "where", "any", "which", "why", "can") else "."
        heard: list[str] = []
        i = 0
        steps = 0
        while i < len(pieces):
            k = self.rng.choice(WORDS_PER_STEP)
            heard += pieces[i:i + k]
            i += k
            steps += 1
            if not heard:
                continue
            p = self.p_mid()
            spoken_now = over_voice and steps <= 3
            if spoken_now:  # the user talks over the voice: the model's listen cuts it
                self.add(window_message(" ".join(heard), p, True), LISTEN, "over-voice", speaking=True)
                continue
            if backchannel_ok and len(heard) > 12 and p <= 0.3 and self.rng.random() < 0.25:
                self.add(window_message(" ".join(heard), p, False), SPEAK, "backchannel", say="Mm-hm.")
                continue
            self.add(window_message(" ".join(heard), p, False), LISTEN, "mid")
        action, delegate, say = answer
        early = self.rng.random() < 0.5  # p crosses the threshold before Kyutai ends the turn: raw text, no punctuation
        if early:
            raw = " ".join(heard)
            self.add(window_message(raw, round(self.rng.uniform(0.6, 0.8), 1), False), action, "final-raw", delegate, say)
        p_final = round(self.rng.uniform(0.6, 1.0), 1)
        if early and action == SPEAK:  # already answered on the raw window: the clean one is quiet (one line per utterance)
            self.add(window_message(final, p_final, True), LISTEN, "after-answer", speaking=True,
                     run=[say] if say else None, run_delegate=delegate)
        else:
            self.add(window_message(final, p_final, False), action, "final", delegate, say)
        self.history.append({"role": "user", "content": window_message(final, p_final, False)})
        if action == SPEAK and (say or delegate):
            run = [say] if say else []
            for _ in range(self.rng.randint(1, 3)):  # the voice plays: silent windows, listen
                self.add(window_message("", None, True), LISTEN, "silent-speaking", speaking=True, run=run, run_delegate=delegate)
            self.history.append({"role": "assistant", "content": json_turn(SPEAK, delegate, say)})
        self.voice_playing = action == SPEAK and bool(say)
        for _ in range(self.rng.randint(0, 2)):
            self.add(window_message("", None, False), LISTEN, "silent")

    # -- beats -------------------------------------------------------------------------------------------------
    def fill(self, s: str) -> str:
        return s.format(**self.fills)

    def new_fills(self) -> None:
        n = self.rng.randrange(len(FILES))
        self.fills = {"f": FILES[n], "t": TARGETS[n], "i": str(self.rng.choice(ISSUES)), "b": self.rng.choice(BRANCHES)}

    def prefix(self, text: str) -> str:
        if self.rng.random() < 0.35:
            return f"{self.rng.choice(FILLERS)}, {text}"
        return text

    def request(self) -> str:
        self.new_fills()
        n = self.rng.randrange(len(TASKS))
        ways, delegate, say = TASKS[n]
        text = self.prefix(self.fill(self.rng.choice(ways)))
        self.utterance(text, (SPEAK, self.fill(delegate), self.fill(say)))
        self.handed_since, self.question_open = True, False
        return TASK_KIND[n]

    def announce(self, kind: str, status: str | None = None) -> None:
        status = status or self.rng.choices(["working", "done", "failed"], [0.35, 0.5, 0.15])[0]
        st, template, question = self.rng.choice(REPORTS[status])
        what, done, failed = (self.fill(w) for w in WHAT[kind])
        n = self.rng.choice(NEXT)
        fills = {"what": what, "what_done": done, "what_failed": failed, "next": self.fill(n), "alt": self.rng.choice(ALT),
                 "open1": self.fill(self.rng.choice(OPEN)), "open2": self.fill(self.rng.choice(OPEN))}
        spoken = template.format(**fills)
        if question:
            spoken = f"{spoken} {question.format(**fills)}"
        retold = f"Status: {st}. {spoken}"
        if self.rng.random() < 0.5:
            self.history.append({"role": "user", "content": notes_message(self.rng.choice(NOTES_POOL))})
        self.history.append({"role": "user", "content": event_message(retold, RETELL_STATUS)})
        said = spoken if st != "working" else spoken.replace("running now", "still running")
        self.history.append({"role": "assistant", "content": json.dumps({"delegate": "", "say": said}, ensure_ascii=False)})
        self.voice_playing = True  # the announcement is being read out
        self.report = (st, spoken, question.format(**fills) if question else None)
        self.handed_since = False
        self.question_open = question is not None
        for _ in range(self.rng.randint(1, 3)):
            self.add(window_message("", None, False), LISTEN, "silent")

    def status_question(self) -> None:
        text = self.prefix(self.rng.choice(STATUS_QUESTIONS))
        if self.report is not None and not self.handed_since:
            st, spoken, _ = self.report
            first = spoken.split(". ")[0].rstrip(".")
            words = {"working": "Still running", "done": "Done", "failed": "It failed"}[st]
            say = f"{words}: {first[0].lower() + first[1:]}."
            self.utterance(text, (SPEAK, "", say))
        else:
            self.utterance(text, (SPEAK, "", HOLD))
        self.question_open = False

    def content_question(self) -> None:
        q, delegate = self.rng.choice(CONTENT_QUESTIONS)
        self.utterance(self.prefix(q), (SPEAK, delegate, HOLD))
        self.handed_since, self.question_open = True, False

    def answer_claude(self) -> None:
        assert self.report is not None and self.report[2]
        question = self.report[2]
        if self.rng.random() < 0.6:
            text = self.prefix(self.rng.choice(YES))
            task = question.removeprefix("Do you want me to ").removeprefix("Should I ").rstrip("?")
            task = task.removeprefix("look into either of them").strip() or "look into the open items"
            first = task.split()[0]
            doing = {"check": "Checking", "run": "Running", "open": "Opening", "close": "Closing", "read": "Reading",
                     "try": "Trying", "look": "Looking into"}.get(first, "Doing")
            rest = " ".join(task.split()[1:])
            self.utterance(text, (SPEAK, f"Yes: {task[0].upper() + task[1:]}", f"{doing} {rest}.".replace("  ", " ")))
            self.handed_since = True
        else:
            text = self.prefix(self.rng.choice(NO))
            self.utterance(text, (LISTEN, "", ""))
        self.question_open = False

    def small_talk(self) -> None:
        text, say = self.rng.choice(SMALL_TALK)
        self.utterance(text, (SPEAK, "", say))

    def backchannel_only(self) -> None:
        self.utterance(self.rng.choice(BACKCHANNEL_ONLY), (LISTEN, "", ""))

    def dictation(self) -> None:
        self.new_fills()
        text = self.rng.choice(DICTATION)
        self.utterance(text, (SPEAK, f"Investigate: {text[0].upper() + text[1:]}", "Looking into it."), backchannel_ok=True)
        self.handed_since, self.question_open = True, False

    def stop(self) -> None:
        """The user cuts the voice: words over it, listen."""
        text = self.rng.choice(STOPS)
        self.add(window_message(text, 0.2, True), LISTEN, "over-voice", speaking=True)
        self.add(window_message(text + ".", 0.8, False), LISTEN, "stop")
        self.history.append({"role": "user", "content": window_message(text + ".", 0.8, False)})

    def run(self) -> None:
        for _ in range(self.rng.randint(0, 2)):
            self.add(window_message("", None, False), LISTEN, "silent")
        if self.rng.random() < 0.3:
            self.small_talk()
        beats = self.rng.randint(2, 5)
        kind = self.request()
        for _ in range(beats):
            roll = self.rng.random()
            if roll < 0.45:
                self.announce(kind)
                if self.question_open and self.rng.random() < 0.7:
                    self.answer_claude()
            elif roll < 0.60:
                self.status_question()
            elif roll < 0.72:
                self.content_question()
            elif roll < 0.84:
                kind = self.request()
            elif roll < 0.90:
                self.small_talk()
            elif roll < 0.93:
                self.backchannel_only()
            elif roll < 0.96:
                self.dictation()
            elif self.voice_playing:
                self.stop()


def system_prompt(workspace: str) -> str:
    return FRONT_PROMPT_EXPERIMENTAL.format(workspace=workspace) + MICRO_PROTOCOL


def to_row(ex: Example, system: str) -> dict[str, Any]:
    return {"prompt": [{"role": "system", "content": system}, *ex.history, {"role": "user", "content": ex.window}],
            "completion": [{"role": "assistant", "content": ex.completion()}], "kind": ex.kind}


def balance(rows: list[dict[str, Any]], rng: random.Random, listen_per_speak: float) -> list[dict[str, Any]]:
    """Keep every speak window; sample listen windows down to `listen_per_speak` times as many, stratified by kind so
    every kind of silence stays represented (the video's listen bias came from a history full of listens)."""
    speak = [r for r in rows if '"action": "speak"' in r["completion"][0]["content"]]
    listen = [r for r in rows if r not in speak]
    want = int(len(speak) * listen_per_speak)
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for r in listen:
        by_kind.setdefault(r["kind"], []).append(r)
    kept: list[dict[str, Any]] = []
    rest: list[dict[str, Any]] = []
    share = max(1, want // max(1, len(by_kind)))
    for k, rs in by_kind.items():
        rng.shuffle(rs)
        kept += rs[:share]
        rest += rs[share:]
    rng.shuffle(rest)
    kept += rest[:max(0, want - len(kept))]  # small kinds leave room: top up from the large ones
    out = speak + kept
    rng.shuffle(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="directory for train.jsonl, heldout.jsonl, heldout_windows.jsonl")
    ap.add_argument("--calls", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--heldout", type=float, default=0.1, help="share of calls held out")
    ap.add_argument("--listen-per-speak", type=float, default=1.5)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    train: list[dict[str, Any]] = []
    held: list[dict[str, Any]] = []
    held_windows: list[dict[str, Any]] = []
    kinds: dict[str, int] = {}
    for n in range(args.calls):
        ws = rng.choice(WORKSPACES)
        call = Call(random.Random(rng.random()), ws)
        call.run()
        system = system_prompt(ws)
        rows = [to_row(e, system) for e in call.examples]
        for e in call.examples:
            kinds[e.kind] = kinds.get(e.kind, 0) + 1
        if n < int(args.calls * args.heldout):
            held += rows
            held_windows += [{"system": system, "history": e.history, "window": e.window, "action": e.action,
                              "delegate": e.delegate, "say": e.say, "kind": e.kind} for e in call.examples]
        else:
            train += rows
    train_b = balance(train, rng, args.listen_per_speak)
    (out / "train.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train_b))
    (out / "heldout.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in held))
    (out / "heldout_windows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in held_windows))
    # the prompt this data teaches; train_lora.py copies it beside the adapter, and it becomes timing_lora.TRAINED
    (out / "meta.json").write_text(json.dumps({"fingerprint": fingerprint(), "calls": args.calls}, indent=2) + "\n")
    speak =sum('"action": "speak"' in r["completion"][0]["content"] for r in train_b)
    print(f"calls {args.calls}: windows {len(train) + len(held)} ({kinds}); train {len(train_b)} rows "
          f"({speak} speak, {len(train_b) - speak} listen); held-out {len(held)} rows, {len(held_windows)} windows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
