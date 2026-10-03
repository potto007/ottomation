"""Front relay eval runner. Drives each fixture in fixtures.py through a sidecar's real FrontSession (LlamaCppBrain,
Delegator, waiting results, notes) with a silent voice, against an OpenAI-compatible server, and records the speech,
any announcement that follows the turn, and every emitted delegate and note. See README.md for the procedure.

  run.py run <variant> <out.jsonl> [--ref REF] [--runs 3] [--url http://127.0.0.1:8092] [--only id,id]
  run.py dry [--ref REF]                       print each fixture's messages; calls no server
  run.py score <a.jsonl> [b.jsonl ...]

--ref picks the sidecar code: "worktree" (default: this checkout's files) or any git ref of this repo (extracted with
git archive into a temporary directory), so one server can be scored against several versions.
"""
from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
BIN = HERE.parents[1]  # plugins/live-vibe/bin, the parent of the sidecar package
WORKSPACE = str(BIN.parents[2])  # the repo: the project the front prompt names
SENT = re.compile(r"(?<=[.!?])\s+")
sys.path.insert(0, str(HERE))


def sidecar_root(ref: str) -> str:
    """The directory to put on sys.path for `import sidecar` at this ref."""
    if ref == "worktree":
        return str(BIN)
    repo = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                          check=True).stdout.strip()
    sub = BIN.relative_to(repo) / "sidecar"
    dest = tempfile.mkdtemp(prefix="live-vibe-relay-")
    archive = subprocess.run(["git", "-C", repo, "archive", ref, str(sub)], capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", dest], input=archive, check=True)
    return str(Path(dest) / BIN.relative_to(repo))


def build(front, fx: dict) -> tuple[list[dict], str]:
    """The history before the turn under test (as that version writes its turns and reports), and its message."""
    order = list(front.LlamaCppBrain.FORMAT["json_schema"]["schema"]["properties"])

    def turn(delegate: str, say: str) -> dict:
        fields = {"delegate": delegate, "say": say}
        return {"role": "assistant", "content": json.dumps({k: fields[k] for k in order}, ensure_ascii=False)}

    def event(texts: list[str]) -> str:
        if not hasattr(front, "report_brief"):
            return f"{front.EVENT} {' '.join(texts).strip()[:front.EVENT_CHARS]}"
        return front.event_message("\n\n".join(front.report_brief(t) for t in texts))

    history: list[dict] = []
    for p in fx["prior"]:
        if p[0] == "report":
            history += [{"role": "user", "content": event([p[1]])}, turn("", p[2])]
        else:
            history += [{"role": "user", "content": p[0]}, turn(p[1], p[2])]
    return history, (event(fx["reports"]) if fx["user"] is None else fx["user"])


class SilentVoice:
    """Voice.speak's contract without audio: every sentence is heard whole."""

    def __init__(self) -> None:
        self.speaking = threading.Event()

    async def speak(self, sentences, heard, on_play=lambda: None) -> bool:
        async for s in sentences:
            on_play()
            heard.append(s)
        return False


async def announce(front, sess, texts: list[str]) -> str:
    """What the announcer would say next for these results (FrontSession.background, minus the floor wait)."""
    report = "\n\n".join(front.report_brief(t) for t in texts)
    await sess.run_turn(front.event_message(report), is_event=True, report=report)
    return " ".join(sess.turn_spoken).strip()


async def run(a: argparse.Namespace) -> None:
    sys.path.insert(0, sidecar_root(a.ref))
    from sidecar import front, protocol  # noqa: E402
    from fixtures import F  # noqa: E402

    takes_waiting = "waiting" in inspect.signature(front.FrontSession.run_turn).parameters
    takes_said = "said" in inspect.signature(front.Delegator.call).parameters
    sink: list[dict] = []
    protocol.capture(sink)
    out = None if a.dry else open(a.out, "a")
    for fx in F:
        if a.only and fx["id"] not in a.only.split(","):
            continue
        history, message = build(front, fx)
        if a.dry:
            print(f"== {fx['id']} ({a.variant}) waiting={len(fx['waiting'])} into the turn={takes_waiting}")
            for m in history + [{"role": "user", "content": message}]:
                print(f"  {m['role']}: {m['content'][:130]!r}")
            continue
        for n in range(a.runs):
            brain = front.LlamaCppBrain(front.FRONT_PROMPT.format(workspace=WORKSPACE), a.url, "")
            brain.history = [dict(m) for m in history]
            sess = front.FrontSession(SilentVoice(), lambda: False, brain, None, lambda: None)
            sess.question_open = bool(fx.get("question_open"))
            for said, request in fx.get("sent", []):
                sess.tools.sent.append((time.monotonic(), f"{said} {request}" if takes_said else request))
            sink.clear()
            if fx["user"] is None:
                report = "\n\n".join(front.report_brief(t) for t in fx["reports"])
                await sess.run_turn(message, is_event=True, report=report)
            elif takes_waiting:
                await sess.run_turn(message, waiting=list(fx["waiting"]))
            else:
                await sess.run_turn(message)
            spoken = " ".join(sess.turn_spoken).strip()
            turn_sink = list(sink)
            # what the announcer says next: results put back by the turn, or (before they joined user turns) the
            # results that waited, which that version announced after the user's turn
            later = sess.take_results() if hasattr(sess, "take_results") else []
            if not takes_waiting and fx["user"] is not None:
                later = list(fx["waiting"])
            announced = await announce(front, sess, later) if later else ""
            await brain.aclose()
            delegates = [o for o in turn_sink if o.get("type") == "delegate"]
            notes = [o for o in turn_sink if o.get("type") == "note"]
            rec = {"variant": a.variant, "id": fx["id"], "run": n, "spoken": spoken, "announced": announced,
                   "delegate": delegates[0]["text"] if delegates else "",
                   "said": delegates[0].get("said", "") if delegates else "", "delegates": len(delegates),
                   "notes": notes, "logs": [o["text"] for o in turn_sink if o.get("type") == "log"]}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            kind = {k: v for k, v in notes[0].items() if k in ("answer", "ask")} if notes else None
            print(f"[{a.variant}] {fx['id']} #{n}: {spoken} | delegate: {rec['delegate']!r}"
                  + (f" | note: {kind}" if notes else "") + (f" | then announced: {announced}" if announced else ""),
                  flush=True)
    protocol.capture(None)


def check(fx: dict, r: dict) -> list[str]:
    """Why the record fails the fixture's own rubric (empty: it passes). The speech is what the user heard: the
    turn's reply and any announcement right after it."""
    k, t, why = fx["rubric"], (r["spoken"] + " " + r.get("announced", "")).strip(), []
    if not any(re.search(p, t, re.I) for p in k["keep"]):
        why.append("keep")
    if not all(re.search(p, t, re.I) for p in k.get("also", [])):
        why.append("also")
    why += [f"forbid {p}" for p in k["forbid"] if re.search(p, t, re.I)]
    if k["delegate"] != "any" and (k["delegate"] == "yes") != bool(r["delegate"]):
        why.append(f"delegate {k['delegate']}")
    if (m := k.get("max_sentences")) and len([s for s in SENT.split(r["spoken"]) if s]) > m:
        why.append(f"over {m} sentences")
    if k.get("deleg_keep") and r["delegate"] and not any(re.search(p, r["delegate"], re.I) for p in k["deleg_keep"]):
        why.append("delegation off topic")
    if k.get("said_verbatim") and r["delegate"] and r.get("said") != fx["user"]:
        why.append("delegation without the user's words")
    note = r["notes"][0] if r["notes"] else {}
    if (want := k.get("note")) is not None and any(note.get(f) != v for f, v in want.items()):
        why.append(f"note {want} (got {note or 'none'})")
    if k.get("answer_or_delegation") and not (note.get("answer") is True
                                              or (r["delegate"] and r.get("said") == fx["user"])):
        why.append("neither an answer note nor a delegation with the user's words")
    return why


def extra(fx: dict, r: dict) -> list[str] | None:
    """Why the record fails the fixture's extra checks; None when they do not apply (no delegation to judge)."""
    x, why = fx["extra"], []
    if x.get("first_keep"):
        first = SENT.split(r["spoken"].strip(), maxsplit=1)[0]
        if not any(re.search(p, first, re.I) for p in x["first_keep"]):
            why.append("first sentence not a yes")
    if x.get("deleg_keep"):
        if not r["delegate"]:
            return None
        if not any(re.search(p, r["delegate"], re.I) for p in x["deleg_keep"]):
            why.append("delegation off topic")
        why += [f"delegation re-asks: {p}" for p in x.get("deleg_forbid", []) if re.search(p, r["delegate"], re.I)]
    return why


def score(paths: list[str]) -> None:
    from fixtures import DOUBLED, F, INVENTED  # noqa: E402

    fxs = {f["id"]: f for f in F}
    by: dict = defaultdict(list)
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            by[r["variant"]].append(r)
    for v, rows in by.items():
        own: dict = defaultdict(lambda: [0, 0])
        ext: dict = defaultdict(lambda: [0, 0])
        inv: dict = defaultdict(int)
        dbl: dict = defaultdict(int)
        notes: dict = defaultdict(list)
        fails = []
        for r in rows:
            fx = fxs[r["id"]]
            why = check(fx, r)
            own[r["id"]][0] += not why
            own[r["id"]][1] += 1
            if "extra" in fx and (e := extra(fx, r)) is not None:
                ext[r["id"]][0] += not e
                ext[r["id"]][1] += 1
                why += [f"extra: {w}" for w in e]
            if fx["user"] is not None and not r["delegate"] and any(re.search(p, r["spoken"], re.I) for p in INVENTED):
                inv[r["id"]] += 1
                why.append("invented action")
            if re.search(DOUBLED, r["spoken"], re.I):
                dbl[r["id"]] += 1
                why.append("asked twice")
            kinds = [k for k in ("answer", "ask") if r["notes"] and r["notes"][0].get(k)]
            notes[r["id"]].append(kinds[0] if kinds else ("note" if r["notes"] else "-"))
            if why:
                then = f" | then announced: {r['announced']}" if r.get("announced") else ""
                fails.append(f"   x {r['id']}#{r['run']} [{', '.join(why)}]: {r['spoken']} | delegate "
                             f"{r['delegate']!r}{then}")
        print(f"== {v}")
        print(f"   {'fixture':20} own   extra     invented  asked-twice  note kinds")
        for i in fxs:
            if i in own:
                e = f"{ext[i][0]}/{ext[i][1]}" if ext[i][1] else ("n/a" if "extra" in fxs[i] else "")
                print(f"   {i:20} {own[i][0]}/{own[i][1]}   {e:8}  {inv[i]:<8}  {dbl[i]:<11}  {','.join(notes[i])}")
        print(f"   own total {sum(o[0] for o in own.values())}/{sum(o[1] for o in own.values())}, invented actions "
              f"{sum(inv.values())}, asked twice {sum(dbl.values())}")
        print("\n".join(fails))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("variant")
    r.add_argument("out")
    d = sub.add_parser("dry")
    for p in (r, d):
        p.add_argument("--ref", default="worktree")
        p.add_argument("--runs", type=int, default=3)
        p.add_argument("--url", default="http://127.0.0.1:8092")
        p.add_argument("--only", default="")
    s = sub.add_parser("score")
    s.add_argument("files", nargs="+")
    a = ap.parse_args()
    if a.cmd == "score":
        score(a.files)
        return
    a.dry = a.cmd == "dry"
    if a.dry:
        a.variant, a.out = a.ref, "-"
    asyncio.run(run(a))


if __name__ == "__main__":
    main()
