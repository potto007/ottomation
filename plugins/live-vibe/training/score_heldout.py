#!/usr/bin/env -S uv run --script --python 3.12 --with numpy --with httpx
"""Score a front server on make_windows.py's held-out windows, through the loop's own stepper (the same two requests
the sidecar makes). With and without `--lora` on llama-server, this is the LoRA's acceptance test.

  uv run --python 3.12 --with numpy --with httpx python plugins/live-vibe/training/score_heldout.py \\
      --url http://127.0.0.1:8092 --windows <dir>/heldout_windows.jsonl [--limit 400] [--out scores.jsonl]

Per window: the decision (listen/speak) against the label, and for a labelled speak the delegation (present when
expected, absent when not) and the say (a claim about work, an echo of the user's words, or close to the label). The
report is accuracy by window kind, so a model that always listens cannot hide behind the silent windows. The
delegation and say columns are over the windows where both the label and the model speak (`spoke`), so a wrong
decision counts once, in the action column, and not again as a missing delegation; the echo test skips a hand-off,
as the loop's floor does, since a hand-off names the task in the user's words.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "bin"))  # plugins/live-vibe/bin: `import sidecar`

import httpx  # noqa: E402

from sidecar.front import _CLAIM, LlamaCppBrain  # noqa: E402
from sidecar.microturn import MICRO_PROTOCOL, SPEAK, MicroStepper, is_echo  # noqa: E402


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(args.windows) if line.strip()]
    if args.limit:
        step = max(1, len(rows) // args.limit)
        rows = rows[::step][:args.limit]
    client = httpx.AsyncClient(timeout=60.0)
    stepper: MicroStepper | None = None
    system_seen = ""
    by_kind: dict[str, Counter] = defaultdict(Counter)
    out_rows = []
    for n, r in enumerate(rows):
        if r["system"] != system_seen:  # the stepper appends MICRO_PROTOCOL itself: hand it the front prompt alone
            base = r["system"].removesuffix(MICRO_PROTOCOL)
            stepper = MicroStepper(client, args.url, "default", base, sampling=LlamaCppBrain.SAMPLING)
            system_seen = r["system"]
        assert stepper is not None
        step = await stepper.step(r["history"], r["window"])
        kind = r["kind"]
        c = by_kind[kind]
        c["n"] += 1
        c["action_ok"] += step.action == r["action"]
        if r["action"] == SPEAK and step.action == SPEAK:
            c["spoke_n"] += 1
            want_delegate = bool(r["delegate"].strip())
            handing = bool(step.delegate.strip())
            c["delegate_ok"] += handing == want_delegate
            if want_delegate and handing:
                overlap = len(words(step.delegate) & words(r["delegate"])) / max(1, len(words(r["delegate"])))
                c["delegate_close"] += overlap >= 0.5
            c["say_claim"] += bool(_CLAIM.search(step.say)) and not bool(_CLAIM.search(r["say"]))
            user_words = r["window"].split("[voice]")[-1].split("[p=")[0] if "[voice]" in r["window"] else ""
            c["say_echo"] += bool(step.say) and bool(user_words.strip()) and not handing and is_echo(step.say, user_words)
            c["say_long"] += len(step.say.split()) > 14
        out_rows.append({"kind": kind, "window": r["window"], "want": [r["action"], r["delegate"], r["say"]],
                         "got": [step.action, step.delegate, step.say], "p_listen": step.p_listen,
                         "ms": round(step.decide_ms + step.speak_ms)})
        if n % 50 == 0:
            print(f"{n}/{len(rows)}", flush=True)
    await client.aclose()
    if args.out:
        Path(args.out).write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out_rows))
    total = Counter()
    print(f"\n== {args.label or args.url}: {len(rows)} windows ==")
    print(f"{'kind':<16} {'n':>5} {'action':>7} {'spoke':>6} {'deleg':>6} {'close':>6} {'claim':>6} {'echo':>5} {'long':>5}")

    def row(name: str, c: Counter) -> str:
        sp = max(1, c["spoke_n"])
        line = f"{name:<16} {c['n']:>5} {100 * c['action_ok'] / c['n']:>6.0f}% "
        if not c["spoke_n"]:
            return line
        return line + (f"{c['spoke_n']:>6} {100 * c['delegate_ok'] / sp:>5.0f}% {100 * c['delegate_close'] / sp:>5.0f}% "
                       f"{100 * c['say_claim'] / sp:>5.0f}% {100 * c['say_echo'] / sp:>4.0f}% {100 * c['say_long'] / sp:>4.0f}%")

    for kind, c in sorted(by_kind.items()):
        total.update(c)
        print(row(kind, c))
    print(row("all", total))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
