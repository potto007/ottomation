#!/usr/bin/env python3
"""Micro-turn bench: replays a scripted call as windows against a running llama-server and reports, per step, P(listen),
the decision and the latency, then the p50/p95 for silent and speaking steps (docs/design/microturn-front.md, increment
1). No mic, no Kyutai: the windows are text, cut from the script at --step-ms with 2.5 words a second, and the
end-of-turn probability is scripted (low while a sentence runs, high at its end).

  uv run --python 3.12 --with httpx --with numpy python microturn_bench.py --url http://127.0.0.1:8092 [--step-ms 300]
         [--threshold 0.30] [--silence 6] [--out /tmp/live-vibe-microturn-bench.jsonl]

Serve the model with launch_front.sh first (see README.md: check the GPU on its own, 8 GiB or more free).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # plugins/live-vibe/bin: `import sidecar`

WORDS_PER_S = 2.5

# A call: the user asks for work, pauses, chats, asks a status question. Each line is one user sentence; the bench
# cuts it into windows and follows it with silent windows. "..." marks a mid-sentence pause (an unfinished p).
SCRIPT = [
    "Hey, can you add a unit test for the parser and run the whole suite?",
    "Also ... the one that checks empty input.",
    "How's your day going so far?",
    "Is the test run done yet?",
]


def windows(sentence: str, step_ms: int) -> list[tuple[str, float]]:
    """(words, p) per window: words at WORDS_PER_S, p low mid-sentence, 0.3 at '...', 0.9 on the last window."""
    words = [w for w in sentence.split() if w != "..."]
    per = max(1, round(WORDS_PER_S * step_ms / 1000))
    out: list[tuple[str, float]] = []
    pause_after = sentence.split().index("...") if "..." in sentence.split() else -1
    for i in range(0, len(words), per):
        chunk = words[i:i + per]
        last = i + per >= len(words)
        p = 0.9 if last else 0.3 if pause_after != -1 and i <= pause_after < i + per else 0.1
        out.append((" ".join(chunk), p))
    return out


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8092")
    ap.add_argument("--model", default="default")
    ap.add_argument("--step-ms", type=int, default=300)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--silence", type=int, default=6, help="silent windows after each sentence")
    ap.add_argument("--out", default="/tmp/live-vibe-microturn-bench.jsonl")
    a = ap.parse_args()

    import httpx
    from sidecar import front, microturn

    system = front.FRONT_PROMPT_EXPERIMENTAL.format(workspace=str(HERE.parents[4]))
    client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))
    stepper = microturn.MicroStepper(client, a.url, a.model, system,
                                     threshold=a.threshold if a.threshold is not None else microturn.THRESHOLD,
                                     sampling=front.LlamaCppBrain.SAMPLING)
    await stepper.resolve_tokens()
    print(f"action tokens: listen={sorted(stepper.listen_ids)} speak={sorted(stepper.speak_ids)}")

    history: list[dict] = []
    steps: list[microturn.MicroStep] = []
    speaking = False
    plan: list[tuple[str, float | None]] = []
    for sentence in SCRIPT:
        plan.extend(windows(sentence, a.step_ms))
        plan.extend([("", 0.9)] * a.silence)
    rows: list[str] = []
    t_all = time.monotonic()
    for i, (words, p) in enumerate(plan):
        window = microturn.window_message(words, p, speaking)
        try:
            s = await stepper.step(history, window)
        except Exception as e:  # noqa: BLE001
            print(f"step {i}: failed {type(e).__name__}: {str(e)[:160]}")
            return 1
        steps.append(s)
        speaking = s.action == microturn.SPEAK and bool(s.say)
        if microturn.keep_in_history(window, s.action, s.say, s.delegate):
            history.append({"role": "user", "content": window})
            if s.action == microturn.SPEAK:
                history.append({"role": "assistant", "content": json.dumps(
                    {"action": s.action, "delegate": s.delegate, "say": s.say}, ensure_ascii=False)})
        pl = "n/a" if s.p_listen is None else f"{s.p_listen:.2f}"
        extra = f"  say={s.say!r}" if s.say else ""
        extra += f"  delegate={s.delegate!r}" if s.delegate else ""
        print(f"{i:3d} {s.decide_ms:5.0f}+{s.speak_ms:4.0f} ms  P(listen)={pl:>4}  {s.action:6s}  "
              f"cache={s.timings.get('cache_n', '?')} prefill={s.timings.get('prompt_n', '?')}  {window[:48]!r}{extra}")
        rows.append(json.dumps({"i": i, "window": window, "action": s.action, "p_listen": s.p_listen,
                                "decide_ms": s.decide_ms, "speak_ms": s.speak_ms, "say": s.say,
                                "delegate": s.delegate, "timings": s.timings}))
    Path(a.out).write_text("\n".join(rows) + "\n", encoding="utf-8")
    await client.aclose()

    def pct(xs: list[float], q: float) -> float:
        if not xs:
            return float("nan")
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(q * len(xs)))]

    silent = [s.decide_ms for s in steps if s.action == microturn.LISTEN]
    spoken = [s.total_ms for s in steps if s.action == microturn.SPEAK]
    print(f"\n{len(steps)} steps in {time.monotonic() - t_all:.1f} s, step budget {a.step_ms} ms, "
          f"history {len(history)} messages")
    if silent:
        print(f"listen steps: {len(silent)}  decide p50 {statistics.median(silent):.0f} ms  p95 {pct(silent, 0.95):.0f} ms")
    if spoken:
        print(f"speak steps:  {len(spoken)}  total p50 {statistics.median(spoken):.0f} ms  p95 {pct(spoken, 0.95):.0f} ms")
    late = sum(1 for s in steps if s.total_ms > a.step_ms)
    print(f"steps over budget: {late}/{len(steps)}")
    print(f"written: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
