# The micro-turn front

Status: in progress, started 2026-10-12. Increments 1 (the decision step and its bench) and 2 (the loop, behind
the `microturnMs` setting) are in this branch; 2 has unit checks and no live session yet.

## Why

`/livevibe` is a cascade: Kyutai STT, a 4B text front on llama-server, Kokoro. The front speaks only after Kyutai
ends the user's turn, and the sidecar fakes duplex around that with speculation, code-side backchannels and a
pause/resume barge-in. GPT-Live and Thinking Machines' interaction model removed the turn detector by training a
duplex model from scratch. The DuplexCascade paper ([arXiv 2603.09180](https://arxiv.org/abs/2603.09180)), and
[Codacus's video](https://www.youtube.com/watch?v=shJLycXjhVY) reproducing it on a text LLM, remove it another way:
the text LLM takes a turn every few hundred milliseconds and answers with words or with `listen`. Time-slicing, as a
CPU core does.

The alternative, a duplex speech model as the front, was tried on 2026-10-12 with NVIDIA PersonaPlex 7B (notes in
`~/src/personaplex-trial/NOTES.md`). It turns in real time at 280 ms end to end, but it cannot voice Claude's text
verbatim without training (forced tokens come out as other words) and follows a delegation instruction about 3 times
in 4. Claude's side would still have been today's text machinery. So the text front stays, and it learns to time-slice.

## The loop

Every `step_ms` (a setting; 400 ms to start, 200 ms once measured), the sidecar sends the front one user message:

```text
[voice] <the words Kyutai transcribed in this window>        # or
[no voice]                                                   # nothing new
[p=0.8]                                                      # Kyutai's end-of-turn probability, as text
[speaking]                                                   # appended while the front's own voice is playing
```

The front answers one JSON object, action first so the decision is known after a few tokens:

```json
{"action": "listen" | "speak", "delegate": "...", "say": "..."}
```

`listen`: nothing is spoken; if the voice is playing, it is cut (the model heard the user and chose to stop).
`speak`: `say` goes to Kokoro as it streams; `delegate` hands off as today. The next window continues or stops.

Silent turns (`[no voice]` in, `listen` out) are never committed to the history. Only windows where the user or the
front said something are kept, so the history grows with the conversation, not the clock, the prompt cache keeps its
prefix, and the "thirty listens in a row" bias from the video never forms (he hit it at 30 turns, about 20 s).

## The listen-probability rule (video, 11:00 to 13:00)

His model was 73% "keep talking" and stopped anyway: `listen` is one token, "keep talking" is many first words, each
with a small share, so `listen` wins the argmax at 27%. The fix is a rule outside the model: read P(listen) each
turn; at 0.30 or more, listen; below, take `listen` off the table and the model must speak. 0.50 made it finish the
user's sentences.

Ours: one `/completion` request on the templated prompt plus the prefix `{"action": "` with `n_predict` 1 and
`n_probs` 10. P(listen) = p(listen) / (p(listen) + p(speak)) from the raw top logprobs (a token outside the top 10
counts as 0). At or above `listen_threshold` (0.30) the step ends there. Below it, a second request runs the full
JSON turn with `logit_bias` blocking the `listen` token, on the same cached prefix.

## Budget

| Stage | Expected | Source |
|---|---|---|
| Prefill of the new window on a cached prefix | 12 to 30 ms | `front-server.log`, `f_keep` 1.0 |
| Decision request (1 token) | about 10 ms after prefill | to measure (bench) |
| Speaking turn, first `say` token | about 5 ms a token, 6 tokens of JSON before `say` | to measure |
| Kyutai text lag behind audio | about 500 ms, 80 ms steps | known |

A silent step should cost about 50 ms and a speaking step about 100 ms, so 200 ms steps are plausible on the 5090
with Kyutai and Kokoro resident. Below about 200 ms the window is usually empty: Kyutai is the floor, not the LLM.

## Bench, 2026-10-12 (increment 1, prompt only, no LoRA)

`eval/microturn_bench.py` against the managed b11146 llama-server, Qwen3-4B-Instruct-2507 Q4_K_M, 16k context, on
the RTX 5090; 58 windows of the scripted call, each window's words at 2.5 words a second.

| Step | Listen steps (decision only) | Speak steps (decision + whole JSON turn, not streamed) | Over budget |
|---|---|---|---|
| 300 ms | p50 17 ms, p95 36 ms | p50 168 ms, p95 190 ms | 1 of 58 |
| 200 ms | p50 16 ms, p95 50 ms | p50 157 ms, p95 186 ms | 1 of 58 |

Latency is settled: a silent step costs about 20 ms, a speaking step under 200 ms even with the whole turn awaited,
and the prompt cache holds (`prefill` 1 to 30 tokens, `cache` the rest) because silent windows are not committed.
200 ms steps are affordable; Kyutai's lag is the floor, as expected.

Behaviour, with the prompt alone, is the video's finding at 11:00 reproduced. The rule reads P(listen) well while the
user starts a sentence (0.94, 0.95, 0.81 on "a", "unit", "for"), then one window reads 0.18 on "parser" and the front
speaks mid-sentence. From there every window is `speak` at P(listen) 0.00 while the user is still talking: it echoes
the user's words back as `say` ("and add a unit test for the parser.", "run the unit test for the parser."), hands
off the same delegation on every window, and on the status question answers from nothing ("the parser test suite is
done."). Three causes, each with its fix in the next increments:

1. **Speak bias from its own history.** Every spoken window was committed, so after the first `speak` the history is
   a run of speak turns with `[speaking]` on the input. The mirror image of his listen bias. Fix in increment 2: commit
   a speaking run as one assistant message (its sentences joined) when it ends, not one per window.
2. **No notion of the floor.** The model has p and `[speaking]` as text and does not use them. Fix in increment 2 as
   sidecar rules, not model judgement: while the user's windows carry words and p is below 0.6, a `speak` may only be
   a backchannel; a delegation is taken only on a window with p at or above 0.6, once per utterance; `say` that
   repeats the window's words is dropped. Then in increment 3 as training data, which is what he needed too.
3. **Echo.** The front treats the window as text to transform. A prompt line helps little; the LoRA data shows the
   right behaviour (listen through a sentence, answer after it).

So the step and the measurements are in; the loop goes behind a setting with the floor rules; the LoRA is what makes
it good.

## Increments

1. **Decision step and bench** (this branch). `microturn.py`: the window message, the two-request step, the rule,
   the commit rule; unit checks on the pure parts; `eval/microturn_bench.py` replays a transcript cut into windows
   against a running llama-server and reports P(listen), the decision and the latency per step. No mic, no training.
   Accept: p50 silent step under 60 ms and speaking step under 120 ms at 16k context; the bench runs from the eval
   README's launcher.
2. **The loop in the sidecar** (`microturn.MicroLoop`, driven by `FrontSession`), behind `microturnMs` (0 = off;
   llama.cpp front and the Kyutai recognizer). What changed from the bench after its results:
   - The window carries the whole utterance so far (`[voice] can you add a unit test [p=0.1]`), not the step's
     fragment, so the model reads a sentence. When the recognizer ends the utterance, the next window asks on its
     whole text with the turn taken as over (`[p=` at least 0.6), and that window then joins the history as the
     user's message; a speaking run joins once, when the voice finishes, with what was heard. Silent windows are
     never committed. (The first live session, 2026-10-12, showed why the final window matters: with `[no voice]`
     after the end, the model listened through every utterance and Claude got the words only as notes.)
   - The stepper resolves the `listen` and `speak` token ids on its first decision. (Same session: the loop never
     called it, every reading was `n/a`, and the rule's fallback for no reading is listen.)
   - A hand-off is never silent: the prompt asks for a few words that name the task, the echo rule does not apply to
     a delegating step's `say`, and a delegation with no `say` speaks `HANDOFF_SAY` ("On it.").
   - While a run plays, the model's context ends with the run so far as a provisional assistant message, so a step
     continues the answer instead of starting it again; a `say` that repeats the run (the echo test against the
     run's text) is not spoken and ends the run. (Second live session, 2026-10-12: with the run invisible until it
     ended, the model restarted its one-sentence answer on every `[no voice] [speaking]` window, 192 times, and
     every copy was queued on the voice.)
   - Floor rules, in the sidecar: while the user talks (p below 0.6, utterance open) only the decision request runs
     and a `speak` is held (backchannels from the model come with the LoRA); a `say` that echoes the user's words is
     dropped; a delegation is taken once per utterance, after it ends, with the user's words as `said`.
   - A `listen` while the voice plays and the user has spoken cuts the voice; a `listen` once the voice is quiet ends
     the run. The sidecar's own pause/resume barge-in stays as the safety net for now; it and the model's cut can
     both fire, which is harmless.
   - An utterance the loop handed nothing off for reaches Claude as a note on the step after it, as `pass_on` does
     after an utterance-level turn; a delegation drops the note.
   - Idle: after IDLE_SKIP (3) silent windows with nothing playing, the loop stops asking until a word or a result.
   - Announcements (Claude's results) still run as utterance-level turns; the loop skips its ticks while one holds
     the floor. Speculation, code-side backchannel cues and the continuous wait are bypassed while the loop is on.
   Accept: the relay eval still passes with the setting off (it exercises the utterance-level path); a live session
   at 300 ms logs `micro step:` lines with no run of late steps. Not yet run live.
3. **Training data and the timing LoRA.** Synthetic calls from a local model, cut into windows at the loop's
   `step_ms`, with pauses, interruptions, fillers and STT errors; balanced across listen / start / stop / filler,
   each of which produced a specific failure in the video (20 repeated warnings; 950 fillers against 164 expected;
   3-word stalls). Train a LoRA on the 4B front; `--lora` on the managed llama-server. Accept: a held-out set for the
   delegate decision, not only turn-taking.

Settled elsewhere: the per-token rule and the JSON turn replace nothing in the Claude relay; `RELAY_SECTION`, the
`Status:` line and the announcer are unchanged.
