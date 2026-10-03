# What GPT-Live teaches the live-vibe voice path

Status: proposal, 2026-10-03. No code changes yet.

Sources, read in full on 2026-10-03:

- [E] https://openai.com/index/continuous-voice-interaction-with-gpt-live/ , dated August 3, 2026, titled "How we built a realtime system for responsive voice AI in six months".
- [L] https://openai.com/index/introducing-gpt-live-1-in-the-api/
- Guides under https://developers.openai.com/api/docs/guides/ : [P] `live-prompting.md`, [D] `live-delegation.md`, [M] `live-migration.md`, [V] `realtime-vad.md`, [C] `voice-latency-cost.md`, [W] `voice-websockets.md?api=live`, [S] `live-conversations.md`.
- [O] Oh My Pi `packages/coding-agent/src/live/{protocol,transport}.ts`. [U] Kyutai Unmute `unmute/unmute_handler.py`.

## 1. What the post says, and what it leaves out

1. **No turn detector.** [E] "GPT-Live ... removes the turn detector from the audio path. Its voice model is full-duplex".
2. **Delegation off the media path.** [E] "A slow tool call or backend service can delay its own result, but cannot stall the flow of media."
3. **Prefilled backend.** [E] The app server "prefills it with the initial conversation context, ensuring the prompt has been fully processed prior to the first delegated request", then keeps it with prompt caching.
4. **The KV cache is managed.** [E] Compaction "invalidates the model's key-value (KV) cache ... Rebuilding that state requires a new prefill". They compact on a second instance "While the original model instance keeps chatting".
5. **Two transcript views.** [E] "a speculative view of the current state and an authoritative record". An overlapping "mm hmm" is not its own message.
6. **Commentary.** [D] `session.commentary.append` is "for results GPT-Live should say aloud; it is trained to paraphrase the text". `session.thinking.append` carries facts it can use "without saying them". Each append is "limited to 500 tokens". `session.delegation.created` carries no task text.
7. **Backchannels and interruptions are prompt text.** [P] "Backchannel policy: Use moderate backchannels. Acknowledge naturally without competing with the main response." "Interruption policy: Stop speaking when the user interrupts." "A brief listening sound is different from taking over the user's turn."
8. **Thinking pauses.** [P] optional control: "Keep listening while the user pauses to think." [L] quotes Speak cutting "interruptions during thinking pauses by almost 80%". [V] on semantic VAD: "When the probability is low, the model will wait for a timeout, whereas when it is high, there is no need to wait."
9. **Speculation.** [D] "Start a speculative lookup when enough information is available ... Discard outdated results".
10. **Playback is the client's.** [W] output audio comes "without timing fields or an output-audio-done event."

Not disclosed anywhere: a per-tick decision loop. No tick or frame size, no action set; "speak / listen / pause / interrupt / delegate" is our framing. Also absent: model size, training, turn-taking latency (only frame delivery, "p95 matching the previous system's p50"), how backchannels or pauses are produced, how commentary merges into speech, how a client flushes queued audio on barge-in, and semantic_vad's timeouts. Everything below the API is inference.

## 2. Mapping to the local path

Today's session (`sidecar.log`, 22 `kyutai turn end` lines, 14:26 to 14:49, main with `endSilenceMs` 1500; `front-server.log`):

- 17 semantic ends, `quiet_ms` mostly 480 (the 6-step floor), final s2 0.70 to 0.95.
- 3 cap ends at 1500 ms, final s2 0.64, 0.30, 0.03. Twice the model forecast more speech.
- 2 max ends at 30 s, final s2 0.01 and 0.00. Cut mid-sentence.
- The front's first ~20 requests hit the prompt cache (`f_keep` 1.0, prefill 12 to 30 ms). Then `f_keep` fell to 0.13 to 0.16 on every request: 3200 to 4300 tokens re-prefilled (200 to 310 ms) plus 270 to 500 ms saving the old cache entry. `Brain.begin` trims the oldest messages each turn once `HISTORY_MAX = 40` is reached, shifting everything after the system prompt. Item 4's invalidation, every turn.

| Mechanism (what the post implies: section 1 item) | Today | Change | Effect | Effort |
|---|---|---|---|---|
| Stable prefix (item 4) | `Brain.begin` trims per turn past 40 messages | Trim in one block, 40 to 20, after a turn ends; then a 1-token warm request with the new history while the floor is free | 0.5 to 0.8 s off time to first token on every turn past ~20 | S |
| (a) Probability-scaled wait (item 8) | main: semantic (s2, s05 > 0.6 for 6 steps, 400 ms drain) or the 1500 ms cap. Branch `worktree-agent-a7c52e7f74ebcfe94` (0975d1b): tiers 1000 / 3000 / 4000 ms by s2 EMA and `turn_tail` | Per step `wait = 400 + (1 - p) * 3600` ms, p = s2 EMA; end when `quiet_ms >= wait`. Unfinished tail caps p at 0.3, terminal punctuation floors it at 0.7; for p > 0.6 also require s05 > 0.6. p 0.95: 580 ms, 0.8: 1120, 0.6: 1840, 0.3: 2920, 0: 4000 | One rule for two triggers and three tiers; today's cap ends would have waited 1700, 2920, 3890 ms | S |
| (f) STT flush ([U], not the post) | Semantic end waits 480 ms of real time after the last piece, since text trails audio ~0.5 s | Unmute pushes `ceil(delay_sec / FRAME_TIME_SEC) + 1` zero frames at model speed. In `KyutaiTurns._step`, on the first step with p > 0.6 and s05 > 0.6, step `delay_steps + 1` zero blocks at once, collect trailing pieces, emit, then feed the queued mic frames | ~400 ms off each semantic end, minus flush compute (CUDA step time unmeasured) | S |
| (b) Speculative prefill (items 3, 9) | The front request starts at the utterance | With a stable prefix, prefill alone saves 12 to 30 ms. Speculate generation: when p crosses 0.6, start `LlamaCppBrain.respond` on the provisional text, speech and `_hand_off` held; a new piece cancels; an unchanged text releases at end of turn. No speculative Claude turns; their prompt cache is the prefill, so keep `RELAY_SECTION` byte-stable | Front prefill and decode (~5 ms/token up to `say`) hide inside the wait | M |
| (c) Interruption (item 7) | Barge-in after `BARGE_IN_WORDS = 3` pieces, ~1 to 1.5 s of overlap. `Voice.speak` keeps text in proportion to played samples | Two stages, as Unmute's pause < 0.4 test suggests. First piece, or s2 EMA < 0.4 while speaking, pauses the player (new `pause`/`resume` beside `cancel` in winplayer.py and the Rust player). Three words or a non-backchannel cuts; else resume. Heard text = the `cut` reply's `played` minus `LATENCY_S` (50 ms), snapped back to a word boundary | Stops in ~0.3 s; a user "mm-hm" no longer cuts or starts a turn | M |
| (d) Backchannels (items 5, 7) | None from the front | In code, not the LLM: a cached Kokoro "mm-hm" when the user has spoken > 6 s, s05 > 0.8 and s2 EMA < 0.4 (a pause, more coming); at most one per 8 s, none in the first 3 s or after a question. Silence when p is high, while a value is spelled, during a retelling. Off by default on open speakers | The user knows they are heard during long dictation | M |
| (e) Relay as commentary (item 6) | `report_brief` cut at `EVENT_CHARS` 6000 (~1500 tokens); reply `max_tokens` 400 | Cap the retold part at 500 tokens (llama-server `/tokenize`). `RELAY_SECTION` asks Claude to end with `Status: working, done, failed or cancelled` and a 3-sentence spoken report. The rest and worker progress join front history as silent context | Right status in fewer words; "how's it going" no longer needs a `HOLD` round trip | S |
| (g) Hard max (item 1) | `max_utterance_s` 30 cut 2 of 22 turns | At max, end only if p ≥ 0.6; else extend 10 s at a time up to 120 s | No mid-sentence cuts | S |

## 3. GPT-Live as `frontBackend: gptlive`

**Session.** `wss://api.openai.com/v1/live/sessions`, `Authorization: Bearer $OPENAI_API_KEY`. First message `session.start` with `model: "gpt-live-1"`, instructions, voice, `audio.format {"type":"audio/pcm","rate":24000}` (fixed, both directions) and `delegation: {"type":"client"}`. Wait for `session.started` [W].

**Audio.** `Listener` keeps the WSLg mic and AEC3. Cleaned 16 kHz frames are resampled to 24 kHz (`KyutaiTurns._resample`) and streamed as `session.input_audio.append`, no commits. `session.output_audio.delta` plays through the Windows player at 24 kHz, Kokoro's rate, so the echo reference is unchanged. `WinPlayer.play()` takes whole clips of known length, so it needs an open-ended streaming clip. Keep the local queue under 150 ms. No barge-in event exists, so stale audio drains only as fast as that queue.

**Delegation.**

1. Accumulate `session.input_transcript.delta` and `session.output_transcript.delta`; emit them as `transcript` lines.
2. On `session.delegation.created`, build the request from the user's transcript since the last delegation, after waiting up to 1 s for trailing fragments ([M]: "The notification may arrive before the full sentence is transcribed"). Emit the existing `delegate` event with `said`, keyed by `delegation.id`. register.tsx is unchanged.
3. Claude's `turn.complete` already POSTs `/event`. Send it as `session.commentary.append` with the oldest pending `delegation_id`, capped as in row (e); `delegation_id: null` when none is pending. Progress goes as `session.thinking.append`.
4. Segments GPT-Live answered itself go to Claude as `note`, as `pass_on` does. Model switch and goodbye stay regexes over the input transcript.

**Local versus remote.** Mic, AEC3, speaker, the mod and Claude Code stay local. Turn-taking, backchannels, interruption and the voice go to OpenAI, with the user's audio.

**Auth.** API key only. Oh My Pi [O] signals through the ChatGPT Codex backend with Codex OAuth, an attestation header, `User-Agent: Codex Desktop/...`, originator "Codex Desktop" and model `gpt-live-1-codex`, using private events (`delegation.context.append`). That impersonates a first-party client; we use the public API only. Worth borrowing: splitting long context into 500-unit appends.

**Cost.** Billing runs per second of open session, silence and backend time included [C]. At $0.05/min, $3.00 an hour; $24 for an 8-hour day left open, before Claude. Close after 120 s idle. Silero VAD reopens on speech, buffering the first words and seeding saved context into `input` [S].

| Failure | Fallback |
|---|---|
| No key | One warning naming `OPENAI_API_KEY`; run the llamacpp front |
| Session creation fails (429 included) | Warn, llamacpp front, retry at next `/livevibe` |
| Network loss mid-turn (socket closes before `session.closed`) | Cut playback; the local front takes the next turn. Delegations stay with Claude; results reach the local front through the same `/event` queue |
| `error` event during speech | Log; the session continues, that speech is cut [S] |

A warm fallback holds Kyutai (3.2 GiB), Kokoro (1.4 GiB) and llama-server (5.3 GiB); today's cold starts took 6.7 to 24.5 s. Default cold, with a "connection lost" clip rendered at setup; `gptliveStandby: warm` keeps the stack loaded. The docs gave no rate-limit numbers.

## 4. Build plan

Local:

1. **L1, one session: stable front prefix.** Block trim in `Brain.begin` (no orphaned tool result), a warm request after it, and a per-turn log line: end of turn, first token, first audio. Accept: over 30 turns, `f_keep` ≥ 0.9 except trim turns; a unit check on trim boundaries.
2. **L2: STT flush.** Accept: semantic `quiet_ms` < 200, flush time logged, no new split utterances in 20 turns.
3. **L3: probability-scaled wait and max extension**, replacing the tiers. Accept: unit table p to wait, replay of logged heads, no max cuts live.
4. **L4: relay as commentary.** Accept: retold status matches `Status:` on the front eval reports.
5. **L5: two-stage barge-in.** Accept: pause/resume checks with `--fake`; "mm-hm" resumes; cut latency logged < 400 ms.
6. **L6: speculative front.** Accept: a unit check that discarded speculation never delegates; median first token after end of turn drops.
7. **L7: backchannels**, behind a setting, default off.

GPT-Live:

1. **G1: spike script.** Mic to WebSocket to player, no delegation. Accept: 5 minutes of talk, `session.usage.updated` seconds match wall time, barge-in stop time measured.
2. **G2: streaming clips** in both players, with unit checks.
3. **G3: `GptLiveSession`** bridging client delegation to `delegate`, `note` and `/event`. Accept: selftest against a fake WebSocket server.
4. **G4: failover and idle close.** Accept: a dropped network hands the next turn to the local front; idle close records `session.closed` usage.
5. **G5: setting, README, setup check** (key present, one start/close).

ADR candidates, not written: GPT-Live as an optional front backend (audio egress, cost, fallback contract); the probability-scaled wait (supersedes the tiers if they land first); two-stage barge-in; the front history compaction policy.
