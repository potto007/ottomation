# live-vibe in T3 Code: a voice conversation on the environment's speech stack

Status: proposal, 2026-11-07. Fork-first: `potto007/t3code`, worktree `~/src/oss/t3code-live-voice`,
branch `feat/live-voice` from `upstream/main` e455200077. After a working demo, post the design to
upstream discussion #22091 for scope approval (their rule: no PR without prior approval).

Line references are against upstream e455200077.

## Goal

Press **Start voice** on a thread and talk. A small local "front" model answers small talk itself,
hands real work to the thread's agent, and reads back the result in a sentence or two. You can talk
over it. Setup should feel the same as dictation: the first press offers one download for any missing
model, then the call starts. You don't need an API key, a URL, or a server to manage.

## What upstream already gives us

| live-vibe piece | Upstream |
|---|---|
| Mic capture | Web worklet, 16 kHz mono Float32 in 0.5 s chunks (`apps/web/src/speech/pcmWorklet.ts`) |
| Audio transport to the environment | `/ws/voice`: media ticket, `AuthOrchestrationOperateScope`, binary PCM frames (`apps/server/src/speech/stream.ts:92-113`) |
| STT (Kyutai/Whisper) | `transcribe-cpp` 0.4.1 in a killable child process; streaming Parakeet/Nemotron with committed and tentative text (`native.ts:81-98`) |
| Silero VAD | Loaded in the worker; `audio.ts:98-124` tracks `inSpeech` but never emits it |
| Model store | `SpeechModelStore`: pinned catalog, Range resume, sha256, `.verified` |
| Consent | Per-environment audio grant (`apps/web/src/speech/voiceConsent.ts`) |
| Hardware, unload | `speechAcceleration` (Auto GPU then CPU once), `speechModelUnloadTimeout` |
| Send to a thread | `ThreadManagementService.sendToThread` (`ThreadManagementService.ts:815-903`), the path `t3_thread_send` uses |
| Watch a run | `waitForThread`, `Orchestrator.streamStoredEventsFrom({threadId, eventType: "run.updated"})` |

## What the dictation stream cannot do (and the fix)

1. One stream per socket. `finish` closes it, and a second `start` fails (`stream.ts:45`, `:50-54`). **Fix:** a
   separate route, `/ws/voice/conversation`. Dictation stays untouched.
2. 10-minute socket cap (`stream.ts:83`), 5-minute audio cap per stream, and an unbounded `received[]` replay
   buffer (`SpeechService.ts:769`, `:816-823`). **Fix:** reset the worker after each utterance (the worker
   already has `reset`, `native.ts:168-172`) and clear the buffer. The 5-minute cap then applies per
   utterance. The session closes after a configurable idle period, default 10 minutes.
3. VAD events never leave the worker. **Fix:** add `speech` (`start|end`, sample offset) to the feed reply.
   The conversation always loads VAD for events, whatever `speechSilenceFiltering` is set to.
4. Only one native operation at a time (`SpeechService.ts:198-200`). **Decision:** a call holds the STT slot,
   as a dictation stream does. Dictation elsewhere on the environment gets `speech_busy` with
   reason `voice_call`. The front and TTS run in their own workers, outside this slot.
5. JSON-only replies to the client. **Fix:** the conversation socket also carries binary TTS frames.
6. Lockstep flow control, with at most 2 chunks in flight. This is fine for 0.5 s chunks if the worker keeps
   up. We log the lag, and if it falls behind, the server drops tentative decoding rather than stalling capture.

## Architecture

```
client (web)                              environment
 mic --AEC--> worklet 16k --PCM-->  /ws/voice/conversation
                                      VoiceConversationService (one per call, bound to a thread)
                                        +- STT worker (existing) -> text + VAD events
                                        +- TurnDetector (pure)   -> utterance-end / barge-in
                                        +- Front worker (node-llama-cpp, GGUF, JSON schema)
                                        |     reply | delegate | note | none
                                        +- ThreadManagementService.sendToThread (delegate)
                                        +- run.updated watcher -> last assistant_message -> front summary
                                        +- TTS worker (kokoro-js, sentence chunks)
 player worklet 24k <--PCM frames-----+
 playback cursor ----acks------------->  (words actually played; echo guard)
```

All three models live on the environment, so remote, relay and tunnel clients behave the same as local
ones. Audio is about 256 kbps up and 384 kbps down as raw Float32. That is fine on a LAN or tailnet. Opus is
a later optimization.

### Wire protocol (`packages/contracts/src/voiceConversation.ts`)

Client to server: JSON `start {threadId, options}`, `played {seq, sample}`, `mute`, `unmute`, `stop`.
Binary frames are mic PCM.
Server to client: JSON `ready`, `caption {who: "you"|"voice", committed, tentative}`, `state`
(`listening|thinking|speaking|delegated`), `speak {seq, text, sampleRate}`, `cut {seq}` (barge-in),
`delegated {runId, text}`, `error`. Binary frames are `[seq u32 LE][Float32 PCM]`, at most 0.5 s each.
`cut {seq}` drops that seq and every earlier one. A synthesis failure also sends `cut`. Nothing marks
the end of a `speak`, so `state` carries "done speaking".

### Turn taking (`TurnDetector`, pure, table-tested)

Port live-vibe's rules. Parakeet has no semantic end-of-turn score like Kyutai's s2, so:
- An utterance ends after a VAD end plus a wait: 800 ms after finished punctuation, 1500 ms by default, or
  3000 ms when the text ends unfinished (a trailing comma, `and`, `so`, `the`). Measured 2026-11-07:
  streaming Parakeet adds a sentence's period only once the next sentence starts, so the 1500 ms
  default ends most turns. live-vibe's 3000 ms left a 2.5 s pause unended.
- An utterance cut off mid-sentence waits up to 2.5 s for the rest, and the pieces go to the front as one.
- Barge-in: VAD start while speaking pauses playback (`cut`). If the words form only a listening sound
  ("mm-hm", "yeah", "okay", "got it") and end within 1.2 s, playback resumes from the cursor, and the
  words reach the agent as a note.
- Echo guard: words found in what was played in the last 2.5 s (taken from `played` acks) don't
  barge in. The browser AEC removes most echo, and this catches the rest.

### Front model

- A worker child process running `node-llama-cpp` 3.22.3, with the same packaging treatment as the speech
  worker: inline entry, SEA re-exec, ASAR unpack.
- The model is a pinned GGUF in a catalog next to the speech catalog, downloaded by `SpeechModelStore`.
  The default is Qwen3-4B-Instruct-2507 Q4_K_M. On 50 voice-style prompts with CUDA it chose the right
  action 45 times (all misses were note vs delegate), with a 150 ms median decision (p90 226 ms) and about
  4.3 GiB of VRAM. Qwen3-1.7B scored 42/50 and also answered project questions itself. Gemma 4 E2B
  spoke on noise. Qwen3.5 (hybrid) got no KV reuse.
- The budget needs a GPU: on CPU a decision takes 1.3-2 s. llama.cpp spins its threads, so they are capped
  at half the cores; uncapped, beside STT and TTS, one decision took 51 s.
- Packaged builds ship CPU plus Vulkan on Linux and Windows x64, and Metal on macOS arm64. CUDA (200-416 MB,
  needs a CUDA runtime) is pruned. NVIDIA users on native Linux and Windows run on Vulkan, whose speed on
  a real GPU is unmeasured. On WSL, Vulkan is a deviceless software driver, so the worker falls back to
  CPU. node-llama-cpp's binary probe forks with the parent's execArgv, which broke both the `-e` worker
  and the SEA until it was cleared.
- Output is forced through a JSON schema:
  `{action: "reply"|"delegate"|"note"|"none", speech: string, task?: string}`.
- Prompt and history follow live-vibe's lessons:
  - Keep a stable prefix. Trim once, from 40 messages to 20, after a turn ends, so the KV cache survives.
  - Cap relayed results at 500 tokens.
  - Summaries are at most two sentences and a question.
- Advanced override (server setting `voiceFrontModel`): an OpenAI-compatible URL and a model name. On the
  shared 32 GB box, point it at the `:8080` router instead of loading a second LLM.

### Delegation

- `sendToThread({mode: "auto", createdBy: "user", creationSource: "web"|"mobile", text})`:
  - `auto` steers an active run or starts one.
  - The text is the user's words plus the front's reading.
- Watch `run.updated` from `getThreadEventSequence(threadId)`.
- On a terminal status, read the last non-streaming `assistant_message` for that run
  (`getProjectThreadRecords(..., {turnItemTypes: ["assistant_message"], turnItemRunIds: [runId]})`),
  have the front summarize it, then speak it. A result is read back, never acted on.
- Results and the user's words go to the front as separate turns. Batched into one decision, the front
  dropped the readback.
- A delegation runs the thread's real provider with the user's words. In testing, a spoken "open a pull
  request and ask Julius to review it" made a real Codex thread do exactly that upstream. Outward actions
  in voice-started runs may need a spoken confirmation (unresolved). Every test harness checks in code that
  its thread is on Replay (dev) before delegating.
- Delegated messages are ordinary thread messages, so they show in the timeline, and history and
  checkpoints behave the same.

### TTS

- A worker child process running `kokoro-js` 1.2.1: ONNX through onnxruntime-node, and a WASM phonemizer,
  so there is no espeak-ng to install. transformers.js runs local-only, with `sharp` and
  `onnxruntime-web` overridden out.
- The model is `onnx-community/Kokoro-82M-v1.0-ONNX@1939ad2a8e`, fp32 (326 MB), stored as a multi-file
  `directory` entry in the same store and catalog. fp32, not q8: q8 was 4.7x slower on CPU at the same
  WER, and fp16 is broken.
- CPU only. Measured 2026-11-07 on the Ryzen 7 9700X: warm first sentence about 315 ms, RTF 0.13, cold
  start 1.43 s, about 0.9 GiB RAM. CUDA gave a 61-74 ms first sentence, but it needs a 343 MB CUDA
  provider `.so` plus cuDNN 9 and CUDA 12 libraries that don't ship.
- The worker unloads after 5 idle minutes. PR 4 holds it for the length of a call.
- Text is split into sentences. The first sentence is rendered and sent while the rest render. Kokoro
  renders a whole piece before sending, at about 0.3 RTF on a loaded CPU, so a first sentence over 40
  characters renders its opening clause alone. That cut the median first-piece render from 829 ms to
  568 ms (long sentences 1796 to 858 ms), with no gaps after it.
- CPU Kokoro varied 3x with load from the agent's own builds (load average 10-33). A GPU Kokoro is worth
  another look.
- Markdown is stripped at the single TTS entry point (`voice/speakable.ts`), so replies, readbacks and
  confirmations all speak plain text.
- The voice is a client setting (`voiceTtsVoice`).

## Surfaces

- Web and desktop: composer **Start voice** button, a command palette entry, and a keybinding (unbound by
  default; open PR #29394 proposes F8 for its Codex call). A call strip shows state, captions, and
  Mute/Stop. Capture uses
  `getUserMedia` with `echoCancellation`, `noiseSuppression` and `autoGainControl` set to true. Today's
  capture sets no constraints (`browserPcmCapture.ts:65-67`).
- Settings > Voice: a **Conversation** section with voice, front model (download/use, or URL),
  idle timeout. It reuses the `CleanupModelRow` and model-browser patterns.
- Mobile: not in the first cut. It needs a PCM event surface in `t3-voice-transcription`,
  `VOICE_COMMUNICATION` capture for AEC, and a play-and-record audio session; `audioSession.ts:50`
  refuses playback while recording. Until then, the button is hidden with "Voice calls are available
  on web and desktop."
- Providers: provider-agnostic. Delegation sends an ordinary message, so it works on every adapter.
- Agents/MCP: none. A call needs a human at a mic.
- Reverse states: Stop ends the call. Closing the tab, losing the connection, revoking the grant, or
  idling out ends the call and frees the STT slot.

## Fork PR stack

1. `feat(speech)`: conversation socket with per-utterance reset, VAD events, `TurnDetector`, idle close.
   Tests: TurnDetector table, socket lifecycle with a fake worker.
2. `feat(speech)`: Kokoro TTS worker, catalog entry, model store, binary downlink framing.
3. `feat(speech)`: front-model worker (node-llama-cpp), catalog, schema output, URL override setting.
4. `feat(voice)`: `VoiceConversationService` (delegation, run watching, summaries, barge-in and echo
   guard, notes). Test with the Replay (dev) provider, not real turns.
5. `feat(web)`: client controller in client-runtime (platform object), AEC capture, player worklet with
   cursor acks, call strip, settings, palette, keybinding. Then one integrated pass in a real browser.
6. `feat(mobile)`: later.

## Findings from the real-model runs (2026-11-07)

- Speech recognition ran on CPU. `transcribe-cpp` 0.4.1 ships only CPU and Vulkan builds for Linux, and WSL2
  has no Vulkan driver for the NVIDIA card (Ubuntu 24.04's Mesa lacks `dzn`), so it lists no GPU. CPU is
  fast enough: 0.5 s chunks decode in about 150-270 ms at p95 on the Ryzen 7 9700X. CUDA itself works in
  WSL (`/usr/lib/wsl/lib/libcuda.so`), which matters for onnxruntime (TTS) and node-llama-cpp (front).
- Utterance text comes back without its final punctuation, and Parakeet misheard "tests" as "test" on a
  synthetic voice. The front model has to tolerate both.

- Parakeet's streaming text lags speech by 1.2-1.9 s, and with silence filtering the last word only
  appears at finalize, so speculation has to finalize at the pause. Finalize also adds `.` or `?` to
  mid-sentence pieces, so the detector reads pause text with trailing punctuation stripped.
- Streaming Parakeet holds a short answer ("No.", "Yes, go ahead.") until more speech arrives. A voiced
  turn now ends after the pause even with empty text, and the finalize flushes the words.
- Where the extra ~0.5 s of endpointing went: Silero lags 100-140 ms, 0.5 s chunks quantize 0-300 ms, the
  endpoint chunk's decode takes 171-239 ms, and a serial finalize took 175-246 ms before the front. The last
  two are now off the critical path. PR 4's harness sent each chunk at its first sample, so its 2.08 s was
  about 0.5 s optimistic.
- Pause sweep over 15 fixtures (14 with mid-sentence pauses): at 0.5 s chunks, 1500 ms without speculation
  cut 0/13 off at a 2082 ms median; 1200 ms with speculation cut 1/13 at 1521 ms; 1000 ms cut 2/13 at
  1381 ms. 250 ms chunks save another 30-90 ms at equal cut-off.
- The front sometimes claims in a reply to have done outward work ("I've addressed the reviewer's comment")
  without sending anything. It needs a prompt eval.

## Open risks

- Native packaging: PR 3 proved `node-llama-cpp` under the Node SEA archive and Electron 44 (a hand-packed
  `app.asar`; a full desktop build needs `libsecret-1-dev`). PR 2 proved `onnxruntime-node` on
  Linux (SEA and Electron 44) with `asarUnpack` of `onnxruntime-node/bin` and pruning of other
  platforms' binaries (about 190 MB to 45 MB). Windows and macOS are unverified.
- Does Chromium AEC in Electron cancel WebAudio worklet playback from the same renderer? PR 5 could not
  measure it (fake mic only); it needs a real test on speakers. If not, the echo guard has to carry it (live-vibe's 3-word fallback).
- Latency from end of speech to first audio, target under 1.2 s, measured 2.08 s end to end (PR 4, idle
  box, front on CUDA): endpoint 1.43-1.80 s after speech ends (the 1.5 s default pause plus 0.5 s mic
  chunks; the 800 ms finished-sentence pause rarely applies), front decision 330-530 ms, then first audio
  320 ms for "On it." and 460-710 ms for readbacks. The pause is the floor, not inference. Options: start
  the front decision and first render speculatively once silence begins and discard them if speech
  resumes, shorten the default pause or the chunk, or let the client set `endSilenceMs`.
- Package size: the CLI archive is 342 MB compressed (864 MB extracted). node-llama-cpp adds 146 MB and
  onnxruntime-node 208 MB. Candidate fix: a voice runtime pack downloaded on demand like the models,
  which would also allow a CUDA build per machine.
- Licensing: `phonemizer` 1.2.1 declares Apache-2.0 but embeds a WASM build of eSpeak NG (GPL-3.0-or-later).
  It cannot ship inside MIT builds as is. Candidate fixes: download it on demand into the store, outside
  the shipped artifact, or replace it with a permissive G2P (CMUdict plus a small model), which risks
  pronunciation because Kokoro was trained on eSpeak-style phonemes.
- On the first, cold call the web page's React tree stopped committing at delegation (SyncLane pending,
  nothing scheduled, no errors). It did not recur in 4 more calls.
- VRAM on this box: Parakeet Q8 about 0.7 GiB plus a front model of about 3 GiB, next to `:8080` and `:8084`.
  Default to the URL override here.
- Upstream fit: one PR stack is large. Upstream will want the scope approved in #22091 first.

## Decisions (Paul, 2026-11-07)

- The voice runtime (onnxruntime-node, kokoro-js with phonemizer and eSpeak NG, node-llama-cpp and its
  binaries) moves into a voice pack downloaded on demand, like the models. Shipped builds carry none of it,
  which settles the eSpeak NG GPL question and the size, and lets the download pick CUDA per machine.
  The download UI credits eSpeak NG's GPL-3.0-or-later license.
- Outward-facing delegations (PRs, pushes, comments, messages, deploys) from a call need a spoken yes.
  The front flags them and a keyword backstop catches the rest; no, silence, or barge-in cancels.
  Built (`feat/live-voice-confirm`): either signal holds the delegation, or a note that steers a running
  run, and the voice asks `Send "<task>" to the agent on <thread>?`. Only an explicit yes sends; "okay" and
  "mm-hm" never confirm; no, 8 s of silence, or new words cancel, and new words become a fresh turn.
  Measured on 40 labelled utterances with Qwen3-4B: the front's flag has precision 0.86 and recall 0.95
  (it flags local git verbs too), and front OR backstop reaches recall 1.00 at precision 0.87.
- Latency: start the front decision and first render speculatively once silence begins, discard them if
  speech resumes, never run side effects early, then retune the pause defaults from measurements.
  Built (`feat/live-voice-latency`): the detector marks a pause at 300 ms of silence, and the worker's
  `segment` op finalizes the words so far without resetting audio state. The front decision and the first
  render run then and are committed only at the endpoint with the same words and context. The default
  pause dropped from 1500 to 1200 ms. End of speech to first audio went from a 3351 ms median to 1734 ms
  on the server path, and from 3338 to 2201 ms in the browser, both under load average 7-15.

## Unresolved

- Follow-ups after the voice pack: a client pause setting (Short 1.0 / Normal 1.2 / Long 1.5 s), 250 ms mic
  chunks, and starting the first render on the decision's first sentence (saves about 100-300 ms under
  load). The floor is then the pause plus chunk quantization.

- Whether a call should open from the sidebar (new thread by voice) or only on an existing thread.
