# live-vibe

Voice and director modes for Claude Code, as a function-hook plugin (a mod) with a Python audio sidecar
(`bin/sidecar/`, started by the mod with `uv run --script bin/sidecar/main.py`). The sidecar talks to the mod over
stdout and a loopback HTTP port guarded by a per-run token.

| Command | What it does |
|---|---|
| `/live` | Full-duplex voice with Claude. The mic stays open; you can talk over an answer to cut it off. What you say goes to Claude, and Claude's final answer is read aloud. Saying "switch to sonnet" runs `/model`. |
| `/vibe [request]` | Director mode. Claude reads and directs worker subagents; every other tool is denied on the main loop. |
| `/livevibe` | You talk with a small, fast voice model (the front). It chats with you directly, hands real work to Claude in vibe mode, and tells you the result in a sentence or two when it comes back. `/livevibe model [name]` lists or picks the front model; `/livevibe url [url]` points it at a server you run (`/livevibe url managed` goes back to the built-in one). |

`/live` and `/livevibe` turn each other off. `/livevibe` off puts vibe mode back the way it was.

## Requirements

- Claude Code 2.1.287 or newer (mods are on by default from that build).
- [uv](https://docs.astral.sh/uv/). The sidecar is a PEP 723 script; uv installs Python 3.12 and its packages on first use.
- A microphone and speaker on the machine running Claude Code. A headless or SSH-only box cannot use voice; `/vibe` still works there. Headphones are best, because the mic stays open while the assistant speaks. On open speakers, the sidecar cancels the assistant's echo from the mic (WebRTC AEC3, through the `livekit` package), keeps barge-in strict until the speaker has gone quiet, and drops utterances that only repeat what it just said. Pick devices with the `mic` and `speaker` settings; `--list-devices` (below) shows the choices.
- Per OS:
  - **macOS (Apple silicon):** `brew install espeak-ng` for Kokoro. Kyutai STT runs on MLX.
  - **Linux:** `sudo apt install espeak-ng libportaudio2`. On x86_64, uv also installs PyTorch with CUDA and Kyutai's `moshi` package (about 3 GB the first time), and Kyutai STT runs on an Nvidia GPU (about 3.2 GB of VRAM; it needs 5 GB free when it starts, else Whisper runs). Without a usable GPU, Whisper runs instead: on an Nvidia GPU when CTranslate2 finds one, CUDA 12 cuBLAS and cuDNN 9 are installed and the GPU has room, otherwise on the CPU. Kokoro runs on the GPU through onnxruntime-gpu (a CUDA 12 build, sharing PyTorch's CUDA and cuDNN wheels), when about 1.4 GB of VRAM still leaves 3 GB free; otherwise on the CPU, which is fast enough. Every GPU backend checks free VRAM first, so a GPU already busy with another model (a local LLM server) does not get pushed into paging; the log says where each one runs (`tts: Kokoro ... on CUDAExecutionProvider` or `CPUExecutionProvider`).
  - **Windows / WSL:** not tested. WSL needs working audio (WSLg) plus the Linux packages above.

## First run

Run `/live setup` once on each machine. uv installs the Python packages, then the sidecar downloads the models your
settings name and tests each piece: PortAudio and the mic and speaker, espeak-ng, the synthesizer, the recognizer
(it speaks a sentence into it and checks what comes back), one sentence through the speaker, and the front server
(with `frontUrl` empty it downloads llama-server and the front model, starts the server once and stops it).
Progress shows in the status line, and a report with a ✓, ! or ✗ per step lands in the transcript. Installing the
plugin installs none of this, and setup never installs system packages: a ✗ line names the command to run (`uv`
itself, `espeak-ng`, `libportaudio2`). Change a setting, run it again.

Without setup, the first `/live` or `/livevibe` downloads the models, and the status line shows `loading` until they are ready:
Kyutai STT about 2.4 GB (Apple silicon, or Linux with an Nvidia GPU), the Whisper model (75 to 500 MB), Kokoro about 330 MB, and Silero VAD about 2 MB. Later starts take a few seconds. Kokoro and Silero are cached in `~/.cache/duplex_voice` (`LIVE_VIBE_CACHE` overrides it); Kyutai and Whisper use the Hugging Face cache.

## Sidecar log

Everything the sidecar says (startup, which recognizer and synthesizer ran or fell back, whether the echo canceller loaded, front warm-up failures, warnings, and library output on stderr) is also appended to `~/.cache/duplex_voice/sidecar.log` (`LIVE_VIBE_CACHE` or `XDG_CACHE_HOME` move it; the `logFile` setting sets the exact path). `/live setup` reports the path. Each session starts with a header line (plugin version, arguments, platform, Python), and each line is `ISO-time LEVEL pid=<sidecar pid> text`, with LEVEL one of INFO, WARNING or STDERR:

```
2026-10-02T23:14:42.826-06:00 INFO pid=786551 session start: live-vibe 0.3.1 argv=[...] platform=Linux-... python=3.12.3
2026-10-02T23:14:50.101-06:00 WARNING pid=786551 Kyutai STT on CUDA unavailable (...); using Whisper.
```

The file rotates at 5 MB and keeps one previous copy (`sidecar.log.1`). If the path cannot be written, the sidecar warns once and carries on without it.

## Settings (`/config`)

| Field | Default | Meaning |
|---|---|---|
| `stt` | `kyutai` | `kyutai` (streaming, ends your turn on what you said; MLX on Apple silicon, CUDA on Linux with an Nvidia GPU, else Whisper) or `whisper` (Silero VAD + faster-whisper). |
| `asr` | `base.en` | Whisper size: `tiny.en`, `base.en`, `small.en`. |
| `tts` | `kokoro` | `kokoro`, or `say` (macOS only). |
| `voice` | empty | A Kokoro voice (`af_heart`) or a macOS `say` voice. |
| `mic` | empty | Input device: an index or part of its name (`AirPods`). Empty means the system default. |
| `speaker` | empty | Output device, the same way. |
| `logFile` | empty | Sidecar log path; empty means `~/.cache/duplex_voice/sidecar.log`. |
| `endSilenceMs` | 1500 | Whisper: the pause that ends your turn. Kyutai: only a cap on a pause between words. |
| `frontBackend` | `llamacpp` | `llamacpp` (any OpenAI-compatible server) or `anthropic` (needs `ANTHROPIC_API_KEY` or `ant auth login`). |
| `frontUrl` | empty | Empty: a managed llama-server (below). Set: the front's server, which can be another host, such as a GPU box; nothing starts locally. |
| `frontServerBin` | empty | Managed server: an existing `llama-server` to run instead of the download. |
| `frontServerModel` | empty | Managed server: an existing GGUF to serve instead of the default model. |
| `frontServerLog` | empty | Managed server: where its stdout and stderr go; empty means `~/.cache/duplex_voice/front-server.log`. |
| `frontModel` | empty | Sent as `model` on every request. Empty means the server's own model (or `claude-haiku-4-5` for anthropic). |

## The managed front server

With `frontUrl` empty and the `llamacpp` backend, `/livevibe` runs its own llama.cpp `llama-server`. `/live setup`
downloads a pinned prebuilt release (llama.cpp v0.5.0, build `b11146`, checked against GitHub's sha256 digests) and the
default model, Qwen3-4B-Instruct-2507 Q4_K_M (2.5 GB, checked against the Hugging Face sha256), into
`~/.cache/duplex_voice`, starts the server once, and reports `front: managed llama-server <version> with <model> on
<GPU|CPU>`. After that, nothing goes to the network. Which build it downloads:

| Machine | Build | Download |
|---|---|---|
| macOS, Apple silicon | Metal | 11 MB |
| Linux x86_64 with an Nvidia GPU (`nvidia-smi` works, WSL2 too) | CUDA 12.8 plus its runtime libraries | 760 MB |
| Linux x86_64, no Nvidia GPU, a Vulkan loader, not WSL | Vulkan | 31 MB |
| Other Linux x86_64 or arm64, macOS Intel | CPU | 11 to 17 MB |
| Windows x86_64 (untested) | CUDA 12.4 with an Nvidia GPU, else CPU | 645 / 19 MB |

CUDA 12.8 rather than 13: 12.8 already has Blackwell kernels, while CUDA 13 needs a newer driver and drops pre-Turing
GPUs, and the rest of the sidecar (PyTorch for Kyutai, onnxruntime for Kokoro) runs CUDA 12 anyway. Vulkan is not used
on WSL2, which has no Nvidia Vulkan driver.

Each `/livevibe` start picks a free port on 127.0.0.1 and runs
`llama-server -m <model> --host 127.0.0.1 --port <port> --jinja -c 16384 -np 1 --no-webui -ngl 99`, waits for
`/health`, and points the front at it; the server stops when live vibe stops or the sidecar exits. If the sidecar is
killed outright, Linux takes the server down with it (`PR_SET_PDEATHSIG`); elsewhere the next start stops a server
whose sidecar is gone. The log (`frontServerLog`) gets a header with the command line, port, model and placement at
every start. The server only answers requests that carry a key made fresh for each start (passed in `LLAMA_API_KEY`, never
on the command line), since llama-server otherwise accepts calls from any web page.

GPU memory is shared with the voice models through one budget. The front asks first, before the recognizer and
Kokoro, since a front model on the CPU costs seconds on every reply, while the recognizer falls back to Whisper and
Kokoro is fine on the CPU. Its need is worked out from the GGUF (weights, f16 KV cache at the context, about 0.75 GB
for the CUDA context): 5.3 GiB for the default model. If the GPU would keep less than 3 GiB free, the server runs on
the CPU (`-ngl 0 --device none`) with one warning that replies will be slow. On Apple silicon it always uses Metal.

To skip the downloads, point `frontServerBin` at a `llama-server` you built and `frontServerModel` at a GGUF you have.
If anything fails, the warning says why and speech goes straight to Claude, as when any front server is down.
To use a server you run instead, set `frontUrl` (or `/livevibe url <url>`), as below.

## The front server

Any OpenAI-compatible chat server that takes `response_format` with a JSON schema works (llama-server does). Each
front turn is one JSON object, `{"say": ..., "delegate": ...}`, held to that schema by the server's grammar: small
models offered a delegate tool tend to say they are on it and call nothing, but they fill a required field. The
`say` text is spoken as it streams. A small model is enough; Qwen3-4B-Instruct-2507 delegated every work request in
testing, in about 5 GB of VRAM with a 16k context (4 GB at 8k):

```sh
llama-server -m Qwen3-4B-Instruct-2507-Q4_K_M.gguf --jinja -c 16384 -np 1 -ngl 99 --flash-attn on --port 8080
```

To switch between small models without keeping a big one in memory, put a router in front (llama-swap, or llama-server's multi-model mode where your build has it) and pick the model with `/livevibe model <name>`. A plain llama-server ignores the model name. Thinking is turned off per request, so a voice turn does not wait for it.

## Checks

```sh
uv run --script bin/sidecar/main.py --list-devices   # the indexes and names for mic and speaker
uv run --script bin/sidecar/main.py --setup          # what /live setup runs (with the default settings)
uv run --script bin/sidecar/main.py --unit           # pure unit checks
uv run --script bin/sidecar/main.py --selftest       # both modes, no mic, speaker or network
```
