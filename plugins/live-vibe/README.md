# live-vibe

Voice and director modes for Claude Code, as a function-hook plugin (a mod) with a Python audio sidecar
(`bin/sidecar/`, started by the mod with `uv run --script bin/sidecar/main.py`). The sidecar talks to the mod over
stdout and a loopback HTTP port guarded by a per-run token.

| Command | What it does |
|---|---|
| `/live` | Full-duplex voice with Claude. The mic stays open; you can talk over an answer to cut it off. What you say goes to Claude, and Claude's final answer is read aloud. Saying "switch to sonnet" runs `/model`. |
| `/vibe [request]` | Director mode. Claude reads and directs worker subagents; every other tool is denied on the main loop. |
| `/livevibe` | You talk with a small, fast voice model (the front). It chats with you directly, hands real work to Claude in vibe mode, and tells you the result in a sentence or two when it comes back. `/livevibe model [name]` lists or picks the front model; `/livevibe url [url]` points it at another server. |

`/live` and `/livevibe` turn each other off. `/livevibe` off puts vibe mode back the way it was.

## Requirements

- Claude Code 2.1.287 or newer (mods are on by default from that build).
- [uv](https://docs.astral.sh/uv/). The sidecar is a PEP 723 script; uv installs Python 3.12 and its packages on first use.
- A microphone and speaker on the machine running Claude Code. A headless or SSH-only box cannot use voice; `/vibe` still works there. Headphones are recommended, because the mic stays open while the assistant speaks. Pick devices with the `mic` and `speaker` settings; `--list-devices` (below) shows the choices.
- Per OS:
  - **macOS (Apple silicon):** `brew install espeak-ng` for Kokoro. Kyutai STT runs on MLX.
  - **Linux:** `sudo apt install espeak-ng libportaudio2`. Kyutai STT is Apple-silicon only, so Whisper runs instead. It uses an Nvidia GPU when CTranslate2 finds one and CUDA 12 cuBLAS and cuDNN 9 are installed; otherwise it uses the CPU.
  - **Windows / WSL:** not tested. WSL needs working audio (WSLg) plus the Linux packages above.

## First run

The first `/live` or `/livevibe` downloads models, and the status line shows `loading` until they are ready:
Kyutai STT about 2.4 GB (Apple silicon), the Whisper model (75 to 500 MB), Kokoro about 330 MB, and Silero VAD about 2 MB. Later starts take a few seconds. Kokoro and Silero are cached in `~/.cache/duplex_voice` (`LIVE_VIBE_CACHE` overrides it); Kyutai and Whisper use the Hugging Face cache.

## Settings (`/config`)

| Field | Default | Meaning |
|---|---|---|
| `stt` | `kyutai` | `kyutai` (streaming, ends your turn on what you said) or `whisper` (Silero VAD + faster-whisper). |
| `asr` | `base.en` | Whisper size: `tiny.en`, `base.en`, `small.en`. |
| `tts` | `kokoro` | `kokoro`, or `say` (macOS only). |
| `voice` | empty | A Kokoro voice (`af_heart`) or a macOS `say` voice. |
| `mic` | empty | Input device: an index or part of its name (`AirPods`). Empty means the system default. |
| `speaker` | empty | Output device, the same way. |
| `endSilenceMs` | 1500 | Whisper: the pause that ends your turn. Kyutai: only a cap on a pause between words. |
| `frontBackend` | `llamacpp` | `llamacpp` (any OpenAI-compatible server) or `anthropic` (needs `ANTHROPIC_API_KEY` or `ant auth login`). |
| `frontUrl` | `http://127.0.0.1:8080` | The front's server. It can be another host, such as a GPU box. |
| `frontModel` | empty | Sent as `model` on every request. Empty means the server's own model (or `claude-haiku-4-5` for anthropic). |

## The front server

Any OpenAI-compatible chat server with tool calling works. With llama.cpp:

```sh
llama-server -m Qwen3-8B-Q4_K_M.gguf --jinja --port 8080   # --jinja is required for tool calls
```

To switch between small models without keeping a big one in memory, put a router in front (llama-swap, or llama-server's multi-model mode where your build has it) and pick the model with `/livevibe model <name>`. A plain llama-server ignores the model name. Thinking is turned off per request, so a voice turn does not wait for it.

## Checks

```sh
uv run --script bin/sidecar/main.py --list-devices   # the indexes and names for mic and speaker
uv run --script bin/sidecar/main.py --unit           # pure unit checks
uv run --script bin/sidecar/main.py --selftest       # both modes, no mic, speaker or network
```
