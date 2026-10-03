# Front relay eval

Scores what the `/livevibe` voice front says and hands to Claude, in the situations that went wrong live or in earlier
runs: a stale "still running", a guessed cause, a rewritten request that lost the user's question, a confirmation
never passed on, a result announced twice or swallowed by a short reply, a question answered from memory. Each
fixture in `fixtures.py` states its pass criterion; `run.py` drives the sidecar's real `FrontSession` (front prompt,
JSON turn, delegator, waiting results, notes) with a silent voice against a model server, and scores the speech, the
announcement that follows the turn, and the emitted `delegate` and `note` messages.

It needs a model, so it is not part of `--unit` or `--selftest`. Results are written outside the repo, to
`$LIVE_VIBE_EVAL_OUT` (default `/tmp/live-vibe-relay-eval`).

## Rules for the GPU and the logs

- Check the GPU on its own first, and launch only with 8 GiB or more free:
  `nvidia-smi --query-gpu=memory.used,memory.free --format=csv`. Never chain the check and the launch.
- `launch_front.sh` starts the managed llama.cpp build (b11146) on `127.0.0.1:8092` with `--jinja -c 16384 -np 1
  -ngl 99`, and appends its stdout and stderr to the canonical `/tmp/llama-server.log` (the file promtail reads).
  Run it as a background process. It writes the server's exact PID to `$LIVE_VIBE_EVAL_OUT/launch_front.pid`.
- Stop it with `kill <that pid>` and confirm with `ps -p <pid>`; never `pkill -f`. Then check `nvidia-smi` again.
- A server someone else runs (a URL) is only read: nothing is started, stopped or restarted.
- The eval takes a few minutes: run it with a `DONE` / `FAILED` last line, through the work band.

## Commands

From `plugins/live-vibe/bin/sidecar/eval/`:

```sh
nvidia-smi --query-gpu=memory.used,memory.free --format=csv          # on its own; 8 GiB or more free

./launch_front.sh ~/.cache/duplex_voice/models/unsloth__Qwen3-4B-Instruct-2507-GGUF/Qwen3-4B-Instruct-2507-Q4_K_M.gguf
cat /tmp/live-vibe-relay-eval/launch_front.pid; ps -p "$(cat /tmp/live-vibe-relay-eval/launch_front.pid)"
curl -s 127.0.0.1:8092/health                                         # until {"status":"ok"}

: > /tmp/live-vibe-relay-eval.log; : > /tmp/live-vibe-relay-eval.log.progress.jsonl
MODEL=~/.cache/duplex_voice/models/unsloth__Qwen3-4B-Instruct-2507-GGUF/Qwen3-4B-Instruct-2507-Q4_K_M.gguf
( ./relay_eval.sh "$MODEL" qwen3-4b 3 worktree main 2>&1; rc=$?; [ $rc -eq 0 ] && echo DONE || echo FAILED ) \
  | tee -a /tmp/live-vibe-relay-eval.log

kill "$(cat /tmp/live-vibe-relay-eval/launch_front.pid)"; nvidia-smi --query-gpu=memory.used,memory.free --format=csv
```

`relay_eval.sh <gguf | url> <label> [runs] [ref ...]`: each ref is `worktree` (this checkout) or a git ref, so one
server scores several versions side by side (`c70cc29` is 0.5.0, before the relay fixes). A URL in place of the GGUF
uses a server that is already running, e.g. `http://172.19.144.1:8090`.

Other entry points:

```sh
uv run --python 3.12 --with numpy --with httpx python run.py dry --ref main      # the messages each fixture sends
uv run --python 3.12 --with numpy --with httpx python run.py run <variant> <out.jsonl> --ref <ref> --only A_what_fix
python3 run.py score /tmp/live-vibe-relay-eval/relay_qwen3-4b_worktree.jsonl ...
```

## Reading the score

Per fixture: its own rubric (`own`), the checks added after its first results (`extra`, scored apart so `own` stays
comparable across runs), and two global checks: `invented` (a turn that hands nothing off announces an action the front
cannot take, such as "I'll mark it verified") and `asked-twice`. `note kinds` shows what reached Claude without a
delegation: `ask` and `answer` start a Claude turn, `note` joins the conversation quietly, `-` is nothing (a
delegation, or an announcement). Every failure is listed with its speech, its delegation and any announcement after it.
