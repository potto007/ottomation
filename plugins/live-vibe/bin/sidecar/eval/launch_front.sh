#!/usr/bin/env bash
# Serves one front model for the relay eval: the managed b11146 llama-server on 127.0.0.1:8092, -c 16384 -np 1
# -ngl 99, stdout and stderr appended to the canonical /tmp/llama-server.log. Run it as a background process, and only
# after a separate `nvidia-smi --query-gpu=memory.used,memory.free --format=csv` shows 8 GiB or more free. It writes
# the server's exact PID to $LIVE_VIBE_EVAL_OUT/launch_front.pid; stop it with `kill <that pid>`, never pkill.
#
#   launch_front.sh <path to .gguf>
set -u
GGUF=${1:?usage: launch_front.sh <path to .gguf>}
[ -f "$GGUF" ] || { echo "no such model: $GGUF" >&2; exit 2; }
BIN=${LIVE_VIBE_LLAMA_SERVER:-$HOME/.cache/duplex_voice/llama.cpp/b11146-linux-x64-cuda/llama-server}
OUT=${LIVE_VIBE_EVAL_OUT:-/tmp/live-vibe-relay-eval}
mkdir -p "$OUT"
"$BIN" -m "$GGUF" --host 127.0.0.1 --port 8092 --jinja -c 16384 -np 1 -ngl 99 > >(tee -a /tmp/llama-server.log) 2>&1 &
echo $! > "$OUT/launch_front.pid"
echo "llama-server pid $! serving $(basename "$GGUF") on 127.0.0.1:8092 (pid in $OUT/launch_front.pid)"
wait
echo "EXITED $?"
