#!/usr/bin/env bash
# The front relay eval for one model and one or more versions of the sidecar, then the score.
#
#   relay_eval.sh <path to .gguf | http://host:port> <label> [runs] [ref ...]
#
# A .gguf path means the model launch_front.sh serves on :8092 (checked first, so a label never scores the wrong
# model); a URL means a server someone else runs (only its /health is checked; nothing is started or stopped here).
# Each ref is "worktree" (this checkout, the default) or a git ref of this repo. Results go to $LIVE_VIBE_EVAL_OUT
# (default /tmp/live-vibe-relay-eval), never into the repo: relay_<label>_<ref>.jsonl. Run it through the work band:
#
#   ( relay_eval.sh ... 2>&1; rc=$?; [ $rc -eq 0 ] && echo DONE || echo FAILED ) | tee -a /tmp/live-vibe-relay-eval.log
set -u
MODEL=${1:?usage: relay_eval.sh <path to .gguf | base url> <label> [runs] [ref ...]}
LABEL=${2:?usage: relay_eval.sh <path to .gguf | base url> <label> [runs] [ref ...]}
RUNS=${3:-3}
shift $(( $# < 3 ? $# : 3 ))
REFS=("$@")
[ ${#REFS[@]} -gt 0 ] || REFS=(worktree)
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${LIVE_VIBE_EVAL_OUT:-/tmp/live-vibe-relay-eval}
P=/tmp/live-vibe-relay-eval.log.progress.jsonl
mkdir -p "$OUT"
case "$MODEL" in
  http://*|https://*)
    URL=${MODEL%/}
    curl -sf -m 10 "$URL/health" >/dev/null || { echo "no healthy server at $URL" >&2; exit 2; }
    echo "using $URL" ;;
  *)
    URL=http://127.0.0.1:8092
    served=$(curl -sf "$URL/v1/models") || { echo "nothing answers on :8092" >&2; exit 2; }
    case "$served" in
      *"$(basename "$MODEL")"*) echo "serving $(basename "$MODEL") on :8092" ;;
      *) echo ":8092 serves something else: $served" >&2; exit 2 ;;
    esac ;;
esac
files=()
i=0
for ref in "${REFS[@]}"; do
  safe=${ref//\//_}
  f="$OUT/relay_${LABEL}_${safe}.jsonl"
  rm -f "$f"
  printf '{"v":1,"phase":"eval","label":"relay %s","current":"%s","done":%d,"total":%d}\n' \
    "$LABEL" "$ref" "$i" "${#REFS[@]}" >> "$P"
  uv run --python 3.12 --with numpy --with httpx python "$HERE/run.py" run "$LABEL/$ref" "$f" --ref "$ref" \
    --runs "$RUNS" --url "$URL" || exit 1
  files+=("$f")
  i=$((i + 1))
done
printf '{"v":1,"phase":"score","label":"relay %s","done":%d,"total":%d}\n' "$LABEL" "$i" "$i" >> "$P"
python3 "$HERE/run.py" score "${files[@]}"
