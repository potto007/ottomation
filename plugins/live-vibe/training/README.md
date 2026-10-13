# The timing LoRA

Increment 3 of the micro-turn front (`docs/design/microturn-front.md`). The loop asks the 4B front every 200 ms whether
to listen or speak; with the prompt alone it speaks mid-sentence, repeats itself and narrates outcomes it was never
given, and the sidecar's floor rules stand in for the judgement it lacks. This folder trains that judgement into a
LoRA adapter on the same model, so the rules become safety nets.

Everything here runs on the local machine. Only `train_lora.py` and the scoring server touch the GPU (ask first).

## 1. Data (CPU)

```sh
uv run --python 3.12 --with numpy --with httpx python plugins/live-vibe/training/make_windows.py --out /tmp/lora-data
```

`make_windows.py` writes synthetic calls cut into the loop's windows, with the answer each window should get. It
imports the sidecar's own prompt, window message, event and notes messages and JSON turn, so an example is exactly
what the loop sends at runtime. Measured on 600 calls: about 25k windows, kept as 5.4k training rows (every speak
window, listen windows sampled to 1.5 per speak and stratified by kind), p50 1055 tokens, max 1840; 10% of calls held
out (`heldout.jsonl` for the eval loss, `heldout_windows.jsonl` for the scorer).

What a window is labelled:

| Window | Answer |
|---|---|
| the user mid-sentence, p low (or a brief 0.4 to 0.6 pause) | listen |
| the user's words over the voice (`[speaking]`) | listen (the sidecar cuts the voice) |
| a finished request, on the raw text as p crosses 0.6 or on the clean final text | speak, the delegation, a few words naming the task |
| the clean final window after the raw one was answered | listen (one line per utterance) |
| a status question with a fresh report | speak the report's status in one sentence |
| a status question with no fresh report, or a question about the work | speak "Let me check." and, for the work question, delegate it |
| "yes" to a question the last report asked | speak, delegate the proposed action, name it |
| "no" to that question | listen (the note reaches Claude as the answer) |
| small talk | speak a short line |
| a long dictation pause | "Mm-hm." about one time in four, else listen |
| `[no voice]`, with or without `[speaking]` | listen |

## 2. Train (GPU)

```sh
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ~/src/rlm-trainer/.venv/bin/python \
    plugins/live-vibe/training/train_lora.py --data /tmp/lora-data/train.jsonl --eval /tmp/lora-data/heldout.jsonl \
    --out /tmp/lora-data/adapter --batch 1 --accum 16
```

TRL `SFTTrainer` with PEFT: bf16 base (`Qwen/Qwen3-4B-Instruct-2507`, 8 GB download), LoRA r=16 on every attention and
MLP projection, loss on the assistant JSON only, one epoch, lr 2e-4 cosine, eval loss on 400 held-out rows once at the
end. Run it through the work band with a `DONE` / `FAILED` sentinel; `--limit 160` is a smoke run. On the 32 GB card
batch 1 with 16 accumulation steps holds about 30 GB reserved (batch 4 spilled to system RAM, and so did an eval pass
in the middle of the epoch); watch power and step time, a drop from about 400 W and a rising s/it is the spill.

## 3. Convert and score

```sh
~/src/rlm-trainer/.venv/bin/python ~/src/llama.cpp/convert_lora_to_gguf.py /tmp/lora-data/adapter \
    --base ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/<rev> --outfile /tmp/lora-data/timing-lora.gguf

bin/sidecar/eval/launch_front.sh <Qwen3-4B Q4_K_M gguf>                              # the base, then
uv run --python 3.12 --with numpy --with httpx python plugins/live-vibe/training/score_heldout.py \
    --url http://127.0.0.1:8092 --windows /tmp/lora-data/heldout_windows.jsonl --limit 400 --label base
kill "$(cat /tmp/live-vibe-relay-eval/launch_front.pid)"
bin/sidecar/eval/launch_front.sh <gguf> /tmp/lora-data/timing-lora.gguf                # with the adapter
uv run ... score_heldout.py --url http://127.0.0.1:8092 --windows ... --limit 400 --label lora
```

The scorer drives the loop's own stepper (the same two requests the sidecar makes) and reports, by window kind, the
decision accuracy, whether a delegation was present when expected and absent when not, and whether the say makes a
claim about work, echoes the user, or runs long. Accept when every kind's decision accuracy is above the base and the
speak windows' claim and echo rates are near zero.

## 4. Use it

Set `frontServerLora` to the adapter's path (`/plugin configure live-vibe@ottomation`); the managed llama-server
starts with `--lora`. With `microturnMs` on, the loop runs on the trained model.
