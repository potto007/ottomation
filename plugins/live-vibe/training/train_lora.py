"""Train the timing LoRA on Qwen3-4B-Instruct-2507 from make_windows.py's train.jsonl (TRL SFT, PEFT LoRA, bf16).

Runs in a venv that has trl, peft, transformers and torch (rlm-trainer's: `~/src/rlm-trainer/.venv/bin/python`):

  <venv>/bin/python plugins/live-vibe/training/train_lora.py --data <dir>/train.jsonl --out <dir>/adapter

Prompt-completion rows: the loss is on the assistant JSON only (TRL's completion_only_loss), never on the long
system prompt or the history. One epoch, LoRA r=16 on every attention and MLP projection: about 30M trainable
parameters, which is what the video's author found enough (he trained the last four layers). The adapter is saved as
PEFT safetensors; convert it for llama-server with llama.cpp's convert_lora_to_gguf.py (see the README).
"""
from __future__ import annotations

import argparse
import json
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--eval", default="", help="heldout.jsonl for the eval loss (optional)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="rows to use (0 = all); a smoke run uses a few hundred")
    ap.add_argument("--eval-rows", type=int, default=400, help="held-out rows for the eval loss (every 100 steps)")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    t0 = time.monotonic()
    data_files = {"train": args.data}
    if args.eval:
        data_files["eval"] = args.eval
    ds = load_dataset("json", data_files=data_files)
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in ("prompt", "completion")])
    if args.limit:
        ds["train"] = ds["train"].select(range(min(args.limit, len(ds["train"]))))
    if "eval" in ds:
        n = min(args.limit // 4 or args.eval_rows, args.eval_rows, len(ds["eval"]))
        ds["eval"] = ds["eval"].shuffle(seed=7).select(range(max(1, n)))
    print(f"rows: train {len(ds['train'])}" + (f", eval {len(ds['eval'])}" if "eval" in ds else ""), flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False

    peft_config = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    cfg = SFTConfig(
        output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr, lr_scheduler_type="cosine",
        warmup_ratio=0.05, per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.accum,
        per_device_eval_batch_size=args.batch, bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, max_length=args.max_length, packing=False,
        completion_only_loss=True, logging_steps=10, save_strategy="no",
        eval_strategy="steps" if "eval" in ds else "no", eval_steps=100, report_to=[], seed=7,
        dataloader_num_workers=2, remove_unused_columns=True,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds["train"], eval_dataset=ds.get("eval"),
                         processing_class=tok, peft_config=peft_config)
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"trainable parameters: {trainable / 1e6:.1f}M", flush=True)
    result = trainer.train()
    print(f"train: {json.dumps(result.metrics)}", flush=True)
    if "eval" in ds:
        print(f"eval: {json.dumps(trainer.evaluate())}", flush=True)
    trainer.model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    peak = torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else 0.0
    print(f"saved adapter to {args.out}; peak reserved VRAM {peak:.1f} GiB; {time.monotonic() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
