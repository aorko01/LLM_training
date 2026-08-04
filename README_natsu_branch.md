# GPT CLM training on TinyStories (single GPU)

Autoregressive next-token-prediction (causal language modelling) training of a
GPT-style model on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
dataset.

## Setup

```bash
pip install -r requirements.txt
```

## 1. Prepare the data

Downloads TinyStories from the HuggingFace Hub, tokenizes it with the GPT-2 BPE
tokenizer (`tiktoken`), and writes `train.bin` / `val.bin` token shards:

```bash
python data/tinystories/prepare.py
```

A WikiText-103 variant is also included (`data/wikitext103/prepare.py`) if you
want to swap datasets - point `train.py` at it with `--data_dir data/wikitext103`.

## 2. Train

```bash
python train.py
```

Useful overrides:

```bash
python train.py --batch_size 8 --seq_len 512 --max_steps 20000 --lr 6e-4
```

Run `python train.py --help` for the full list of CLI options (data dir,
checkpoint dir, seed, `--no-compile`, `--profile_attn`, checkpoint interval,
`--resume`, etc). Config defaults live in `GPT_Config.py` (`ModelConfig` for
the architecture, `TrainingConfig` for optimization/eval/checkpoint/metrics
settings).

## 3. Resume a run

Every `ckpt_interval` steps (and whenever a new best val loss is hit), a
checkpoint is written to `out_dir`. Resume with:

```bash
python train.py --resume out/ckpt_latest.pt
```

This restores the model, optimizer, step count, dataloader position (so
training continues from the same point in the token stream instead of
restarting at the beginning of the data), and RNG state - and continues
appending to the same `metrics.json` instead of starting a fresh report.

## What changed from the original scaffold

- `Dataloader.py` no longer hardcodes a local Shakespeare `.txt` file. It now
  streams from pre-tokenized `train.bin` / `val.bin` shards via `np.memmap`
  (see `data/tinystories/prepare.py`), and supports both sequential batches
  (`next_batch`, used for training) and randomly sampled batches
  (`random_batch`, used for validation).
- `train.py`:
  - Autocast dtype and `torch.cuda.synchronize()` are now guarded by an actual
    CUDA check instead of being hardcoded to `"cuda"`, so the script no longer
    crashes on CPU-only machines.
  - Optimizer is built before `torch.compile` wraps the model, to avoid
    compiled parameter name prefixes leaking into checkpoints.
  - Added a validation loop (`eval_interval` / `eval_iters`) with
    best-checkpoint saving.
  - Added linear warmup + cosine LR decay (`get_lr` in `training_utils.py`)
    instead of driving `CosineAnnealingLR` off a fixed, disconnected
    `T_max=50000`.
  - Added `argparse` CLI overrides and a fixed random seed for reproducibility.
  - Added `MetricsTracker` integration (`metrics.py`) - per-step train/val
    curves, aggregate step time / tokens-per-sec / peak memory, and optional
    per-layer attention time+memory profiling (`--profile_attn`), all written
    to `metrics.json`.
  - Added checkpoint/resume support: `ckpt_best.pt` (best val loss so far) and
    `ckpt_latest.pt` (refreshed every `--ckpt_interval` steps) are each
    overwritten in place rather than accumulating files. Checkpoints store the
    model, optimizer, step, best-val-loss-so-far, dataloader position, and
    RNG state. `--resume <path>` restores all of it, so training - and the
    dataloader's position in the token stream - continues from exactly where
    it left off instead of restarting from scratch.
- `training_utils.py`:
  - Fixed the `configure_optimizers` signature - the first argument was
    named `self` despite being a plain function (it worked, but was
    misleading); it's now named `model`.
  - Fixed the misspelled `claculate_loss` -> `calculate_loss`, with a
    backwards-compatible alias kept for anything still importing the old name.
  - Added `get_lr` (warmup + cosine schedule) and `estimate_loss`
    (validation-loss averaging helper).
- `GPT_Config.py`: added `TrainingConfig` fields needed by the above
  (`data_dir`, `out_dir`, `eval_interval`, `eval_iters`, `warmup_steps`,
  `min_lr`, `seed`, `metrics_path`, `log_interval`, `metrics_warmup_steps`,
  `ckpt_interval`).
- `metrics.py` (new): `MetricsTracker` collects per-step train/val curves and
  aggregate stats (step time, tokens/sec, peak memory, attention time+memory)
  and writes them atomically to `metrics.json` after every logged step.
  `MetricsTracker.resume()` reloads an existing `metrics.json` - including the
  raw per-step samples aggregates are computed from, not just the curves - so
  a resumed run continues the same report instead of starting a new one.
- `data/wikitext103/prepare.py` (new): same shard-writing approach as the
  TinyStories prep script, pointed at WikiText-103 instead.