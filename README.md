# GPT

A minimal, from-scratch GPT-2 style transformer (nanoGPT-inspired) with a basic training loop.

## Structure

```
.
├── model/
│   ├── GPT.py             # Full model: embeddings, stacked blocks, weight tying, init
│   ├── attention_block.py # Transformer block (pre-norm attn + mlp, residual connections)
│   ├── attention.py       # Multi-head causal self-attention ("vanilla")
│   ├── attention_registry.py # BaseAttention + name→class registry (the attention swap point)
│   ├── mlp.py             # Feed-forward (4x expansion, GELU)
│   └── layer_norm.py      # LayerNorm wrapper (supports bias=False)
├── GPT_Config.py           # ModelConfig (model size) and TrainingConfig (training hparams)
├── Dataloader.py           # Sequential batch loader over tokenized text (tiktoken, gpt2 encoding)
├── training_utils.py       # loss calculation + AdamW optimizer setup (decay/no-decay param groups)
├── metrics.py              # metrics.json writer: loss/perplexity curves + timing & memory aggregates
├── train.py                # Training loop entrypoint
├── data/
│   └── shakespeare/
│       └── input.txt       # training corpus (add your own text file here)
└── requirements.txt
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate       # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Put a plain text file at `data/shakespeare/input.txt`. It gets tokenized with the GPT-2 BPE (`tiktoken`).

## Train

```bash
python3 train.py
```

Model/training hyperparameters (layers, heads, embedding size, batch size, learning rate, steps, etc.) live in `GPT_Config.py`.

## Swapping the attention

The attention implementation is selected by name, in one place: `ModelConfig.attention` (or `--attention`
on the command line). Nothing else — the block, the profiler, the checkpoint — names a concrete class,
so a new variant needs one new file and no edits anywhere else:

```python
# model/custom_attention.py
from .attention_registry import BaseAttention, register_attention

@register_attention("custom")
class CustomAttention(BaseAttention):
    def __init__(self, config):
        super().__init__(config)
        ...
    def forward(self, X):        # (B, T, C) -> (B, T, C)
        ...
```

```bash
python3 train.py --list_attentions        # names registered under model/
python3 train.py --attention custom       # train with it
```

Modules under `model/` are imported on first lookup, so dropping the file in is enough to make the name
resolvable. An implementation living outside the package can be given as an import path instead:
`--attention my_experiments.linear_attn:LinearAttention`.

Because `--profile_attn` hooks every `BaseAttention` subclass, a swapped-in variant is timed and
memory-measured automatically; the name it ran under is recorded as `run.attention` in `metrics.json`,
and in each checkpoint, so resuming rebuilds the architecture the weights belong to (a `--attention`
that contradicts the checkpoint is rejected rather than failing later in `load_state_dict`).

## Metrics

Every run writes `metrics.json` (`--metrics_path` to change it), rewritten atomically on every logged
step so an interrupted run still leaves usable results.

* `train` / `val` — per-step curves. Train loss + perplexity, lr and grad norm every `--log_interval`
  steps; val loss + perplexity every `--eval_interval` steps. Each entry carries `tokens_seen`, so
  curves from different batch sizes can be plotted against compute rather than step count.
* `aggregates` — step time, tokens/sec, per-phase time (data / forward / backward / optimizer), peak
  memory, and attention-layer time and peak memory. The first `--metrics_warmup_steps` steps (default 10)
  are discarded — allocator warm-up, cuDNN autotuning and `torch.compile` tracing make them
  unrepresentative — and everything after is reported as mean/std/median/min/max/p90 rather than
  per step.

Attention-level numbers come from forward/backward hooks (`--profile_attn`, on by default). Hooks break
the compiled graph, so enabling them turns `torch.compile` off; run with `--no-profile_attn` for
compiled-throughput numbers. `run.torch_compile` in the JSON records which mode produced the file.

GPU timings use CUDA events resolved once per step after a single synchronize, so instrumentation stays
off the critical path; peak memory is tracked in segments around each attention call, which yields an
exact whole-step high-water mark as well as the attention-only one.

## Contributing notes

- Model code lives under `model/`, one component per file. `Attention_Block` composes an attention layer (resolved by name through `attention_registry.py`) + `MLP` with the pre-norm + residual pattern; don't remove the residual adds (`x + ...`) — training was verified to blow up without them.
- New attention variants subclass `BaseAttention` and register a name — see "Swapping the attention" above. Don't import a concrete attention class outside `model/attention_registry.py`; that's what keeps the swap to a single config value.
- Data loading, config, and training-loop utilities are kept flat at the project root, outside `model/`, since they're training-specific rather than part of the model definition.