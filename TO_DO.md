# To-Do: Efficient Attention Mechanism — Experiment Tracking

## Open Items / Fixes Needed
- [ ] Fix the cusotm attention  
- [ ] while scaling need to consider the max learning rate 
- [ ] for profiling need to change the attention layer for profiling only swapping the attention would give incorrect results
---

## 1. Model Quality Metrics ("does it still work")
- [x] Log train loss (continuous, every N steps)
- [x] Log val loss (continuous, every N steps)
- [x] Log val perplexity (continuous, every N steps)
- [ ] Report perplexity on a standard held-out corpus (WikiText-103 / PG-19 / own val split)
- [ ] Run zero-/few-shot downstream evals via `lm-eval-harness`:
  - [ ] HellaSwag
  - [ ] OpenBookQA
  - [ ] WinoGrande
  - [ ] ARC-Easy
  - [ ] ARC-Challenge
  - [ ] BoolQ
  - [ ] Social-IQA
  - [ ] PIQA
- [ ] Long-context evals (if targeting long sequences):
  - [ ] Needle-in-a-haystack / RULER
  - [ ] Associative recall / MQAR
  - [ ] Long Range Arena (LRA)
  - [ ] Long-doc QA: TriviaQA, NQ, SQuADv2, DROP

## 2. Efficiency / Systems Metrics ("is it actually faster")
- [ ] Throughput (tokens/sec) — inference, prefill vs. decode split
- [x] Training step time (tokens/sec)
- [ ] Report throughput at multiple sequence lengths: 1K / 4K / 16K / 64K
- [x] Peak memory + time inside the attention layers, and for the whole training step
- [ ] Peak memory (activations + KV cache) vs. sequence length curve
- [ ] FLOPs (theoretical) vs. wall-clock (measured)
- [ ] Latency per generated token — fixed batch size, fixed GPU, stated precision (e.g., A100-80GB, bf16)
- [ ] GPU speedup relative to baseline
- [ ] (Optional but strong) Average Energy Consumption (AEC)

## 3. Scaling Behavior
- [ ] Train at 2–3 model scales (e.g., 50M / 125M / 350M)
- [ ] Use compute-optimal (Chinchilla-style, ~20 tokens/param) training budgets
- [ ] Plot loss vs. compute across scales
- [ ] Confirm scaling curve shape matches/deviates from baseline Transformer

## 4. Continuous Logging (every step / eval interval)
- [x] Train/val loss + perplexity
- [x] Gradient norm
- [x] Learning rate
- [x] Throughput (tokens/sec) + step time
- [x] GPU memory (peak allocated/reserved)
- [ ] Attention-pattern diagnostics (effective receptive field, synthetic recall probe)
- [ ] Wall-clock time and power draw (if efficiency claims are made)
- [ ] Set up W&B or TensorBoard for all of the above as time series

## 5. Baselines to Compare Against
- [ ] Vanilla self-attention w/ FlashAttention-2/3 (fast, exact baseline)
- [ ] MQA
- [ ] GQA
- [ ] MLA
- [ ] Sliding-window / block-sparse attention
- [ ] NSA / MoBA-style sparse approaches
- [ ] Linear/SSM family: Performer, Linformer, RetNet, Mamba, GLA (at least 1–2)
- [ ] 1–2 recent SOTA methods in the specific niche (check last ~6 months of arXiv)

## 6. Statistical Rigor / Reporting for Publication
- [ ] Run 3+ seeds for headline results; report mean ± std
- [ ] Same tokenizer, data, and total training tokens across all baselines
- [ ] Report exact hardware, precision, batch size, and sequence length for every efficiency number
- [ ] Ensure reproducibility: fix and document all hyperparameters and configs