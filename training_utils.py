import inspect
import math

import torch
import torch.nn.functional as F


def calculate_loss(logits, targets):
    """
    Cross-entropy loss for next-token prediction.
    logits:  (B, T, vocab_size)
    targets: (B, T)
    """
    # cross_entropy expects 2D input, so flatten batch and time dims together
    # (B, T, C) -> (B*T, C)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return loss


# Backwards-compatible alias for the old (misspelled) name, in case other code
# in the repo still imports `claculate_loss`.
claculate_loss = calculate_loss


def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    """
    Split parameters into two groups: everything with ndim >= 2 (matmul weights,
    embeddings) gets weight decay; everything else (biases, LayerNorm gains) does
    not. Uses fused AdamW when running on CUDA and the installed torch supports it.
    """
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    decay_params = [p for p in param_dict.values() if p.dim() >= 2]
    nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    print(f"using fused AdamW: {use_fused}")

    return optimizer


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Linear warmup for `warmup_steps`, then cosine decay to `min_lr` by `max_steps`."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # ranges 1 -> 0
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def estimate_loss(model, dataloader, eval_iters, device, device_type, ptdtype):
    """
    Average loss over `eval_iters` randomly sampled batches from `dataloader`.
    Temporarily switches the model to eval mode and restores whatever mode it
    was in beforehand.
    """
    was_training = model.training
    model.eval()
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        x, y = dataloader.random_batch()
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device_type, dtype=ptdtype):
            logits = model(x)
            loss = calculate_loss(logits, y)
        losses[i] = loss.item()
    if was_training:
        model.train()
    return losses.mean().item()