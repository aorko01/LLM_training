import argparse
import math
import os

import torch

from GPT_Config import ModelConfig, TrainingConfig
from Dataloader import Dataloader
from metrics import MetricsTracker, safe_exp
from model.GPT import GPT
from model.attention_registry import BaseAttention, available_attentions
from training_utils import calculate_loss, configure_optimizers, estimate_loss, get_lr


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GPT model for causal language modelling on TinyStories")
    parser.add_argument("--data_dir", type=str, default=TrainingConfig.data_dir,
                         help="directory containing train.bin / val.bin (see data/tinystories/prepare.py)")
    parser.add_argument("--out_dir", type=str, default=TrainingConfig.out_dir,
                         help="directory to write checkpoints to")
    parser.add_argument("--max_steps", type=int, default=TrainingConfig.max_steps)
    parser.add_argument("--batch_size", type=int, default=TrainingConfig.Batch)
    parser.add_argument("--seq_len", type=int, default=TrainingConfig.Sequence_length)
    parser.add_argument("--total_batches", type=int, default=TrainingConfig.Total_batches,
                         help="target tokens per optimizer step (drives gradient accumulation)")
    parser.add_argument("--lr", type=float, default=TrainingConfig.lr)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument("--compile", action="store_true", default=True,
                         help="use torch.compile (only takes effect on CUDA)")
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    parser.add_argument("--eval_interval", type=int, default=TrainingConfig.eval_interval,
                         help="run validation (and log val loss/perplexity) every N steps")
    parser.add_argument("--eval_iters", type=int, default=TrainingConfig.eval_iters,
                         help="number of random batches averaged per validation pass")
    parser.add_argument("--metrics_path", type=str, default=TrainingConfig.metrics_path,
                         help="where to write the metrics.json report")
    parser.add_argument("--log_interval", type=int, default=TrainingConfig.log_interval,
                         help="append train loss/perplexity to metrics.json every N steps")
    parser.add_argument("--metrics_warmup_steps", type=int, default=TrainingConfig.metrics_warmup_steps,
                         help="steps to discard before averaging step time / peak memory")
    parser.add_argument("--profile_attn", action="store_true", default=True,
                         help="measure time and peak memory inside the attention layers "
                              "(forces torch.compile off, since the hooks break the graph)")
    parser.add_argument("--no-profile_attn", dest="profile_attn", action="store_false")
    parser.add_argument("--ckpt_interval", type=int, default=getattr(TrainingConfig, "ckpt_interval", 500),
                         help="save a 'latest' checkpoint (for --resume) every N steps, in addition to "
                              "the best-val-loss checkpoint")
    parser.add_argument("--resume", type=str, default=None,
                         help="path to a checkpoint to resume from (restores model, optimizer, step, "
                              "dataloader position, and RNG state). If omitted, train.py will "
                              "automatically resume from whichever of <out_dir>/ckpt_latest.pt or "
                              "<out_dir>/ckpt_best.pt has the higher step, if either file exists, "
                              "otherwise it starts from scratch. Pass --no_resume to force a fresh run "
                              "even if a checkpoint is present.")
    parser.add_argument("--no_resume", action="store_true", default=False,
                         help="ignore any existing checkpoint in out_dir and start from scratch")
    parser.add_argument("--force_resume", action="store_true", default=False,
                         help="resume even if --lr/--batch_size/--seq_len/--total_batches differ "
                              "from the values recorded in the checkpoint (these changes silently "
                              "alter the LR schedule, effective batch size, and data stride "
                              "mid-run, so they're rejected by default). --max_steps is exempt "
                              "since extending training length across a resume is a normal thing "
                              "to do; a mismatch there is only a warning.")
    parser.add_argument("--attention", type=str, default=None,
                         help="which attention implementation to train with (default: "
                              f"{ModelConfig.attention!r}). Registered names are discovered from "
                              "the model/ package; an out-of-package implementation can be given "
                              "as 'package.module:ClassName'. See --list_attentions.")
    parser.add_argument("--run_name", type=str, default=None,
                         help="identifies this experiment; namespaces out_dir/metrics_path as "
                              "<out_dir>/<run_name> so different experiments never share "
                              "checkpoints or metrics.json by accident. Defaults to the attention "
                              "implementation name. Only applied when --out_dir/--metrics_path are "
                              "left at their defaults; explicit values are always respected as-is.")
    parser.add_argument("--list_attentions", action="store_true", default=False,
                         help="print the attention implementations registered in model/ and exit")
    return parser.parse_args()


def build_checkpoint(raw_model, optimizer, step, val_loss, best_val_loss, train_loader, tracker,
                     model_config, training_args, next_step):
    return {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        # The step to resume *at*, which is not always step+1: ckpt_best.pt is
        # written from the eval block at the top of the loop body, before step
        # `step` has trained, so it resumes at `step` itself; ckpt_latest.pt is
        # written at the end of the body, after the optimizer update, so it
        # resumes at step+1. Everything else captured here (weights, optimizer
        # state, train_loader_position, metrics_stats) is "as of" next_step, so
        # inferring it as step+1 for both would silently skip a whole optimizer
        # step and shift the data stream against the step counter.
        "next_step": next_step,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        # the config the weights were actually built from, so a resume can tell
        # which attention implementation this state_dict belongs to
        "model_config": model_config,
        # lr/batch_size/seq_len/total_batches/max_steps as of this checkpoint, so a
        # resume can detect a changed flag (e.g. a stale auto-restart script) instead
        # of silently continuing with a different LR schedule, effective batch size,
        # or data stride
        "training_args": training_args,
        # sequential offset into train_loader's token stream, so resuming
        # continues from here instead of restarting at the beginning of the data
        "train_loader_position": train_loader.current_position,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        # running aggregate accumulators as of exactly this step, so resuming
        # from this checkpoint restores averages in sync with start_step instead
        # of whatever metrics.json last happened to have on disk (which is
        # written more often than checkpoints and can be ahead of them)
        "metrics_stats": tracker.stats(),
    }


def save_checkpoint_atomic(payload, path):
    """Write a checkpoint via tmp-file + os.replace, so a kill mid-write (OOM,
    preemption, power loss) can never leave a half-written file at `path` for
    auto-resume to load and crash on. Same pattern as metrics.py's write()."""
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def main():
    args = parse_args()

    if args.list_attentions:
        print("registered attention implementations:")
        for name in available_attentions():
            print(f"  {name}")
        return

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    max_steps = args.max_steps
    lr = args.lr
    Batch = args.batch_size
    Sequence_length = args.seq_len
    Total_batches = args.total_batches

    assert Total_batches % (Batch * Sequence_length) == 0, "Total_batches must be divisible by Batch*Sequence_length"
    grad_accumulation_steps = Total_batches // (Batch * Sequence_length)
    print(f"grad_accumulation_steps: {grad_accumulation_steps}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = device
    # bf16 autocast only helps (and is only supported) on CUDA GPUs that support it;
    # fall back to plain fp32 everywhere else instead of hardcoding device_type="cuda".
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    ptdtype = torch.bfloat16 if use_bf16 else torch.float32
    print(f"using device: {device} | autocast dtype: {ptdtype}")

    # Namespace this run's artifacts (out_dir, metrics_path) by experiment identity
    # -- the attention implementation, or --run_name if given -- so that training
    # two different experiments never silently overwrites one run's checkpoints
    # and metrics.json with another's. Only kicks in when the user left
    # --out_dir/--metrics_path at their defaults; explicit values win as-is.
    attention_name = args.attention if args.attention is not None else ModelConfig.attention
    run_name = args.run_name or attention_name
    if args.out_dir == TrainingConfig.out_dir:
        args.out_dir = os.path.join(args.out_dir, run_name)
    if args.metrics_path == TrainingConfig.metrics_path:
        args.metrics_path = os.path.join(args.out_dir, TrainingConfig.metrics_path)
    print(f"run_name: {run_name} | out_dir: {args.out_dir} | metrics_path: {args.metrics_path}")

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Auto-resume logic -------------------------------------------------
    # If the user didn't explicitly pass --resume, look for the "latest"
    # checkpoint that gets written every --ckpt_interval steps. If it exists,
    # resume from it automatically; otherwise start fresh. --no_resume
    # overrides this and always starts from scratch, even if a checkpoint
    # file is sitting in out_dir.
    resume_path = args.resume
    latest_ckpt_path = os.path.join(args.out_dir, "ckpt_latest.pt")
    best_ckpt_path = os.path.join(args.out_dir, "ckpt_best.pt")
    if args.no_resume:
        resume_path = None
        if os.path.exists(latest_ckpt_path) or os.path.exists(best_ckpt_path):
            print(f"--no_resume set: ignoring existing checkpoints in {args.out_dir}")
    elif resume_path is None:
        # ckpt_latest.pt is only written every --ckpt_interval steps, while
        # ckpt_best.pt is written every --eval_interval steps (whenever val loss
        # improves), so it can be further along if the process was killed between
        # two ckpt_interval saves. Resume from whichever actually has the higher
        # step, instead of always preferring ckpt_latest.pt and silently losing
        # progress.
        candidates = [p for p in (latest_ckpt_path, best_ckpt_path) if os.path.exists(p)]
        steps = {}
        for p in candidates:
            try:
                probe = torch.load(p, map_location="cpu", weights_only=False)
                # compare on the step each would *resume at*, not the step it
                # records: the two files disagree on what `step` means (see
                # build_checkpoint's next_step comment)
                steps[p] = probe.get("next_step", probe["step"] + 1)
                # this pulled a whole checkpoint into RAM to read one integer;
                # drop it before probing the next candidate rather than holding
                # both (and then the last one, for the rest of the run)
                del probe
            except Exception as e:
                # a truncated/corrupt file (bad copy, disk error, kill mid-write
                # on an older version without atomic saves) must not crash-loop
                # the whole process under systemd Restart=on-failure - drop it
                # and fall back to whichever other checkpoint is still readable
                print(f"warning: checkpoint at {p} is unreadable ({e}); ignoring it")
        if steps:
            resume_path = max(steps, key=steps.get)
        if resume_path is not None:
            print(f"found existing checkpoint at {resume_path}; resuming automatically")
        elif candidates:
            print(f"all checkpoints in {args.out_dir} were unreadable; starting from scratch")
    # ------------------------------------------------------------------------

    # Data
    train_loader = Dataloader(B=Batch, T=Sequence_length, split="train", data_dir=args.data_dir)
    val_loader = Dataloader(B=Batch, T=Sequence_length, split="val", data_dir=args.data_dir)

    # Model config. This is the single place the attention implementation is
    # chosen; everything downstream (the blocks, the profiler, the checkpoint,
    # metrics.json) reads it from here instead of importing a concrete class.
    model_config = ModelConfig()
    if args.attention is not None:
        model_config.attention = args.attention

    # lr/batch_size/seq_len/total_batches/max_steps as of this run, embedded in every
    # checkpoint and checked against on resume (see below) so a stale flag in a
    # restart script can't silently change the LR schedule / effective batch size /
    # data stride mid-run.
    training_args = {
        "lr": lr,
        "max_steps": max_steps,
        "batch_size": Batch,
        "seq_len": Sequence_length,
        "total_batches": Total_batches,
    }

    ckpt = None
    if resume_path:
        print(f"resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        # A checkpoint's weights only fit the attention they were trained with,
        # so adopt the name recorded in it rather than silently rebuilding a
        # different architecture and failing in load_state_dict.
        ckpt_attention = getattr(ckpt.get("model_config"), "attention", None)
        if ckpt_attention is not None and ckpt_attention != model_config.attention:
            if args.attention is not None:
                raise SystemExit(
                    f"--attention {args.attention!r} conflicts with the checkpoint at "
                    f"{resume_path}, which was trained with {ckpt_attention!r}. Pass "
                    "--no_resume (or a different --out_dir) to train the new attention "
                    "from scratch."
                )
            print(f"checkpoint was trained with attention={ckpt_attention!r}; using that "
                  f"instead of the config default {model_config.attention!r}")
            model_config.attention = ckpt_attention

        # Detect drifted hyperparameters (e.g. a stale flag in a restart script).
        # max_steps is exempt from the hard fail: resuming to train longer than
        # originally planned is a normal thing to do, so it's only a warning.
        ckpt_training_args = ckpt.get("training_args")
        if ckpt_training_args is not None:
            hard_checked = ("lr", "batch_size", "seq_len", "total_batches")
            hard_mismatches = {
                k: (ckpt_training_args[k], training_args[k]) for k in hard_checked
                if k in ckpt_training_args and ckpt_training_args[k] != training_args[k]
            }
            if hard_mismatches and not args.force_resume:
                detail = "; ".join(f"{k}: checkpoint={old!r} vs this run={new!r}"
                                    for k, (old, new) in hard_mismatches.items())
                raise SystemExit(
                    f"resume hyperparameter mismatch ({detail}). Continuing would "
                    "silently change the LR schedule, effective batch size, or data "
                    "stride mid-run. Fix the flags to match the checkpoint, or pass "
                    "--force_resume if the change is intentional."
                )
            if hard_mismatches:
                detail = "; ".join(f"{k}: checkpoint={old!r} vs this run={new!r}"
                                    for k, (old, new) in hard_mismatches.items())
                print(f"warning: resume hyperparameter mismatch ({detail}); continuing "
                      "because --force_resume was passed")
            if ckpt_training_args.get("max_steps") != training_args["max_steps"]:
                print(f"note: max_steps changed across resume (checkpoint="
                      f"{ckpt_training_args.get('max_steps')!r} vs this run="
                      f"{training_args['max_steps']!r}); the cosine LR schedule will "
                      "be recomputed against the new max_steps")
        else:
            print("checkpoint has no embedded training_args (from an older run); "
                  "cannot verify lr/batch_size/seq_len/total_batches/max_steps "
                  "consistency across resume")

    # Model
    model = GPT(model_config)
    print(f"attention implementation: {model_config.attention}")
    model.to(device)

    # Build the optimizer against the un-compiled model (compiling first can prefix
    # parameter names, which is best avoided before we've grabbed named_parameters()).
    optimizer = configure_optimizers(
        model, weight_decay=0.1, learning_rate=lr, betas=(0.9, 0.95), device_type=device_type
    )

    raw_model = model  # keep an uncompiled reference for clean checkpointing

    start_step = 0
    best_val_loss = float("inf")
    checkpoint_stats = None
    if ckpt is not None:
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        # step+1 is only the right resume point for checkpoints saved after the
        # optimizer update; next_step records it explicitly (absent on older
        # checkpoints, which were all ckpt_latest.pt-style saves)
        start_step = ckpt.get("next_step", ckpt["step"] + 1)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        train_loader.current_position = ckpt.get("train_loader_position", 0)

        # Restore RNG state if available and valid. torch.set_rng_state /
        # torch.cuda.set_rng_state_all both require CPU ByteTensors, but `ckpt`
        # was loaded with map_location=device, so on a CUDA run these tensors
        # arrive on the GPU and must be moved back to CPU explicitly - otherwise
        # set_rng_state raises "RNG state must be a torch.ByteTensor" (the dtype
        # is still uint8 on GPU, so the format check below doesn't catch it) and
        # restoration silently no-ops on every CUDA run.
        rng_state = ckpt.get("rng_state")
        if rng_state is not None:
            try:
                if isinstance(rng_state, torch.Tensor) and rng_state.dtype == torch.uint8:
                    torch.set_rng_state(rng_state.cpu())
                else:
                    print(f"warning: checkpoint RNG state has unexpected format (dtype={getattr(rng_state, 'dtype', 'N/A')}); skipping RNG restoration")
            except Exception as e:
                print(f"warning: failed to restore RNG state: {e}; continuing with fresh RNG")

        if ckpt.get("cuda_rng_state") is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["cuda_rng_state"]])
            except Exception as e:
                print(f"warning: failed to restore CUDA RNG state: {e}; continuing with fresh CUDA RNG")

        checkpoint_stats = ckpt.get("metrics_stats")
        if checkpoint_stats is None:
            print("checkpoint has no embedded metrics stats (from an older run); "
                  "resumed aggregate metrics (step time, tokens/sec, peak memory, attention) "
                  "may double-count steps between this checkpoint and the last metrics.json write")
        print(f"resumed at step {start_step}, best_val_loss so far: {best_val_loss:.4f}")

        # Release the checkpoint now that everything has been read out of it.
        # It was loaded with map_location=device, so every tensor in it is on the
        # GPU, and load_state_dict *copies* into the model's parameters instead of
        # aliasing them - so without this the checkpoint's own copy of the weights
        # (~0.5 GB here) stays resident for the entire run. That leaves a resumed
        # run with less headroom than the fresh run that produced the checkpoint,
        # which can turn a restart into an OOM that repeats on every restart.
        # (The optimizer state genuinely does alias, so this frees only dead memory.)
        del ckpt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("no checkpoint found (or --no_resume set); starting from scratch")

    # Per-layer hooks force graph breaks, so attention profiling and torch.compile
    # can't both be on if the numbers are to describe one consistent execution mode.
    use_compile = args.compile and device == "cuda" and not args.profile_attn
    if args.compile and device == "cuda" and args.profile_attn:
        print("torch.compile disabled: --profile_attn instruments the attention layers "
              "with hooks, which break the compiled graph. Use --no-profile_attn for "
              "compiled-throughput numbers.")
    if use_compile:
        model = torch.compile(model)

    tracker_kwargs = dict(
        path=args.metrics_path,
        run_info={
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "torch_version": torch.__version__,
            "autocast_dtype": str(ptdtype),
            "torch_compile": use_compile,
            "attention_profiling": args.profile_attn,
            "attention": model_config.attention,
            "seed": args.seed,
            "num_parameters": sum(p.numel() for p in raw_model.parameters()),
            "num_parameters_non_embedding": sum(
                p.numel() for p in raw_model.parameters()
            ) - raw_model.transformer.wpe.weight.numel(),
            "model_config": vars(model_config),
            "batch_size": Batch,
            "seq_len": Sequence_length,
            "grad_accumulation_steps": grad_accumulation_steps,
            "max_steps": max_steps,
            "max_lr": lr,
            "min_lr": TrainingConfig.min_lr,
            "warmup_steps": TrainingConfig.warmup_steps,
            "eval_interval": args.eval_interval,
            "eval_iters": args.eval_iters,
            "data_dir": args.data_dir,
        },
        tokens_per_step=Batch * Sequence_length * grad_accumulation_steps,
        warmup_steps_skipped=args.metrics_warmup_steps,
        log_interval=args.log_interval,
        device_type=device_type,
    )
    if resume_path:
        tracker = MetricsTracker.resume(
            start_step=start_step, checkpoint_stats=checkpoint_stats, **tracker_kwargs
        )
    else:
        tracker = MetricsTracker(**tracker_kwargs)
    if args.profile_attn:
        # hooks every BaseAttention subclass, so a swapped-in implementation is
        # profiled without touching this line
        tracker.attach_attention_profiler(raw_model, BaseAttention)
    print(f"writing metrics to {os.path.abspath(args.metrics_path)}")

    # holds the most recently computed validation loss, so the periodic "latest"
    # checkpoint below always has a value to record even between eval_interval steps
    val_loss = best_val_loss if best_val_loss != float("inf") else float("nan")

    # latches once training blows up to NaN; see the divergence check in the loop
    diverged = False

    model.train()

    for step in range(start_step, max_steps):
        last_step = step == max_steps - 1

        if step % args.eval_interval == 0 or last_step:
            val_loss = estimate_loss(model, val_loader, args.eval_iters, device, device_type, ptdtype)
            val_ppl = safe_exp(val_loss)
            tracker.log_validation(step, val_loss)
            tracker.write()
            print(f"step {step}: val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}")

            agg = tracker.aggregates()
            if agg["measured_steps"]:
                attn = agg.get("attention") or {}
                peak_mib = agg.get("avg_peak_memory_mib") or {}
                line = (
                    f"  avg tok/sec: {agg['avg_tokens_per_sec']:.2f}"
                    f" | avg step time: {agg['avg_step_time_ms']:.2f}ms"
                )
                if agg.get("avg_model_time_ms") is not None:
                    line += f" | model time: {agg['avg_model_time_ms']:.2f}ms"
                if attn.get("avg_time_ms") is not None:
                    line += f" | attn time: {attn['avg_time_ms']:.2f}ms"
                if peak_mib.get("allocated") is not None:
                    line += f" | avg peak mem: {peak_mib['allocated']:.1f} MiB"
                print(line)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt_path = os.path.join(args.out_dir, "ckpt_best.pt")
                save_checkpoint_atomic(
                    # step hasn't trained yet at this point in the loop body
                    build_checkpoint(raw_model, optimizer, step, val_loss, best_val_loss, train_loader,
                                     tracker, model_config, training_args, next_step=step),
                    best_ckpt_path,
                )
                print(f"saved best checkpoint to {best_ckpt_path}")

        # linear warmup + cosine decay
        step_lr = get_lr(step, TrainingConfig.warmup_steps, max_steps, lr, TrainingConfig.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = step_lr

        # Timed region starts here so the validation pass above is excluded from
        # the step time and the peak-memory high-water mark.
        tracker.start_step(step)

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        # Gradient accumulation: sum gradients over several micro-batches before
        # stepping the optimizer, to simulate a larger effective batch size
        # (Total_batches tokens) than fits in memory at once.
        for micro_step in range(grad_accumulation_steps):
            tracker.timer.start("data")
            x, y = train_loader.next_batch()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            tracker.timer.stop("data")

            tracker.timer.start("forward")
            with torch.autocast(device_type=device_type, dtype=ptdtype):
                logits = model(x)
                loss = calculate_loss(logits, y)
            tracker.timer.stop("forward")

            loss_accum += loss.detach()
            # average the loss over accumulation steps since gradients add up otherwise
            loss = loss / grad_accumulation_steps
            tracker.timer.start("backward")
            loss.backward()
            tracker.timer.stop("backward")

        # Clip the global gradient norm to 1.0 to prevent exploding gradients.
        tracker.timer.start("optimizer")
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tracker.timer.stop("optimizer")

        train_loss = loss_accum.item() / grad_accumulation_steps
        grad_norm = norm.item()
        # end_step does the single per-step cuda synchronize and resolves the timers
        tracker.end_step(step, loss=train_loss, lr=step_lr, grad_norm=grad_norm)

        # Has this step poisoned the weights? Either symptom is enough:
        #   * non-finite loss  - the forward pass already saw NaN parameters;
        #   * non-finite grad norm - clip_grad_norm_ scales every gradient by
        #     max_norm/(total_norm + 1e-6) clamped to 1, which is NaN when the
        #     norm is NaN and 0 when it is inf (and 0 * inf is NaN again), so
        #     optimizer.step() has just written NaNs into the parameters even
        #     though the loss that produced them was still finite.
        # Checking the loss alone would let exactly one poisoned checkpoint
        # through, on the step where the gradients blew up but the loss had not
        # caught up yet.
        if not diverged and not (math.isfinite(train_loss) and math.isfinite(grad_norm)):
            # Latch rather than re-test per step: NaN parameters stay NaN, and
            # latching means a single blow-up can never be "recovered from" into
            # writing a poisoned checkpoint. Announce it here, at the step it
            # happens, rather than at the next checkpoint boundary - which with
            # the default ckpt_interval can be hundreds of steps later.
            diverged = True
            print(f"WARNING: step {step} diverged (train_loss={train_loss}, "
                  f"grad_norm={grad_norm}). The parameters are NaN from here on, so "
                  f"{latest_ckpt_path} will no longer be updated and the last good "
                  "checkpoint survives. Stop the run and investigate.")

        if step % args.log_interval == 0 or last_step:
            train_ppl = safe_exp(train_loss)
            print(f"step {step}: train_loss {train_loss:.4f} | train_ppl {train_ppl:.2f}")

        if step % args.ckpt_interval == 0 or last_step:
            # Overwriting ckpt_latest.pt with diverged weights would destroy the
            # last good resume point, and every later restart would faithfully
            # reload the garbage. Keep the previous file instead, so the run can
            # be resumed from before the blow-up. ckpt_best.pt is already safe:
            # `val_loss < best_val_loss` is False for NaN, so it is never
            # overwritten by a diverged eval.
            if diverged:
                print(f"step {step}: diverged, leaving {latest_ckpt_path} untouched")
            else:
                save_checkpoint_atomic(
                    # step is fully trained by the time we get here
                    build_checkpoint(raw_model, optimizer, step, val_loss, best_val_loss, train_loader,
                                     tracker, model_config, training_args, next_step=step + 1),
                    latest_ckpt_path,
                )
                print(f"saved latest checkpoint to {latest_ckpt_path}")

    tracker.write(status="completed")
    print(f"wrote metrics to {os.path.abspath(args.metrics_path)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # metrics.json is rewritten on every logged step, so an interrupted run
        # still leaves the curves and aggregates collected so far on disk.
        print("interrupted; metrics.json holds everything up to the last logged step")