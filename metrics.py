"""
Metrics collection for training runs, serialised to `metrics.json`.

Two kinds of numbers are recorded, because they answer different questions:

* **Curves** - train loss / val loss / val perplexity, appended every N steps.
  These are per-step time series and are meant to be plotted. Train points are
  appended every `log_interval` steps, val points every `eval_interval` steps.
* **Aggregates** - running averages of step time, tokens/sec, peak memory, and
  the time spent inside the model (forward+backward) vs. specifically inside
  the attention layers. These are kept as running sum/count accumulators
  rather than per-step sample lists, so `metrics.json` stays small no matter
  how long a run goes. The first q`warmup_steps_skipped` steps are excluded
  (allocator warm-up, cuDNN autotuning, `torch.compile` recompiles are noisy).

Timing on CUDA uses CUDA events instead of `time.perf_counter()`: kernel launches
are asynchronous, so a host-side timer around a launch measures the launch, not
the work. Events are recorded in-stream and resolved once per step after a single
`torch.cuda.synchronize()`, which keeps the instrumentation off the critical path.

`MetricsTracker.resume()` lets a restarted training run continue the same
`metrics.json` (curves + running aggregates) instead of starting a fresh report -
see its docstring below.
"""

import json
import math
import os
import time
from datetime import datetime, timezone

import torch

MIB = 1024 ** 2


class _RunningStat:
    """A mean, kept as a running sum/count instead of a stored sample list."""

    __slots__ = ("count", "sum")

    def __init__(self):
        self.count = 0
        self.sum = 0.0

    def add(self, value):
        self.count += 1
        self.sum += value

    @property
    def mean(self):
        return self.sum / self.count if self.count else None

    def to_dict(self):
        return {"count": self.count, "sum": self.sum}

    @classmethod
    def from_dict(cls, data):
        stat = cls()
        stat.count = data.get("count", 0)
        stat.sum = data.get("sum", 0.0)
        return stat


class _Timer:
    """
    Accumulates named durations for the current step.

    On CUDA every `start`/`stop` pair records a pair of CUDA events; the elapsed
    times are only read in `drain()`, which the caller invokes after synchronising.
    Events are pooled and reused so a long run doesn't allocate thousands of them.
    Repeated start/stop of the same label within a step sum together (e.g. the
    forward pass runs once per gradient-accumulation micro-step).
    """

    def __init__(self, device_type):
        self.cuda = device_type == "cuda"
        self._pool = []
        self._in_use = 0
        self._pending = []
        self._open = {}

    def _event(self):
        if self._in_use == len(self._pool):
            self._pool.append(torch.cuda.Event(enable_timing=True))
        event = self._pool[self._in_use]
        self._in_use += 1
        return event

    def start(self, label):
        if self.cuda:
            event = self._event()
            event.record()
            self._open[label] = event
        else:
            self._open[label] = time.perf_counter()

    def stop(self, label):
        started = self._open.pop(label, None)
        if started is None:  # stop without a matching start (e.g. profiler paused mid-step)
            return
        if self.cuda:
            end = self._event()
            end.record()
            self._pending.append((label, started, end))
        else:
            self._pending.append((label, (time.perf_counter() - started) * 1000.0))

    def drain(self):
        """Resolve and clear everything recorded since the last drain. Call after sync."""
        totals = {}
        for item in self._pending:
            if self.cuda:
                label, start, end = item
                totals[label] = totals.get(label, 0.0) + start.elapsed_time(end)
            else:
                label, elapsed_ms = item
                totals[label] = totals.get(label, 0.0) + elapsed_ms
        self._pending.clear()
        self._open.clear()
        self._in_use = 0
        return totals


class AttentionProfiler:
    """
    Measures wall time and peak memory inside the attention layers only.

    Memory is tracked in *segments*: on entering attention we read the running
    allocator peak, fold it into the pipeline peak, then call
    `reset_peak_memory_stats()` (which rebases the peak to the current allocation).
    On exit we read the peak again - that value is the true high-water mark reached
    while inside attention. Taking a max over every segment therefore still yields
    an exact whole-pipeline peak, without the attention measurement and the
    pipeline measurement clobbering each other.

    Two attention memory numbers are reported:
      * `absolute`  - high-water mark of the whole process while inside attention,
                      which includes everything earlier layers are still holding.
      * `delta_over_entry` - peak minus what was already allocated on entry, i.e.
                      the memory attributable to the attention layer itself.

    Note: this attaches forward/backward hooks, which force `torch.compile` to
    break the graph. `train.py` disables compilation when this profiler is active
    so the reported timings describe one consistent execution mode.
    """

    def __init__(self, model, module_cls, timer, tracker):
        self.timer = timer
        self.tracker = tracker
        self.enabled = False
        self.layers = 0
        self._entry_alloc = 0
        self.reset_step()

        for module in model.modules():
            if isinstance(module, module_cls):
                self.layers += 1
                module.register_forward_pre_hook(self._forward_pre)
                module.register_forward_hook(self._forward_post)
                module.register_full_backward_pre_hook(self._backward_pre)
                module.register_full_backward_hook(self._backward_post)

    def reset_step(self):
        self.peak_absolute = 0
        self.peak_delta = 0

    # --- shared segment bookkeeping -------------------------------------------------
    def _open_segment(self):
        if not self.tracker.cuda:
            return
        self.tracker.observe_peak()
        torch.cuda.reset_peak_memory_stats()
        self._entry_alloc = torch.cuda.memory_allocated()

    def _close_segment(self):
        if not self.tracker.cuda:
            return
        peak = torch.cuda.max_memory_allocated()
        self.peak_absolute = max(self.peak_absolute, peak)
        self.peak_delta = max(self.peak_delta, peak - self._entry_alloc)
        self.tracker.observe_peak()
        torch.cuda.reset_peak_memory_stats()

    # --- hooks ----------------------------------------------------------------------
    def _forward_pre(self, module, args):
        if not self.enabled:
            return
        self._open_segment()
        self.timer.start("attention_forward")

    def _forward_post(self, module, args, output):
        if not self.enabled:
            return
        self.timer.stop("attention_forward")
        self._close_segment()

    def _backward_pre(self, module, grad_output):
        if not self.enabled:
            return
        self._open_segment()
        self.timer.start("attention_backward")

    def _backward_post(self, module, grad_input, grad_output):
        if not self.enabled:
            return
        self.timer.stop("attention_backward")
        self._close_segment()


class MetricsTracker:
    """
    Collects curves + running aggregates and writes them to `metrics.json`.

    Typical use inside the training loop::

        tracker.start_step(step)
        ...  tracker.timer.start("forward") / .stop("forward") ...
        tracker.end_step(step, loss=..., lr=..., grad_norm=...)

    `end_step` performs the single per-step CUDA synchronize, so the caller should
    not add another one.

    For a fresh run, construct normally with `MetricsTracker(...)`. For a run
    resumed from a checkpoint, use `MetricsTracker.resume(...)` instead so the
    new process continues the same curves and running aggregates rather than
    starting `metrics.json` over from nothing.
    """

    def __init__(self, path, run_info, tokens_per_step, warmup_steps_skipped,
                 log_interval, device_type):
        self.path = path
        self.tokens_per_step = tokens_per_step
        self.warmup_steps_skipped = warmup_steps_skipped
        self.log_interval = log_interval
        self.cuda = device_type == "cuda"
        self.timer = _Timer(device_type)
        self.attention = None

        self.run_info = dict(run_info)
        self.run_info.update({
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tokens_per_step": tokens_per_step,
            "warmup_steps_skipped": warmup_steps_skipped,
            "log_interval": log_interval,
        })

        self.train_curve = []
        self.val_curve = []

        # Running aggregates, only accumulated for steps >= warmup_steps_skipped.
        # Kept as sum/count rather than per-step lists so metrics.json doesn't
        # grow with run length.
        self._step_time = _RunningStat()
        self._tokens_per_sec = _RunningStat()
        self._phase_time = {}  # label -> _RunningStat, e.g. "forward"/"backward"
        self._attn_forward = _RunningStat()
        self._attn_backward = _RunningStat()
        self._peak_alloc_avg = _RunningStat()
        self._peak_reserved_avg = _RunningStat()
        self._attn_peak_absolute = 0  # running max, not averaged
        self._attn_peak_delta = 0    # running max, not averaged

        # instantaneous per-step high-water marks, reset every start_step()
        self._peak_alloc = 0
        self._peak_reserved = 0

        self._t0 = None
        self._measuring = False
        self._wall_start = time.time()

    @classmethod
    def resume(cls, path, run_info, tokens_per_step, warmup_steps_skipped,
               log_interval, device_type, start_step=0, checkpoint_stats=None):
        """
        Reconstruct a MetricsTracker from an existing `metrics.json`, so a
        resumed training run continues the same curves and running aggregates
        instead of starting a fresh report at `start_step`.

        Falls back to a normal, empty tracker if `path` doesn't exist yet
        (e.g. resuming a checkpoint from a run that never got far enough to
        write metrics), or if it exists but can't be parsed.

        `start_step` is the step the resumed run will begin at (i.e. the
        checkpoint's `step + 1`). Any prior curve entries at or beyond that
        step are dropped - if the checkpoint being resumed from was saved
        after the last metrics.json flush (or the process crashed between the
        two), this avoids duplicate/stale curve entries.

        The running aggregates (step time, tokens/sec, peak memory, attention
        time) are cumulative sum/count accumulators, not per-step samples, so
        they can't be trimmed by step the way the curves are. metrics.json is
        written more often than checkpoints (every `log_interval`/
        `eval_interval` steps vs. every `ckpt_interval` steps), so its `_stats`
        can be ahead of the checkpoint being resumed from; blindly restoring
        it would re-add the steps between the checkpoint and the crash a
        second time as the run redoes them. `checkpoint_stats` - the
        `stats()` snapshot embedded in the checkpoint itself (see `train.py`'s
        `build_checkpoint`) - is saved atomically with `step`, so it's exactly
        in sync with `start_step` and is used instead when available. Only
        checkpoints written before this was added lack it, in which case this
        falls back to metrics.json's `_stats` and a handful of steps around
        the crash/restart boundary may be double-counted in the averages.
        """
        tracker = cls(path, run_info, tokens_per_step, warmup_steps_skipped,
                      log_interval, device_type)

        if not os.path.exists(path):
            return tracker

        try:
            with open(path) as handle:
                prior = json.load(handle)
            if not isinstance(prior, dict):
                raise ValueError(f"expected a JSON object, got {type(prior).__name__}")
        except (OSError, ValueError) as e:
            # write() is atomic (tmp file + os.replace), so a torn report should
            # be unreachable - but a disk error or a bad copy between machines
            # can still leave one unparseable, and losing the old curves must
            # not take down the training run itself (which, under an
            # auto-restarting supervisor, would crash-loop it). Continue with a
            # fresh report; the checkpoint still carries the running aggregates,
            # so only the historical curve points are lost. The unreadable file
            # is kept aside rather than silently overwritten by the first write.
            print(f"warning: existing metrics at {path} are unreadable ({e}); "
                  "starting a fresh report for this run")
            try:
                os.replace(path, path + ".corrupt")
                print(f"moved the unreadable report aside to {path}.corrupt")
            except OSError as move_error:
                print(f"warning: could not preserve it ({move_error}); "
                      "it will be overwritten on the next write")
            return tracker

        # Keep the original run's start time so elapsed_wall_clock_sec spans
        # the whole job rather than resetting on every restart; keep a
        # breadcrumb of every resume instead.
        prior_run = prior.get("run", {})
        if "started_at" in prior_run:
            tracker.run_info["started_at"] = prior_run["started_at"]
            tracker._wall_start = datetime.fromisoformat(prior_run["started_at"]).timestamp()
        resumed_at = list(prior_run.get("resumed_at", []))
        resumed_at.append(datetime.now(timezone.utc).isoformat(timespec="seconds"))
        tracker.run_info["resumed_at"] = resumed_at

        tracker.train_curve = [row for row in prior.get("train", []) if row["step"] < start_step]
        tracker.val_curve = [row for row in prior.get("val", []) if row["step"] < start_step]

        stats = checkpoint_stats if checkpoint_stats is not None else prior.get("_stats", {})
        tracker._step_time = _RunningStat.from_dict(stats.get("step_time", {}))
        tracker._tokens_per_sec = _RunningStat.from_dict(stats.get("tokens_per_sec", {}))
        tracker._phase_time = {
            label: _RunningStat.from_dict(value)
            for label, value in stats.get("phase_time", {}).items()
        }
        tracker._attn_forward = _RunningStat.from_dict(stats.get("attn_forward", {}))
        tracker._attn_backward = _RunningStat.from_dict(stats.get("attn_backward", {}))
        tracker._peak_alloc_avg = _RunningStat.from_dict(stats.get("peak_alloc_avg", {}))
        tracker._peak_reserved_avg = _RunningStat.from_dict(stats.get("peak_reserved_avg", {}))
        tracker._attn_peak_absolute = stats.get("attn_peak_absolute", 0)
        tracker._attn_peak_delta = stats.get("attn_peak_delta", 0)

        return tracker

    def attach_attention_profiler(self, model, module_cls):
        self.attention = AttentionProfiler(model, module_cls, self.timer, self)
        self.run_info["attention_layers_profiled"] = self.attention.layers
        return self.attention

    # --- memory ---------------------------------------------------------------------
    def observe_peak(self):
        """Fold the allocator's current running peak into this step's high-water marks."""
        if not self.cuda:
            return
        self._peak_alloc = max(self._peak_alloc, torch.cuda.max_memory_allocated())
        self._peak_reserved = max(self._peak_reserved, torch.cuda.max_memory_reserved())

    # --- step lifecycle -------------------------------------------------------------
    def start_step(self, step):
        """Begin timing a step. Call *after* any validation pass so eval memory and
        eval time are excluded from the training-step numbers."""
        self._measuring = step >= self.warmup_steps_skipped
        self._peak_alloc = 0
        self._peak_reserved = 0
        if self.cuda:
            torch.cuda.reset_peak_memory_stats()
        if self.attention is not None:
            self.attention.enabled = True
            self.attention.reset_step()
        self._t0 = time.perf_counter()

    def end_step(self, step, loss, lr, grad_norm):
        """Close out a step: synchronize once, resolve timers, update aggregates."""
        if self.cuda:
            torch.cuda.synchronize()
        step_ms = (time.perf_counter() - self._t0) * 1000.0
        tokens_per_sec = self.tokens_per_step * 1000.0 / step_ms

        phases = self.timer.drain()
        self.observe_peak()
        if self.attention is not None:
            self.attention.enabled = False

        if self._measuring:
            self._step_time.add(step_ms)
            self._tokens_per_sec.add(tokens_per_sec)
            for label, value in phases.items():
                self._phase_time.setdefault(label, _RunningStat()).add(value)
            if self.cuda:
                self._peak_alloc_avg.add(self._peak_alloc)
                self._peak_reserved_avg.add(self._peak_reserved)
            if self.attention is not None:
                self._attn_forward.add(phases.get("attention_forward", 0.0))
                self._attn_backward.add(phases.get("attention_backward", 0.0))
                self._attn_peak_absolute = max(self._attn_peak_absolute, self.attention.peak_absolute)
                self._attn_peak_delta = max(self._attn_peak_delta, self.attention.peak_delta)

        if step % self.log_interval == 0 or step == self.run_info.get("max_steps", 0) - 1:
            self.train_curve.append({
                "step": step,
                "train_loss": loss,
                "train_perplexity": safe_exp(loss),
                "lr": lr,
                "grad_norm": grad_norm,
                "step_time_ms": step_ms,
                "tokens_per_sec": tokens_per_sec,
                "tokens_seen": (step + 1) * self.tokens_per_step,
            })
            # Flush on every logged step. The file is small and a step takes seconds,
            # so this is free, and a killed run still leaves complete metrics behind.
            self.write()

        return step_ms, tokens_per_sec, phases

    def log_validation(self, step, val_loss):
        self.val_curve.append({
            "step": step,
            "val_loss": val_loss,
            "val_perplexity": safe_exp(val_loss),
            "tokens_seen": step * self.tokens_per_step,
        })

    # --- output ---------------------------------------------------------------------
    def aggregates(self):
        forward = self._phase_time.get("forward")
        backward = self._phase_time.get("backward")
        model_time_ms = None
        if forward and backward and forward.count and backward.count:
            model_time_ms = forward.mean + backward.mean

        out = {
            "measured_steps": self._step_time.count,
            "note": (
                f"the first {self.warmup_steps_skipped} steps are excluded; "
                "these are running averages per optimizer step (all gradient-accumulation "
                "micro-steps), accumulated across every resumed process sharing this metrics.json"
            ),
            "avg_step_time_ms": self._step_time.mean,
            "avg_tokens_per_sec": self._tokens_per_sec.mean,
            "avg_model_time_ms": model_time_ms,
        }

        if self.cuda:
            out["avg_peak_memory_mib"] = {
                "allocated": (
                    self._peak_alloc_avg.mean / MIB if self._peak_alloc_avg.mean is not None else None
                ),
                "reserved": (
                    self._peak_reserved_avg.mean / MIB if self._peak_reserved_avg.mean is not None else None
                ),
            }

        if self.attention is not None:
            attn_time_ms = None
            if self._attn_forward.count:
                attn_time_ms = self._attn_forward.mean + self._attn_backward.mean
            attention = {
                "layers": self.attention.layers,
                "avg_time_ms": attn_time_ms,
                "fraction_of_step_time": (
                    attn_time_ms / self._step_time.mean
                    if attn_time_ms is not None and self._step_time.mean else None
                ),
            }
            if self.cuda:
                attention["peak_memory_mib"] = {
                    "absolute": self._attn_peak_absolute / MIB,
                    "delta_over_entry": self._attn_peak_delta / MIB,
                }
            out["attention"] = attention

        return out

    def stats(self):
        """
        The running sum/count accumulators that `aggregates()` computes means
        from. Persisted alongside the curves (as a handful of numbers, not
        per-step samples) so `resume()` can carry the averages forward across
        a restart. Also embedded in checkpoints (see `train.py`'s
        `build_checkpoint`) so a resume can restore the aggregates exactly as
        of the checkpointed step, instead of whatever metrics.json last
        happened to have on disk.
        """
        return {
            "step_time": self._step_time.to_dict(),
            "tokens_per_sec": self._tokens_per_sec.to_dict(),
            "phase_time": {k: v.to_dict() for k, v in self._phase_time.items()},
            "attn_forward": self._attn_forward.to_dict(),
            "attn_backward": self._attn_backward.to_dict(),
            "peak_alloc_avg": self._peak_alloc_avg.to_dict(),
            "peak_reserved_avg": self._peak_reserved_avg.to_dict(),
            "attn_peak_absolute": self._attn_peak_absolute,
            "attn_peak_delta": self._attn_peak_delta,
        }

    def snapshot(self):
        return {
            "run": self.run_info,
            "aggregates": self.aggregates(),
            "train": self.train_curve,
            "val": self.val_curve,
            "_stats": self.stats(),
        }

    def write(self, status="running"):
        """Atomically (re)write metrics.json. Cheap enough to call at every eval."""
        payload = self.snapshot()
        payload["run"]["status"] = status
        payload["run"]["elapsed_wall_clock_sec"] = time.time() - self._wall_start
        payload["run"]["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp_path, self.path)


def safe_exp(loss):
    """exp(loss), guarding against overflow on an early/diverged run."""
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")
