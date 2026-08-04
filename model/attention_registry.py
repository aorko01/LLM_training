"""
Single swap point for attention implementations.

Everything that needs to know "which attention are we using" goes through here:

  * `Attention_Block` builds its attention with `build_attention(config)` instead
    of importing a concrete class, so the block never has to change.
  * `train.py` profiles `isinstance(module, BaseAttention)`, so any registered
    implementation is timed and memory-measured with no changes to the profiler.
  * `ModelConfig.attention` (or `--attention` on the command line) picks the name.

Adding a new one is one file plus zero edits elsewhere::

    # model/custom_attention.py
    from .attention_registry import BaseAttention, register_attention

    @register_attention("custom")
    class CustomAttention(BaseAttention):
        def __init__(self, config):
            super().__init__(config)
            ...
        def forward(self, X):
            ...

Then run `python train.py --attention custom`. Modules inside the `model`
package are imported on first lookup, so simply dropping the file in is enough
for the name to be found.

As an escape hatch, a name containing a colon is treated as an import path to a
class living outside this package, e.g. ``--attention my_research.attn:MyAttn``.
"""

import importlib
import pkgutil

import torch.nn as nn


class BaseAttention(nn.Module):
    """
    Base class every attention implementation must subclass.

    It exists so the rest of the codebase can refer to "an attention layer"
    without naming a concrete class - most importantly the profiler in
    `metrics.py`, which hooks every module that is an instance of this.

    Subclasses take the model config and consume whatever fields they need
    (`n_embd`, `n_head`, `block_size`, `dropout`, `bias`, ...).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, X):
        raise NotImplementedError


_REGISTRY = {}
_discovered = False


def _normalize(name):
    return name.strip().lower().replace("-", "_")


def register_attention(*names):
    """
    Class decorator that registers an attention implementation under one or more
    names. Names are matched case-insensitively, with '-' and '_' equivalent.
    """

    def decorator(cls):
        if not (isinstance(cls, type) and issubclass(cls, BaseAttention)):
            raise TypeError(
                f"{cls.__name__} must subclass BaseAttention to be registered "
                "(train.py profiles attention layers by that base class)"
            )
        for name in names:
            key = _normalize(name)
            existing = _REGISTRY.get(key)
            if existing is not None and existing is not cls:
                raise ValueError(
                    f"attention name {name!r} is already registered to "
                    f"{existing.__module__}.{existing.__name__}"
                )
            _REGISTRY[key] = cls
        return cls

    return decorator


def _discover():
    """
    Import every module in the `model` package once, so that any file defining a
    `@register_attention` class is registered without needing an explicit import.
    """
    global _discovered
    if _discovered:
        return
    _discovered = True  # set first: the imports below re-enter this module

    package = importlib.import_module(__package__)
    for info in pkgutil.walk_packages(package.__path__, prefix=f"{__package__}."):
        if info.name == __name__ or info.name.rsplit(".", 1)[-1].startswith("_"):
            continue
        importlib.import_module(info.name)


def _load_from_path(spec):
    """Resolve a 'module.path:ClassName' spec into a class."""
    module_path, _, class_name = spec.partition(":")
    module = importlib.import_module(module_path)
    try:
        cls = getattr(module, class_name)
    except AttributeError:
        raise ValueError(f"module {module_path!r} has no attribute {class_name!r}") from None
    if not (isinstance(cls, type) and issubclass(cls, BaseAttention)):
        raise TypeError(f"{spec} must be a BaseAttention subclass")
    return cls


def available_attentions():
    """Sorted list of registered names, for CLI help and error messages."""
    _discover()
    return sorted(_REGISTRY)


def get_attention(name):
    """Look up a registered attention class by name (or by 'module:Class' path)."""
    if ":" in name:
        return _load_from_path(name)
    _discover()
    try:
        return _REGISTRY[_normalize(name)]
    except KeyError:
        raise KeyError(
            f"unknown attention {name!r}; available: {', '.join(available_attentions())}. "
            "Add a new one by dropping a module in model/ with a "
            "@register_attention(...) BaseAttention subclass."
        ) from None


def build_attention(config, name=None):
    """
    Instantiate the attention implementation selected by `config.attention`
    (or by an explicit `name` override).
    """
    if name is None:
        name = getattr(config, "attention", "vanilla")
    return get_attention(name)(config)
