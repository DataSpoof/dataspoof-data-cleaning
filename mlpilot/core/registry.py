"""Step and function registry for the pipeline core.

Stateful step classes register themselves with :func:`register_step`. Plain ``df -> df``
functions become usable in pipelines either one at a time (:func:`register_function`) or by
exposing a whole module (:func:`register_function_namespace`).
"""

from __future__ import annotations

import inspect
from types import ModuleType
from typing import Callable

_STEPS: dict[str, type] = {}
_FUNCTIONS: dict[str, Callable] = {}
_NAMESPACES: list[tuple[ModuleType, frozenset]] = []


def register_step(cls):
    """Class decorator: make a ``Step`` subclass usable in configs under ``cls.name``."""
    name = getattr(cls, "name", "")
    if not name:
        raise ValueError(f"{cls.__name__} must define a non-empty 'name' to be registered.")
    _STEPS[name] = cls
    return cls


def register_function(fn=None, *, name=None):
    """Register a ``df -> df`` function so pipelines and saved pipelines can refer to it by name.

    Use directly (``register_function(my_fn)``) or as a decorator
    (``@register_function(name="my_step")``).
    """
    def _register(f):
        _FUNCTIONS[name or f.__name__] = f
        return f

    return _register(fn) if fn is not None else _register


def register_function_namespace(module: ModuleType, exclude=()):
    """Expose every public function defined in ``module`` as a pipeline step, minus ``exclude``."""
    _NAMESPACES.append((module, frozenset(exclude)))


def get_step_class(name: str):
    return _STEPS.get(name)


def _namespace_functions():
    """Yield ``(name, function)`` for usable functions, first registration of a name wins."""
    seen = set()
    for module, excluded in _NAMESPACES:
        for attr, obj in vars(module).items():
            if (attr in seen or attr.startswith("_") or attr in excluded
                    or not inspect.isfunction(obj) or obj.__module__ != module.__name__):
                continue
            seen.add(attr)
            yield attr, obj


def resolve_function(name: str):
    """Return the function registered under ``name``, or ``None``."""
    if name in _FUNCTIONS:
        return _FUNCTIONS[name]
    for attr, obj in _namespace_functions():
        if attr == name:
            return obj
    return None


def _summary(obj) -> str:
    doc = (inspect.getdoc(obj) or "").strip()
    return doc.splitlines()[0] if doc else ""


def available_steps() -> dict:
    """Return ``{name: {"kind", "description"}}`` for every step name a config can use.

    ``stateful`` steps learn their rules in ``fit`` and replay them in ``transform``.
    ``function`` steps re-run their logic on every batch. When a function shares its name
    with a stateful step, use it as ``fn:<name>``.
    """
    out = {name: {"kind": "stateful", "description": _summary(cls)}
           for name, cls in sorted(_STEPS.items())}
    funcs = dict(_namespace_functions())
    funcs.update(_FUNCTIONS)
    for name in sorted(funcs):
        key = f"fn:{name}" if name in out else name
        out[key] = {"kind": "function", "description": _summary(funcs[name])}
    return out
