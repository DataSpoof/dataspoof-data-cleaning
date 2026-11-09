"""Compatibility with AutomatedCleaning, the library's name before 2.1.0.

Objects saved by AutomatedCleaning (raw pickles of models, MLflow pyfunc wrappers, anything
that recorded ``automatedcleaning.<module>.<Class>``) must keep loading. Importing ``mlpilot``
registers an import hook that resolves ``automatedcleaning`` and ``automatedcleaning.*`` to the
same modules under ``mlpilot``, so classes keep their identity (``isinstance`` still works).

The hook steps aside for a real ``automatedcleaning`` package (the transitional
``AutomatedCleaning`` release, which itself forwards here), and only answers when the name is
missing or is just an empty namespace folder.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys

OLD, NEW = "automatedcleaning", "mlpilot"


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target: str):
        self.target = target

    def create_module(self, spec):
        return importlib.import_module(self.target)      # the real module object, not a copy

    def exec_module(self, module):
        pass


def _is_real(spec) -> bool:
    """A regular module or package (not an empty namespace package, e.g. a leftover folder)."""
    return spec is not None and spec.loader is not None and spec.origin not in (None, "namespace")


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != OLD and not fullname.startswith(OLD + "."):
            return None
        parent = sys.modules.get(OLD)
        aliased_parent = fullname != OLD and getattr(parent, "__name__", None) == NEW
        # Under an aliased parent the path finder would load mlpilot's files a *second* time as new
        # modules (duplicate classes), so submodules always map; otherwise a real package wins.
        if not aliased_parent and _is_real(importlib.machinery.PathFinder.find_spec(fullname, path)):
            return None
        new = NEW + fullname[len(OLD):]
        try:
            importlib.util.find_spec(new)
        except (ImportError, ValueError):
            return None
        is_pkg = hasattr(importlib.import_module(new), "__path__")
        return importlib.machinery.ModuleSpec(fullname, _AliasLoader(new), is_package=is_pkg)


def install() -> None:
    """Register the hook once, ahead of the path finder (it defers to real packages itself)."""
    if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _AliasFinder())


import importlib.util  # noqa: E402  (used above; kept explicit for type checkers)
