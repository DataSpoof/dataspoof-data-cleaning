"""AutomatedCleaning is now MLPilot.

This package only forwards to ``mlpilot``: ``import automatedcleaning`` gives the same library,
and ``automatedcleaning.<module>`` is ``mlpilot.<module>``, so existing code and saved models keep
working. New code: ``pip install mlpilot`` and ``import mlpilot``.
"""

import sys
import warnings

warnings.warn("AutomatedCleaning has been renamed MLPilot: use `pip install mlpilot` and `import mlpilot`. "
              "The automatedcleaning package only forwards to it and gets no further updates.",
              FutureWarning, stacklevel=2)

import mlpilot as _mlpilot  # noqa: E402  (registers the automatedcleaning.* -> mlpilot.* import hook)

sys.modules[__name__] = _mlpilot
