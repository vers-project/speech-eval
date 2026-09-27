"""Metric implementations.

Importing this package registers every built-in metric, so
``speech_eval.core.build_metric`` can resolve a config's ``type`` without the
caller importing the module by hand.  Each new metric family adds one import
here and one optional dependency extra in ``pyproject.toml``.
"""
from __future__ import annotations

# Importing a metric module never imports its backend: model and toolkit imports
# live in Metric.load(), which raises a message naming the extra to install.  So
# registration is safe regardless of which extras this environment has.
from speech_eval.metrics import asr  # noqa: F401
from speech_eval.metrics import egemaps  # noqa: F401
from speech_eval.metrics import f0  # noqa: F401
from speech_eval.metrics import formants  # noqa: F401
from speech_eval.metrics import level  # noqa: F401
from speech_eval.metrics import naturalness  # noqa: F401
from speech_eval.metrics import speaker  # noqa: F401
