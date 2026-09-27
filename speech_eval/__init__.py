"""Objective speech evaluation: per-utterance metrics, and comparisons over them.

Importing this package registers every built-in metric, so
:func:`speech_eval.core.build_metric` can resolve a config's ``type`` without the
caller remembering to import :mod:`speech_eval.metrics` by hand.  Forgetting that
import produced ``Unknown metric type 'level'. Available: []`` -- an empty
registry rather than a misspelling -- three separate times, which is two more than
a footgun of this size deserves.

It costs almost nothing: a metric module never imports its own backend.  Models
and toolkits are loaded in :meth:`speech_eval.core.Metric.load`, so registration
touches no GPU, no network and none of the optional extras.
"""
from speech_eval import metrics as _metrics  # noqa: F401
