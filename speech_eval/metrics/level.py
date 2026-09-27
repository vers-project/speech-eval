"""Level and duration — the reference metric.

Needs no model and no download, so it doubles as the end-to-end test of the
runner and as the sanity check any evaluation should start with: an utterance
whose duration or level is not what the manifest implies is a data problem, and
finding that out before spending GPU hours on it is worth one cheap pass.

It is also the one metric that must **not** be level-normalised — it measures
the very quantity normalisation removes.  That contrast is why
``level_norm_dbfs`` lives in :class:`Requirements` rather than inside each
metric.
"""
from __future__ import annotations

from audio_utils.data.transforms import peak_dbfs, rms_dbfs

from speech_eval.core import Features, Metric, Requirements, UtteranceBatch, register_metric


@register_metric("level")
class LevelMetric(Metric):
    """Duration, RMS and peak level, and the crest factor between them.

    ``sample_rate=None``: level and duration are rate-independent, and a
    resampler applied for no reason would still filter the signal.
    """

    requirements = Requirements(sample_rate=None, mono=True, level_norm_dbfs=None)

    def compute(self, batch: UtteranceBatch) -> list[Features]:
        results = []
        for utterance in batch.utterances:
            rms = rms_dbfs(utterance.wav)
            peak = peak_dbfs(utterance.wav)
            results.append({
                "duration_s": utterance.duration_s,
                "sample_rate": utterance.sample_rate,
                "rms_dbfs": rms,
                "peak_dbfs": peak,
                # High for clipped or peaky signals, low for compressed ones —
                # a cheap flag for anything that went wrong in the audio chain.
                "crest_factor_db": peak - rms,
            })
        return results
