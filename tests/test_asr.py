"""ASR metrics and the WER comparisons they feed.

The backend-independent contract is tested against a stub recogniser rather than
a real one: the point of these tests is that ``asr_whisper.hypothesis`` and
``asr_wav2vec2.hypothesis`` mean the same thing — same guards, same flag values,
same batch ordering — and that is settled by the base class, not by either
model.  A 3 GB download would test transformers, not this repo.

The real backends are exercised by ``test_asr_backends.py``-style checks at the
bottom, gated behind an environment variable so they never fire by accident.
"""
from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd
import pytest
import torch

from speech_eval.compare import (
    basic_normalize,
    pooled_wer,
    transcription_errors,
    word_errors,
)
from speech_eval.core import Utterance, UtteranceBatch, build_metric
from speech_eval.metrics.asr import (
    STATUS_EMPTY,
    STATUS_OK,
    STATUS_TOO_LONG,
    ASRMetric,
    Transcript,
)

jiwer = pytest.importorskip("jiwer", reason="needs the `asr` extra")

SAMPLE_RATE = 16_000


# ---------------------------------------------------------------------------
# A recogniser that transcribes nothing
# ---------------------------------------------------------------------------


class StubASRMetric(ASRMetric):
    """Returns a canned transcript per call and records what it was given.

    Deliberately not registered: the registry is global, and a test type in it
    would leak into ``available_metrics()``.
    """

    # What @register_metric would have set; supplied by hand so the name
    # resolves without the class entering the global registry.
    metric_type = "stub"
    backend = "stub"
    decoder = "stub_decoder"
    default_model_id = "stub/model"
    default_max_duration_s = 5.0

    def __init__(self, *args, texts: list[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.texts = texts
        self.calls: list[tuple[int, str]] = []

    def load(self, device) -> None:  # noqa: ANN001
        self._device = device
        self._loaded = True

    def _transcribe(self, waveforms, language):
        self.calls.append((len(waveforms), language))
        return [
            Transcript(
                self.texts[i] if self.texts else f"word {i}",
                -0.5,
            )
            for i in range(len(waveforms))
        ]


def utterance(utt_id: str, duration_s: float = 1.0, **meta) -> Utterance:
    samples = int(duration_s * SAMPLE_RATE)
    return Utterance(
        utt_id=utt_id,
        wav=torch.zeros(1, samples),
        sample_rate=SAMPLE_RATE,
        meta=meta,
    )


def make_metric(**kwargs) -> StubASRMetric:
    metric = StubASRMetric(**kwargs)
    metric.load(torch.device("cpu"))
    return metric


# ---------------------------------------------------------------------------
# The per-utterance contract
# ---------------------------------------------------------------------------


def test_one_result_per_utterance_in_batch_order():
    metric = make_metric(texts=["first one", "second"])
    batch = UtteranceBatch([utterance("a"), utterance("b")])
    results = metric.compute(batch)
    assert [r["hypothesis"] for r in results] == ["first one", "second"]
    assert [r["n_words"] for r in results] == [2, 1]


def test_provenance_is_on_every_row():
    metric = make_metric()
    row = metric.compute(UtteranceBatch([utterance("a")]))[0]
    assert row["backend"] == "stub"
    assert row["decoder"] == "stub_decoder"
    assert row["model_id"] == "stub/model"
    assert row["language"] == "en"
    assert row["language_source"] == "configured"
    assert row["duration_s"] == pytest.approx(1.0)


def test_status_is_never_null_so_resume_terminates():
    """The reason `status` exists at all.

    An utterance that legitimately decodes to "" writes an empty cell, which
    pandas reads back as NaN; without a guaranteed non-null column under the
    metric's prefix the runner would consider the row unfinished and
    re-transcribe it on every resume.
    """
    metric = make_metric(texts=[""])
    row = metric.compute(UtteranceBatch([utterance("a")]))[0]
    assert row["hypothesis"] == ""
    assert row["status"] == STATUS_OK

    frame = pd.DataFrame([{f"stub.{k}": v for k, v in row.items()}])
    reloaded = pd.read_csv(io.StringIO(frame.to_csv(index=False)))
    assert reloaded["stub.hypothesis"].isna().all(), "empty text does read back as NaN"
    assert reloaded.filter(like="stub.").notna().any(axis=1).all()


def test_resume_does_not_re_transcribe_an_empty_hypothesis(manifest_path, tmp_path):
    """The same claim, through the runner and a real CSV round trip.

    A conversion that destroys the speech decodes to "", which is a result and
    not a gap.  Without `status` the resumed run would find only NaN under the
    metric's prefix and transcribe the whole corpus again, every time.
    """
    from speech_eval.io import load_manifest
    from speech_eval.runner import run_metrics

    metadata = load_manifest(manifest_path, dataset_root=tmp_path / "audio")
    output_dir = tmp_path / "out"

    # One destroyed utterance among two decoded ones: the realistic shape.
    metric = StubASRMetric(texts=["", "spoke clearly", "spoke clearly"])
    run_metrics([metric], metadata, output_dir=output_dir)
    assert metric.calls, "first run should have decoded"

    resumed = StubASRMetric(texts=["", "spoke clearly", "spoke clearly"])
    run_metrics([resumed], metadata, output_dir=output_dir)
    assert resumed.calls == [], "resume re-transcribed rows that were already done"

    written = pd.read_csv(output_dir / "metrics.csv")
    assert written["stub.hypothesis"].isna().sum() == 1, "the empty decode reads as NaN"
    assert (written["stub.status"] == STATUS_OK).all()


def test_over_long_utterances_are_flagged_and_never_decoded():
    metric = make_metric()  # guard at 5 s
    batch = UtteranceBatch([utterance("short", 1.0), utterance("long", 6.0)])
    results = metric.compute(batch)

    assert results[0]["status"] == STATUS_OK
    assert results[1]["status"] == STATUS_TOO_LONG
    assert results[1]["hypothesis"] == ""
    assert np.isnan(results[1]["mean_logprob"])
    # Truncating instead would have turned every word past the cut into a
    # deletion and produced a plausible-looking, badly inflated WER.
    assert metric.calls == [(1, "en")], "the flagged utterance reached the backend"


def test_the_guard_can_be_disabled():
    metric = make_metric(max_duration_s=None)
    row = metric.compute(UtteranceBatch([utterance("long", 600.0)]))[0]
    assert row["status"] == STATUS_OK


def test_empty_audio_is_data_not_an_error():
    metric = make_metric()
    empty = Utterance(utt_id="a", wav=torch.zeros(1, 0), sample_rate=SAMPLE_RATE)
    row = metric.compute(UtteranceBatch([empty]))[0]
    assert row["status"] == STATUS_EMPTY
    assert metric.calls == []


def test_language_comes_from_metadata_when_present():
    metric = make_metric()
    rows = metric.compute(UtteranceBatch([
        utterance("a", language="fr"), utterance("b"),
    ]))
    assert (rows[0]["language"], rows[0]["language_source"]) == ("fr", "meta")
    assert (rows[1]["language"], rows[1]["language_source"]) == ("en", "configured")
    # One decode call per language: both backends decode under a single setting.
    assert sorted(metric.calls) == [(1, "en"), (1, "fr")]


def test_a_backend_with_a_fixed_language_ignores_the_column():
    class FixedLanguage(StubASRMetric):
        language_is_configurable = False

    metric = FixedLanguage()
    metric.load(torch.device("cpu"))
    row = metric.compute(UtteranceBatch([utterance("a", language="fr")]))[0]
    assert (row["language"], row["language_source"]) == ("en", "backend_fixed")


def test_sample_rate_mismatch_is_refused():
    metric = make_metric(sample_rate=16_000)
    wrong = Utterance(utt_id="a", wav=torch.zeros(1, 8000), sample_rate=8_000)
    with pytest.raises(ValueError, match="8000 Hz"):
        metric.compute(UtteranceBatch([wrong]))


def test_compute_before_load_is_refused():
    with pytest.raises(RuntimeError, match="load"):
        StubASRMetric().compute(UtteranceBatch([utterance("a")]))


@pytest.mark.parametrize("metric_type", ["asr_whisper", "asr_wav2vec2"])
def test_registered_backends_declare_asr_level_normalisation(metric_type):
    """-20 dBFS, not the -27 the phonetics metrics use — see the asr docstring."""
    metric = build_metric({"type": metric_type})
    assert metric.requirements.level_norm_dbfs == pytest.approx(-20.0)
    assert metric.requirements.sample_rate == 16_000
    assert metric.requirements.mono is True
    # The transcript is not a computation input; a metric declaring needs_text
    # would have every untranscribed row dropped from under it by select_rows.
    assert metric.requirements.needs_text is False


def test_whisper_guard_is_thirty_seconds():
    """Architectural: the feature extractor pads or truncates to 30 s, silently."""
    assert build_metric({"type": "asr_whisper"}).max_duration_s == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# Word errors
# ---------------------------------------------------------------------------


def test_word_errors_counts_each_edit_type():
    counts = word_errors("the quick brown fox", "the quick brown fox")
    assert counts["n_err"] == 0 and counts["wer"] == 0.0

    substitution = word_errors("the quick brown fox", "the slow brown fox")
    assert (substitution["n_sub"], substitution["n_del"], substitution["n_ins"]) == (1, 0, 0)

    deletion = word_errors("the quick brown fox", "the brown fox")
    assert (deletion["n_sub"], deletion["n_del"], deletion["n_ins"]) == (0, 1, 0)

    insertion = word_errors("the quick brown fox", "the very quick brown fox")
    assert (insertion["n_sub"], insertion["n_del"], insertion["n_ins"]) == (0, 0, 1)
    assert insertion["n_ref"] == 4
    assert insertion["wer"] == pytest.approx(0.25)


def test_word_errors_handles_an_empty_hypothesis():
    counts = word_errors("the quick brown fox", "")
    assert counts["n_del"] == 4 and counts["n_err"] == 4 and counts["wer"] == 1.0


def test_word_errors_on_an_empty_reference_is_undefined_but_still_counts():
    counts = word_errors("", "invented words here")
    assert counts["n_ins"] == 3 and counts["n_err"] == 3
    assert counts["n_ref"] == 0 and np.isnan(counts["wer"])


def test_pooled_wer_is_not_the_mean_of_per_utterance_rates():
    """The reason pooled_wer exists.

    One two-word segment with a single error, one thirty-word segment with none.
    Averaging the rates says 25%; the corpus made one mistake in thirty-two
    words, which is 3.1%.  On VAD segments this gap is the norm, not a corner
    case.
    """
    errors = pd.DataFrame([
        {"n_sub": 1, "n_del": 0, "n_ins": 0, "n_err": 1, "n_ref": 2, "wer": 0.5},
        {"n_sub": 0, "n_del": 0, "n_ins": 0, "n_err": 0, "n_ref": 30, "wer": 0.0},
    ])
    pooled = pooled_wer(errors)
    assert len(pooled) == 1
    assert pooled["wer"].iloc[0] == pytest.approx(1 / 32)
    assert pooled["n_utterances"].iloc[0] == 2
    assert errors["wer"].mean() == pytest.approx(0.25)


def test_pooled_wer_groups():
    errors = pd.DataFrame([
        {"condition": "real", "n_sub": 1, "n_del": 0, "n_ins": 0, "n_err": 1,
         "n_ref": 10, "wer": 0.1},
        {"condition": "converted", "n_sub": 4, "n_del": 0, "n_ins": 0, "n_err": 4,
         "n_ref": 10, "wer": 0.4},
    ])
    pooled = pooled_wer(errors, by="condition").set_index("condition")
    assert pooled.loc["real", "wer"] == pytest.approx(0.1)
    assert pooled.loc["converted", "wer"] == pytest.approx(0.4)
    # The headline number: how much worse the recogniser did on converted audio.
    assert pooled.loc["converted", "wer"] - pooled.loc["real", "wer"] == pytest.approx(0.3)


def test_pooled_wer_rejects_a_table_that_is_not_an_error_table():
    with pytest.raises(KeyError, match="n_err"):
        pooled_wer(pd.DataFrame({"wer": [0.1, 0.2]}))


# ---------------------------------------------------------------------------
# The two reference modes
# ---------------------------------------------------------------------------


@pytest.fixture
def results():
    return pd.DataFrame([
        {"utt_id": "u1", "group_id": "g1", "condition": "real", "text": "the quick fox",
         "asr.hypothesis": "the quick fox", "asr.status": "ok", "subject_id": "s1"},
        {"utt_id": "u2", "group_id": "g1", "condition": "converted",
         "text": "the quick fox", "asr.hypothesis": "the quiet fox",
         "asr.status": "ok", "subject_id": "s1"},
        {"utt_id": "u3", "group_id": "g2", "condition": "real", "text": "hello world",
         "asr.hypothesis": "hello world", "asr.status": "ok", "subject_id": "s2"},
        {"utt_id": "u4", "group_id": "g2", "condition": "converted",
         "text": "hello world", "asr.hypothesis": "hello word",
         "asr.status": "ok", "subject_id": "s2"},
    ])


def test_ground_truth_mode(results):
    errors = transcription_errors(
        results, "asr.hypothesis", reference_column="text", normalizer="basic",
    ).set_index("utt_id")
    assert errors.loc["u1", "n_err"] == 0
    assert errors.loc["u2", "n_sub"] == 1
    assert errors.loc["u4", "n_sub"] == 1

    pooled = pooled_wer(errors.reset_index(), by="condition").set_index("condition")
    assert pooled.loc["real", "wer"] == pytest.approx(0.0)
    assert pooled.loc["converted", "wer"] == pytest.approx(2 / 5)


def test_pseudo_reference_mode_scores_against_the_originals_own_decode(results):
    errors = transcription_errors(
        results, "asr.hypothesis", reference_condition="real", normalizer="basic",
    ).set_index("utt_id")
    # The reference condition is scored against itself, so it is 0 by
    # construction and carries no information.
    assert errors.loc["u1", "n_err"] == 0
    assert errors.loc["u3", "n_err"] == 0
    assert errors.loc["u2", "n_sub"] == 1
    assert errors.loc["u4", "n_sub"] == 1


def test_pseudo_reference_reads_zero_when_the_recogniser_errs_identically():
    """What the pseudo-reference mode does and does not measure.

    Both conditions were misrecognised the same way.  Against ground truth that
    is a 50% error rate on both; against the original's own hypothesis it is
    zero, because this mode measures divergence from that decode and not
    intelligibility.
    """
    frame = pd.DataFrame([
        {"utt_id": "u1", "group_id": "g", "condition": "real", "text": "cat sat",
         "asr.hypothesis": "cat sad", "asr.status": "ok"},
        {"utt_id": "u2", "group_id": "g", "condition": "converted", "text": "cat sat",
         "asr.hypothesis": "cat sad", "asr.status": "ok"},
    ])
    against_truth = transcription_errors(
        frame, "asr.hypothesis", reference_column="text", normalizer="basic",
    )
    against_original = transcription_errors(
        frame, "asr.hypothesis", reference_condition="real", normalizer="basic",
    )
    assert pooled_wer(against_truth)["wer"].iloc[0] == pytest.approx(0.5)
    assert pooled_wer(against_original)["wer"].iloc[0] == pytest.approx(0.0)


def test_flagged_rows_are_excluded_and_reported(results, capsys):
    results.loc[results["utt_id"] == "u2", "asr.status"] = STATUS_TOO_LONG
    errors = transcription_errors(
        results, "asr.hypothesis", reference_column="text", normalizer="basic",
    )
    assert "u2" not in set(errors["utt_id"])
    assert "too_long" in capsys.readouterr().out


def test_rows_without_a_reference_are_dropped_and_reported(results, capsys):
    results.loc[results["utt_id"] == "u2", "text"] = ""
    errors = transcription_errors(
        results, "asr.hypothesis", reference_column="text", normalizer="basic",
    )
    assert "u2" not in set(errors["utt_id"])
    assert "dropped" in capsys.readouterr().out


def test_a_group_missing_its_reference_condition_is_dropped(results, capsys):
    orphan = results[results["utt_id"] != "u3"]
    errors = transcription_errors(
        orphan, "asr.hypothesis", reference_condition="real", normalizer="basic",
    )
    assert set(errors["group_id"]) == {"g1"}
    assert "dropped" in capsys.readouterr().out


def test_a_duplicated_reference_condition_is_refused(results):
    doubled = pd.concat([results, results.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="more than once"):
        transcription_errors(
            doubled, "asr.hypothesis", reference_condition="real", normalizer="basic",
        )


def test_exactly_one_reference_must_be_named(results):
    with pytest.raises(ValueError, match="exactly one"):
        transcription_errors(results, "asr.hypothesis")
    with pytest.raises(ValueError, match="exactly one"):
        transcription_errors(
            results, "asr.hypothesis", reference_column="text",
            reference_condition="real",
        )


def test_carry_columns_reach_the_error_table(results):
    errors = transcription_errors(
        results, "asr.hypothesis", reference_column="text", normalizer="basic",
        carry=["subject_id"],
    )
    assert set(errors["subject_id"]) == {"s1", "s2"}


def test_status_column_is_found_from_the_hypothesis_prefix(results):
    results.loc[:, "asr.status"] = STATUS_TOO_LONG
    errors = transcription_errors(
        results, "asr.hypothesis", reference_column="text", normalizer="basic",
    )
    assert errors.empty


# ---------------------------------------------------------------------------
# Normalisers
# ---------------------------------------------------------------------------


def test_basic_normalizer_strips_case_and_punctuation():
    assert basic_normalize("Hello, World!  It's  here.") == "hello world its here"


def test_basic_normalizer_does_not_expand_numbers():
    """The documented difference from the Whisper normaliser."""
    assert basic_normalize("20 colours") != basic_normalize("twenty colours")


# ---------------------------------------------------------------------------
# Real backends — never run by accident
# ---------------------------------------------------------------------------

RUN_MODELS = os.environ.get("SPEECH_EVAL_RUN_ASR_MODELS") == "1"
PROBE_WAV = os.environ.get("SPEECH_EVAL_ASR_PROBE_WAV")

pytestmark_models = pytest.mark.skipif(
    not RUN_MODELS,
    reason="set SPEECH_EVAL_RUN_ASR_MODELS=1 (downloads several GB of checkpoints)",
)


@pytestmark_models
def test_whisper_normalizer_matches_its_documented_behaviour():
    from speech_eval.compare import build_normalizer

    normalize = build_normalizer("whisper_en")
    # Fillers are stripped, numbers become digits, contractions expand.
    assert "uh" not in normalize("he said uh twenty three").split()
    assert normalize("twenty three") == "23"
    assert normalize("won't") == "will not"


@pytestmark_models
@pytest.mark.skipif(not PROBE_WAV, reason="set SPEECH_EVAL_ASR_PROBE_WAV to real speech")
@pytest.mark.parametrize("metric_type", ["asr_whisper", "asr_wav2vec2"])
def test_hypothesis_is_stable_across_input_level(metric_type):
    """The known-answer probe behind the -20 dBFS choice.

    Whisper's log-mel affine is absolute, so a gain of X dB shifts every input
    feature by X/40 units; wav2vec2's processor z-scores the waveform and should
    be exactly invariant.  This asserts the level actually chosen is on a flat
    part of that curve for real speech.  If it fails, the level is load-bearing
    and the failing direction says which way to move it.

    Needs real speech: a synthetic tone has no transcript to be stable.
    """
    import soundfile as sf

    from audio_utils.data.transforms import rms_normalize

    audio, sample_rate = sf.read(PROBE_WAV, dtype="float32", always_2d=True)
    wav = torch.from_numpy(audio.T).mean(dim=0, keepdim=True)
    assert sample_rate == SAMPLE_RATE, f"probe wav must be {SAMPLE_RATE} Hz"

    metric = build_metric({"type": metric_type})
    metric.load(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    hypotheses = {}
    for level in (-27.0, -20.0, -14.0):
        batch = UtteranceBatch([Utterance(
            utt_id=str(level),
            wav=rms_normalize(wav, level),
            sample_rate=SAMPLE_RATE,
        )])
        hypotheses[level] = metric.compute(batch)[0]["hypothesis"]

    assert len(set(hypotheses.values())) == 1, (
        f"{metric_type} is level-sensitive over -27..-14 dBFS: {hypotheses}"
    )
