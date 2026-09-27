"""Speaker embedding metrics and the identity comparisons they feed.

The backend-independent contract is tested against a stub embedder rather than a
real one: the point of these tests is that ``speaker_ecapa.embedding`` and
``speaker_wavlm.embedding`` mean the same thing — same guards, same flag values,
same normalisation, same batch ordering — and that is settled by the base class,
not by either model.  Downloading two checkpoints would test speechbrain and
transformers, not this repo.

The comparison layer is tested on synthetic vectors whose answers are known in
closed form: orthogonal directions score 0, a vector scores 1 against itself,
perfectly separated trials give an EER of 0 and perfectly overlapping ones 0.5.

The real backends are exercised at the bottom, gated behind an environment
variable so they never fire by accident.  That is also where the level-invariance
claim in ``speech_eval/metrics/speaker.py`` is *probed* rather than reasoned
about.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
import torch

from speech_eval.compare import (
    accept_rate,
    array_features,
    calibrate,
    calibrated_similarity,
    cosine,
    equal_error_rate,
    paired_similarity,
    similarity_to_centroid,
    speaker_centroids,
    verification_trials,
)
from speech_eval.core import Utterance, UtteranceBatch, build_metric
from speech_eval.metrics.speaker import (
    STATUS_EMPTY,
    STATUS_OK,
    STATUS_SHORT,
    SpeakerEmbeddingMetric,
)

SAMPLE_RATE = 16_000


# ---------------------------------------------------------------------------
# An embedder that embeds nothing
# ---------------------------------------------------------------------------


class StubSpeakerMetric(SpeakerEmbeddingMetric):
    """Returns a deterministic vector per input and records what it was given.

    Deliberately not registered: the registry is global, and a test type in it
    would leak into ``available_metrics()``.
    """

    metric_type = "stub_speaker"
    backend = "stub"
    frontend = "stub_frontend"
    default_model_id = "stub/model"

    DIM = 8

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls: list[list[int]] = []

    def _load_backend(self, device) -> None:  # noqa: ANN001
        self._model = object()

    def _embed(self, waveforms: list[np.ndarray]) -> np.ndarray:
        self.calls.append([len(w) for w in waveforms])
        # Length-dependent and unnormalised, so the base class's normalisation
        # and its `embedding_norm` are both observable.
        return np.stack([
            np.full(self.DIM, float(len(w)) / SAMPLE_RATE, dtype="float32")
            for w in waveforms
        ])


def make_batch(specs: list[tuple[str, float]]) -> UtteranceBatch:
    """``[(utt_id, duration_s)]`` → a batch of tones; 0 s means empty audio."""
    utterances = []
    for utt_id, duration_s in specs:
        n = int(duration_s * SAMPLE_RATE)
        t = np.arange(n) / SAMPLE_RATE
        wav = torch.from_numpy(
            (0.1 * np.sin(2 * np.pi * 200 * t)).astype("float32")
        ).unsqueeze(0)
        utterances.append(Utterance(utt_id=utt_id, wav=wav, sample_rate=SAMPLE_RATE))
    return UtteranceBatch(utterances=utterances)


@pytest.fixture
def metric() -> StubSpeakerMetric:
    m = StubSpeakerMetric(min_duration_s=2.0)
    m.load(torch.device("cpu"))
    return m


# ---------------------------------------------------------------------------
# The metric contract
# ---------------------------------------------------------------------------


def test_load_measures_the_embedding_dimension(metric):
    assert metric.embedding_dim == StubSpeakerMetric.DIM


def test_dimension_is_unknown_before_load():
    with pytest.raises(RuntimeError, match="load"):
        _ = StubSpeakerMetric().embedding_dim


def test_compute_before_load_raises():
    with pytest.raises(RuntimeError, match="load"):
        StubSpeakerMetric().compute(make_batch([("a", 3.0)]))


def test_one_result_per_utterance_in_batch_order(metric):
    results = metric.compute(make_batch([("a", 3.0), ("b", 4.0), ("c", 5.0)]))
    assert len(results) == 3
    # The stub's vector encodes duration, so order is checkable from the output.
    assert [r["duration_s"] for r in results] == pytest.approx([3.0, 4.0, 5.0])


def test_embeddings_are_stored_as_unit_vectors(metric):
    results = metric.compute(make_batch([("a", 3.0), ("b", 5.0)]))
    for result in results:
        assert np.linalg.norm(result["embedding"]) == pytest.approx(1.0)


def test_embedding_norm_keeps_the_pre_normalisation_length(metric):
    (result,) = metric.compute(make_batch([("a", 4.0)]))
    # The stub emits a constant vector of 4.0 in each of 8 dimensions.
    assert result["embedding_norm"] == pytest.approx(4.0 * np.sqrt(8))


def test_short_utterances_are_flagged_but_still_embedded(metric):
    (result,) = metric.compute(make_batch([("a", 0.5)]))
    assert result["status"] == STATUS_SHORT
    assert np.isfinite(result["embedding"]).all()
    assert np.linalg.norm(result["embedding"]) == pytest.approx(1.0)


def test_long_enough_utterances_are_ok(metric):
    (result,) = metric.compute(make_batch([("a", 3.0)]))
    assert result["status"] == STATUS_OK


def test_min_duration_none_disables_the_short_flag():
    m = StubSpeakerMetric(min_duration_s=None)
    m.load(torch.device("cpu"))
    (result,) = m.compute(make_batch([("a", 0.1)]))
    assert result["status"] == STATUS_OK


def test_empty_audio_is_flagged_and_yields_a_nan_embedding(metric):
    (result,) = metric.compute(make_batch([("a", 0.0)]))
    assert result["status"] == STATUS_EMPTY
    assert np.isnan(result["embedding"]).all()
    assert np.isnan(result["embedding_norm"])


def test_empty_audio_never_reaches_the_backend(metric):
    metric.calls.clear()
    metric.compute(make_batch([("a", 0.0), ("b", 3.0)]))
    assert metric.calls == [[3 * SAMPLE_RATE]]


def test_a_nan_embedding_propagates_to_nan_through_cosine(metric):
    (empty,), (real,) = (
        metric.compute(make_batch([("a", 0.0)])),
        metric.compute(make_batch([("b", 3.0)])),
    )
    assert np.isnan(cosine(empty["embedding"], real["embedding"]))


def test_provenance_is_on_every_row(metric):
    (result,) = metric.compute(make_batch([("a", 3.0)]))
    assert result["backend"] == "stub"
    assert result["frontend"] == "stub_frontend"
    assert result["model_id"] == "stub/model"
    assert result["dim"] == StubSpeakerMetric.DIM


def test_status_is_never_null(metric):
    results = metric.compute(make_batch([("a", 0.0), ("b", 0.5), ("c", 3.0)]))
    assert [r["status"] for r in results] == [STATUS_EMPTY, STATUS_SHORT, STATUS_OK]


def test_wrong_sample_rate_raises(metric):
    batch = make_batch([("a", 3.0)])
    batch.utterances[0].sample_rate = 8_000
    with pytest.raises(ValueError, match="8000 Hz"):
        metric.compute(batch)


def test_inner_batch_size_chunks_the_backend_calls():
    m = StubSpeakerMetric(min_duration_s=None, batch_size=2)
    m.load(torch.device("cpu"))
    m.calls.clear()
    m.compute(make_batch([("a", 1.0), ("b", 1.0), ("c", 1.0), ("d", 1.0), ("e", 1.0)]))
    assert [len(c) for c in m.calls] == [2, 2, 1]


def test_a_backend_returning_the_wrong_count_raises():
    class Broken(StubSpeakerMetric):
        def _embed(self, waveforms):
            return super()._embed(waveforms)[:1]

    m = Broken(min_duration_s=None)
    m.load(torch.device("cpu"))
    with pytest.raises(RuntimeError, match="one row per input"):
        m.compute(make_batch([("a", 1.0), ("b", 1.0)]))


def test_both_backends_are_registered():
    for metric_type in ("speaker_ecapa", "speaker_wavlm"):
        built = build_metric({"type": metric_type})
        assert built.name == metric_type
        assert built.requirements.sample_rate == 16_000
        assert built.requirements.level_norm_dbfs == -20.0
        assert built.requirements.mono is True


def test_the_two_backends_declare_different_frontends():
    ecapa = build_metric({"type": "speaker_ecapa"})
    wavlm = build_metric({"type": "speaker_wavlm"})
    assert ecapa.frontend != wavlm.frontend
    assert ecapa.backend != wavlm.backend


def test_backends_share_the_asr_preparation_pass():
    """Speaker and ASR must group together, or the corpus is decoded twice."""
    from speech_eval.runner import group_by_requirements

    metrics = [
        build_metric({"type": "speaker_ecapa"}),
        build_metric({"type": "speaker_wavlm"}),
        build_metric({"type": "asr_whisper"}),
    ]
    groups = group_by_requirements(metrics)
    assert len(groups) == 1, "speaker and ASR should share one requirements group"


# ---------------------------------------------------------------------------
# Cosine
# ---------------------------------------------------------------------------


def test_cosine_of_a_vector_with_itself_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine(v, v) == pytest.approx(1.0)


def test_cosine_of_orthogonal_vectors_is_zero():
    assert cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_cosine_of_opposite_vectors_is_minus_one():
    assert cosine(np.array([1.0, 0.0]), np.array([-2.0, 0.0])) == pytest.approx(-1.0)


def test_cosine_is_scale_invariant():
    a, b = np.array([1.0, 2.0]), np.array([2.0, 1.0])
    assert cosine(a, b) == pytest.approx(cosine(a, 100 * b))


def test_cosine_of_a_zero_vector_is_nan():
    assert np.isnan(cosine(np.zeros(3), np.ones(3)))


def test_cosine_broadcasts_row_wise():
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert cosine(a, b) == pytest.approx([1.0, 0.0])


# ---------------------------------------------------------------------------
# A small synthetic corpus of embeddings
# ---------------------------------------------------------------------------

EMBEDDING_COL = "speaker_ecapa.embedding"


def speaker_direction(index: int, dim: int = 16) -> np.ndarray:
    """A distinct unit direction per speaker."""
    v = np.zeros(dim)
    v[index] = 1.0
    return v


def build_corpus(
    n_speakers: int = 4,
    n_groups: int = 3,
    jitter: float = 0.05,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Results table and artifacts for a corpus with a known identity structure.

    Each speaker sits on its own axis; conditions perturb it by a little
    (``real_loud``) or a lot (``converted``), so the ordering of the similarities
    is known in advance and can be asserted on.
    """
    rng = np.random.default_rng(seed)
    rows, artifacts = [], {}
    perturbation = {"real_normal": 0.0, "real_loud": 1.0, "converted": 3.0}
    for speaker in range(n_speakers):
        base = speaker_direction(speaker)
        for group in range(n_groups):
            group_id = f"s{speaker}_g{group}"
            for condition, scale in perturbation.items():
                utt_id = f"{group_id}_{condition}"
                noise = rng.standard_normal(base.size)
                vector = base + jitter * (1.0 + scale) * noise
                vector /= np.linalg.norm(vector)
                artifacts[f"{utt_id}|{EMBEDDING_COL}"] = vector.astype("float32")
                rows.append({
                    "utt_id": utt_id,
                    "group_id": group_id,
                    "condition": condition,
                    "subject_id": f"spk{speaker}",
                    "speaker_ecapa.status": "ok",
                    "speaker_ecapa.duration_s": 4.0,
                    "speaker_ecapa.dim": base.size,
                })
    return pd.DataFrame(rows), artifacts


@pytest.fixture
def corpus_results():
    return build_corpus()


# ---------------------------------------------------------------------------
# array_features
# ---------------------------------------------------------------------------


def test_array_features_strips_the_utterance_key(corpus_results):
    _, artifacts = corpus_results
    found = array_features(artifacts, EMBEDDING_COL)
    assert "s0_g0_real_normal" in found
    assert found["s0_g0_real_normal"].shape == (16,)


def test_array_features_restricts_to_the_requested_utterances(corpus_results):
    _, artifacts = corpus_results
    found = array_features(artifacts, EMBEDDING_COL, ["s0_g0_real_normal"])
    assert list(found) == ["s0_g0_real_normal"]


def test_array_features_raises_on_an_unknown_column(corpus_results):
    _, artifacts = corpus_results
    with pytest.raises(KeyError, match="No artifact holds column"):
        array_features(artifacts, "speaker_nonexistent.embedding")


# ---------------------------------------------------------------------------
# paired_similarity
# ---------------------------------------------------------------------------


def test_paired_similarity_scores_every_other_condition(corpus_results):
    results, artifacts = corpus_results
    paired = paired_similarity(results, artifacts, EMBEDDING_COL, "real_normal")
    assert set(paired["condition"]) == {"real_loud", "converted"}
    assert len(paired) == 4 * 3 * 2


def test_paired_similarity_reference_rows_are_excluded_by_default(corpus_results):
    results, artifacts = corpus_results
    paired = paired_similarity(results, artifacts, EMBEDDING_COL, "real_normal")
    assert "real_normal" not in set(paired["condition"])


def test_paired_similarity_can_keep_the_reference_and_it_scores_one(corpus_results):
    results, artifacts = corpus_results
    paired = paired_similarity(
        results, artifacts, EMBEDDING_COL, "real_normal", include_reference=True
    )
    reference = paired[paired["condition"] == "real_normal"]
    assert reference["cosine"].to_numpy() == pytest.approx(1.0)


def test_paired_similarity_pairs_within_the_group(corpus_results):
    results, artifacts = corpus_results
    paired = paired_similarity(results, artifacts, EMBEDDING_COL, "real_normal")
    for _, row in paired.iterrows():
        assert row["reference_utt_id"] == f"{row['group_id']}_real_normal"


def test_a_bigger_perturbation_gives_a_lower_similarity(corpus_results):
    results, artifacts = corpus_results
    paired = paired_similarity(results, artifacts, EMBEDDING_COL, "real_normal")
    means = paired.groupby("condition")["cosine"].mean()
    assert means["real_loud"] > means["converted"]


def test_paired_similarity_carries_requested_columns(corpus_results):
    results, artifacts = corpus_results
    paired = paired_similarity(
        results, artifacts, EMBEDDING_COL, "real_normal", carry=["subject_id"]
    )
    assert "subject_id" in paired.columns


def test_paired_similarity_drops_groups_without_a_reference(corpus_results, capsys):
    results, artifacts = corpus_results
    results = results[
        ~((results["group_id"] == "s0_g0") & (results["condition"] == "real_normal"))
    ]
    paired = paired_similarity(results, artifacts, EMBEDDING_COL, "real_normal")
    assert "s0_g0" not in set(paired["group_id"])
    assert "dropped" in capsys.readouterr().out


def test_paired_similarity_excludes_flagged_rows(corpus_results, capsys):
    results, artifacts = corpus_results
    results.loc[results["utt_id"] == "s0_g0_converted", "speaker_ecapa.status"] = \
        "empty_audio"
    paired = paired_similarity(results, artifacts, EMBEDDING_COL, "real_normal")
    assert "s0_g0_converted" not in set(paired["utt_id"])
    assert "empty_audio" in capsys.readouterr().out


def test_short_rows_are_kept_by_default_and_droppable_on_request(corpus_results):
    results, artifacts = corpus_results
    results.loc[results["utt_id"] == "s0_g0_converted", "speaker_ecapa.status"] = "short"

    lenient = paired_similarity(results, artifacts, EMBEDDING_COL, "real_normal")
    assert "s0_g0_converted" in set(lenient["utt_id"])

    strict = paired_similarity(
        results, artifacts, EMBEDDING_COL, "real_normal", ok_statuses=("ok",)
    )
    assert "s0_g0_converted" not in set(strict["utt_id"])


def test_min_duration_filter_uses_the_metrics_own_duration_column(corpus_results):
    results, artifacts = corpus_results
    results.loc[results["utt_id"] == "s0_g0_converted", "speaker_ecapa.duration_s"] = 0.5
    paired = paired_similarity(
        results, artifacts, EMBEDDING_COL, "real_normal", min_duration_s=1.0
    )
    assert "s0_g0_converted" not in set(paired["utt_id"])


def test_rows_without_an_embedding_are_dropped(corpus_results, capsys):
    results, artifacts = corpus_results
    del artifacts[f"s0_g0_converted|{EMBEDDING_COL}"]
    paired = paired_similarity(results, artifacts, EMBEDDING_COL, "real_normal")
    assert "s0_g0_converted" not in set(paired["utt_id"])
    assert "artifacts.npz" in capsys.readouterr().out


def test_a_duplicated_reference_condition_raises(corpus_results):
    results, artifacts = corpus_results
    duplicate = results[results["utt_id"] == "s0_g0_real_normal"].copy()
    duplicate["utt_id"] = "s0_g0_real_normal_again"
    artifacts[f"s0_g0_real_normal_again|{EMBEDDING_COL}"] = \
        artifacts[f"s0_g0_real_normal|{EMBEDDING_COL}"]
    results = pd.concat([results, duplicate], ignore_index=True)
    with pytest.raises(ValueError, match="more than once"):
        paired_similarity(results, artifacts, EMBEDDING_COL, "real_normal")


# ---------------------------------------------------------------------------
# Trials, EER and calibration
# ---------------------------------------------------------------------------


def test_verification_trials_labels_targets_by_speaker(corpus_results):
    results, artifacts = corpus_results
    trials = verification_trials(
        results, artifacts, EMBEDDING_COL, conditions=["real_normal"]
    )
    same = trials["speaker_a"] == trials["speaker_b"]
    assert (trials["label"] == same).all()


def test_verification_trials_exclude_same_group_targets_by_default(corpus_results):
    results, artifacts = corpus_results
    trials = verification_trials(results, artifacts, EMBEDDING_COL)
    targets = trials[trials["label"]]
    assert (targets["group_a"] != targets["group_b"]).all()


def test_same_group_targets_can_be_re_enabled(corpus_results):
    results, artifacts = corpus_results
    with_same = verification_trials(
        results, artifacts, EMBEDDING_COL, exclude_same_group=False
    )
    targets = with_same[with_same["label"]]
    assert (targets["group_a"] == targets["group_b"]).any()


def test_conditions_filter_restricts_the_pool(corpus_results):
    results, artifacts = corpus_results
    trials = verification_trials(
        results, artifacts, EMBEDDING_COL, conditions=["real_normal"]
    )
    assert set(trials["condition_a"]) == {"real_normal"}
    assert set(trials["condition_b"]) == {"real_normal"}


def test_targets_score_higher_than_nontargets(corpus_results):
    results, artifacts = corpus_results
    trials = verification_trials(
        results, artifacts, EMBEDDING_COL, conditions=["real_normal"]
    )
    assert trials[trials["label"]]["cosine"].mean() > \
        trials[~trials["label"]]["cosine"].mean()


def test_trials_are_reproducible_under_a_seed(corpus_results):
    results, artifacts = corpus_results
    a = verification_trials(results, artifacts, EMBEDDING_COL, seed=7)
    b = verification_trials(results, artifacts, EMBEDDING_COL, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_trial_caps_are_respected(corpus_results):
    results, artifacts = corpus_results
    trials = verification_trials(
        results, artifacts, EMBEDDING_COL, max_target=5, max_nontarget=7
    )
    assert int(trials["label"].sum()) <= 5
    assert int((~trials["label"]).sum()) <= 7


def test_a_missing_speaker_column_raises(corpus_results):
    results, artifacts = corpus_results
    with pytest.raises(KeyError, match="speaker column"):
        verification_trials(
            results.drop(columns=["subject_id"]), artifacts, EMBEDDING_COL
        )


def test_a_corpus_of_one_utterance_per_speaker_explains_itself(corpus_results):
    results, artifacts = corpus_results
    single = results[results["group_id"].str.endswith("_g0")]
    single = single[single["condition"] == "real_normal"]
    with pytest.raises(ValueError, match="No target trials"):
        verification_trials(single, artifacts, EMBEDDING_COL)


def test_eer_is_zero_when_the_classes_are_separable():
    scores = np.array([0.9, 0.8, 0.1, 0.2])
    labels = np.array([True, True, False, False])
    assert equal_error_rate(scores, labels)["eer"] == pytest.approx(0.0)


def test_eer_is_a_half_when_the_classes_are_identical():
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([True, True, False, False])
    assert equal_error_rate(scores, labels)["eer"] == pytest.approx(0.5)


def test_eer_threshold_separates_the_classes():
    scores = np.array([0.9, 0.8, 0.1, 0.2])
    labels = np.array([True, True, False, False])
    summary = equal_error_rate(scores, labels)
    assert 0.2 < summary["threshold"] <= 0.8


def test_eer_counts_the_trials():
    summary = equal_error_rate(
        np.array([0.9, 0.8, 0.1]), np.array([True, True, False])
    )
    assert summary["n_target"] == 2 and summary["n_nontarget"] == 1


def test_eer_needs_both_classes():
    with pytest.raises(ValueError, match="target and nontarget"):
        equal_error_rate(np.array([0.9, 0.8]), np.array([True, True]))


def test_eer_ignores_nan_scores():
    scores = np.array([0.9, np.nan, 0.1])
    labels = np.array([True, True, False])
    assert equal_error_rate(scores, labels)["n_target"] == 1


def test_calibrate_reports_both_distributions(corpus_results):
    results, artifacts = corpus_results
    trials = verification_trials(
        results, artifacts, EMBEDDING_COL, conditions=["real_normal"]
    )
    calibration = calibrate(trials)
    assert calibration["mu_target"] > calibration["mu_nontarget"]
    assert calibration["d_prime"] > 0
    assert calibration["n_speakers"] == 4


def test_calibrated_similarity_anchors_on_the_two_means():
    calibration = {"mu_target": 0.8, "mu_nontarget": 0.2}
    assert calibrated_similarity([0.8], calibration) == pytest.approx([1.0])
    assert calibrated_similarity([0.2], calibration) == pytest.approx([0.0])
    assert calibrated_similarity([0.5], calibration) == pytest.approx([0.5])


def test_calibrated_similarity_is_not_clipped():
    calibration = {"mu_target": 0.8, "mu_nontarget": 0.2}
    assert calibrated_similarity([0.9], calibration)[0] > 1.0
    assert calibrated_similarity([0.1], calibration)[0] < 0.0


def test_calibrated_similarity_refuses_a_degenerate_axis():
    with pytest.raises(ValueError, match="coincide"):
        calibrated_similarity([0.5], {"mu_target": 0.4, "mu_nontarget": 0.4})


def test_calibrated_similarity_needs_a_real_calibration():
    with pytest.raises(KeyError, match="mu_target"):
        calibrated_similarity([0.5], {"eer": 0.1})


def test_accept_rate_counts_scores_at_or_above_the_threshold():
    assert accept_rate([0.9, 0.5, 0.1], 0.5)["accept_rate"] == pytest.approx(2 / 3)


def test_accept_rate_ignores_nan():
    summary = accept_rate([0.9, np.nan, 0.1], 0.5)
    assert summary["n"] == 2 and summary["accept_rate"] == pytest.approx(0.5)


def test_accept_rate_of_nothing_is_nan():
    assert np.isnan(accept_rate([], 0.5)["accept_rate"])


def test_the_whole_calibrated_pipeline_ranks_the_conditions(corpus_results):
    """The end-to-end shape a report actually uses."""
    results, artifacts = corpus_results
    trials = verification_trials(
        results, artifacts, EMBEDDING_COL, conditions=["real_normal"]
    )
    calibration = calibrate(trials)
    paired = paired_similarity(results, artifacts, EMBEDDING_COL, "real_normal")
    paired["calibrated"] = calibrated_similarity(paired["cosine"], calibration)

    means = paired.groupby("condition")["calibrated"].mean()
    assert means["real_loud"] > means["converted"]
    # Both stay nearer "same speaker" than "different speaker" on this corpus.
    assert means["converted"] > 0.5

    loud = paired[paired["condition"] == "real_loud"]["cosine"]
    converted = paired[paired["condition"] == "converted"]["cosine"]
    assert accept_rate(loud, calibration["threshold"])["accept_rate"] >= \
        accept_rate(converted, calibration["threshold"])["accept_rate"]


# ---------------------------------------------------------------------------
# Centroid enrolment
# ---------------------------------------------------------------------------


def test_centroids_are_built_per_speaker(corpus_results):
    results, artifacts = corpus_results
    enrolment = speaker_centroids(
        results, artifacts, EMBEDDING_COL, conditions=["real_normal"]
    )
    assert len(enrolment) == 4
    assert set(enrolment.centroids) == {"spk0", "spk1", "spk2", "spk3"}


def test_centroids_are_unit_vectors(corpus_results):
    results, artifacts = corpus_results
    enrolment = speaker_centroids(
        results, artifacts, EMBEDDING_COL, conditions=["real_normal"]
    )
    for vector in enrolment.centroids.values():
        assert np.linalg.norm(vector) == pytest.approx(1.0)


def test_a_centroid_points_at_its_speakers_own_direction(corpus_results):
    results, artifacts = corpus_results
    enrolment = speaker_centroids(
        results, artifacts, EMBEDDING_COL, conditions=["real_normal"]
    )
    assert cosine(enrolment.centroids["spk1"], speaker_direction(1)) > 0.9


def test_speakers_below_min_utterances_are_skipped(corpus_results, capsys):
    results, artifacts = corpus_results
    # Leave spk3 with a single enrollable utterance while the others keep three.
    thin = results[
        (results["subject_id"] != "spk3") | (results["group_id"] == "s3_g0")
    ]
    enrolment = speaker_centroids(
        thin, artifacts, EMBEDDING_COL, conditions=["real_normal"], min_utterances=2
    )
    assert set(enrolment.centroids) == {"spk0", "spk1", "spk2"}
    assert "1 speaker(s) not enrolled" in capsys.readouterr().out


def test_enrolling_nobody_raises(corpus_results):
    results, artifacts = corpus_results
    with pytest.raises(ValueError, match="No speaker could be enrolled"):
        speaker_centroids(
            results, artifacts, EMBEDDING_COL, conditions=["real_normal"],
            min_utterances=99,
        )


def test_similarity_to_centroid_scores_every_row(corpus_results):
    results, artifacts = corpus_results
    enrolment = speaker_centroids(
        results, artifacts, EMBEDDING_COL, conditions=["real_normal"]
    )
    scored = similarity_to_centroid(results, artifacts, EMBEDDING_COL, enrolment)
    assert len(scored) == len(results)
    assert scored["cosine"].notna().all()


def test_exclude_self_lowers_the_score_of_an_enrolled_utterance(corpus_results):
    results, artifacts = corpus_results
    enrolment = speaker_centroids(
        results, artifacts, EMBEDDING_COL, conditions=["real_normal"]
    )
    with_self = similarity_to_centroid(
        results, artifacts, EMBEDDING_COL, enrolment, exclude_self=False
    ).set_index("utt_id")
    without = similarity_to_centroid(
        results, artifacts, EMBEDDING_COL, enrolment, exclude_self=True
    ).set_index("utt_id")

    enrolled = "s0_g0_real_normal"
    assert without.loc[enrolled, "cosine"] < with_self.loc[enrolled, "cosine"]
    # A row that never contributed is untouched by the option.
    assert without.loc["s0_g0_converted", "cosine"] == pytest.approx(
        with_self.loc["s0_g0_converted", "cosine"]
    )


def test_exclude_self_reports_the_enrolment_size(corpus_results):
    results, artifacts = corpus_results
    enrolment = speaker_centroids(
        results, artifacts, EMBEDDING_COL, conditions=["real_normal"]
    )
    scored = similarity_to_centroid(
        results, artifacts, EMBEDDING_COL, enrolment
    ).set_index("utt_id")
    assert scored.loc["s0_g0_real_normal", "n_enrolment"] == 2
    assert scored.loc["s0_g0_converted", "n_enrolment"] == 3


def test_centroid_route_ranks_the_conditions_like_the_paired_route(corpus_results):
    results, artifacts = corpus_results
    enrolment = speaker_centroids(
        results, artifacts, EMBEDDING_COL, conditions=["real_normal"]
    )
    scored = similarity_to_centroid(results, artifacts, EMBEDDING_COL, enrolment)
    means = scored.groupby("condition")["cosine"].mean()
    assert means["real_loud"] > means["converted"]


def test_unenrolled_speakers_are_dropped(corpus_results, capsys):
    results, artifacts = corpus_results
    enrolment = speaker_centroids(
        results[results["subject_id"] != "spk3"], artifacts, EMBEDDING_COL,
        conditions=["real_normal"],
    )
    scored = similarity_to_centroid(results, artifacts, EMBEDDING_COL, enrolment)
    assert "spk3" not in set(scored["subject_id"])
    assert "no enrolment" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Real backends — opt-in
# ---------------------------------------------------------------------------

REAL_BACKENDS = pytest.mark.skipif(
    not os.environ.get("SPEECH_EVAL_TEST_SPEAKER_BACKENDS"),
    reason="set SPEECH_EVAL_TEST_SPEAKER_BACKENDS=1 to download and run the models",
)


def speech_like(duration_s: float, seed: int = 0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """A harmonic stack with a moving envelope — enough structure for an embedder."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * sr)) / sr
    f0 = 120 + 20 * np.sin(2 * np.pi * 3 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    signal = sum(np.sin(k * phase) / k for k in range(1, 12))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
    signal = signal * envelope + 0.01 * rng.standard_normal(t.size)
    return (0.1 * signal / np.abs(signal).max()).astype("float32")


@REAL_BACKENDS
@pytest.mark.parametrize("metric_type", ["speaker_ecapa", "speaker_wavlm"])
def test_real_backend_embeds_and_reports_its_dimension(metric_type):
    m = build_metric({"type": metric_type})
    m.load(torch.device("cpu"))
    batch = UtteranceBatch(utterances=[
        Utterance("a", torch.from_numpy(speech_like(3.0)).unsqueeze(0), SAMPLE_RATE),
    ])
    (result,) = m.compute(batch)
    assert result["status"] == STATUS_OK
    assert result["dim"] == m.embedding_dim > 0
    assert np.linalg.norm(result["embedding"]) == pytest.approx(1.0, abs=1e-5)


@REAL_BACKENDS
@pytest.mark.parametrize("metric_type", ["speaker_ecapa", "speaker_wavlm"])
def test_real_backend_ignores_batch_padding(metric_type):
    """A signal must embed identically alone and beside a much longer neighbour.

    This is the padding trap: speechbrain's ``wav_lens`` are relative and default
    to "no padding at all", and a mask omitted for WavLM does the same damage.
    Either mistake shows up here and nowhere else.
    """
    m = build_metric({"type": metric_type})
    m.load(torch.device("cpu"))
    short, long = speech_like(2.5, seed=1), speech_like(6.0, seed=2)

    alone = m._embed([short])[0]
    padded = m._embed([short, long])[0]
    assert cosine(alone, padded) == pytest.approx(1.0, abs=1e-3)


@REAL_BACKENDS
def test_ecapa_is_level_invariant():
    """Probes the mean-var-norm argument in the module docstring, rather than
    trusting it: a gain is a constant offset in the log domain and is subtracted
    away, so the embedding should not move at all."""
    m = build_metric({"type": "speaker_ecapa"})
    m.load(torch.device("cpu"))
    signal = speech_like(3.0)
    quiet, loud = m._embed([signal * 0.1])[0], m._embed([signal * 1.0])[0]
    assert cosine(quiet, loud) == pytest.approx(1.0, abs=1e-3)


@REAL_BACKENDS
def test_wavlm_level_sensitivity_is_measured_not_assumed(capsys):
    """The counterpart: ``do_normalize: false``, so this one may well move.

    Deliberately not an assertion on a bound — the point is to print the number,
    so the -20 dBFS normalisation can be justified from a measurement rather than
    from a config file.
    """
    m = build_metric({"type": "speaker_wavlm"})
    m.load(torch.device("cpu"))
    signal = speech_like(3.0)
    quiet, loud = m._embed([signal * 0.1])[0], m._embed([signal * 1.0])[0]
    with capsys.disabled():
        print(f"\nwavlm cosine across a 20 dB gain change: {cosine(quiet, loud):.4f}")
