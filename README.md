# speech-eval

Objective evaluation metrics for speech generation and conversion systems.

Built for the vocal-intensity-conversion work, but deliberately independent of
it: **`speech_eval` never imports `vic`**. A project produces audio and a
manifest describing it; this repo turns waveforms into numbers. That constraint
is what makes the suite reusable for the affective-TTS evaluation as well.

## Design

**Metrics are per-utterance feature extractors, never comparisons.**

An ASR metric emits a hypothesis *string*. A speaker metric emits an
*embedding*. An F0 metric emits a contour and its summaries. WER, cosine
similarity and recovery rate are then computed downstream in `compare.py`, from
features already on disk.

Three reasons this is worth the indirection:

* A file serving as reference for several pairs is embedded **once**. In the
  pairwise formulation, five conditions per utterance means running the model on
  the same reference audio four extra times.
* Changing the comparison design — different pairing, an added baseline, another
  text normaliser — costs no GPU time, because the expensive stage is already
  computed.
* `speech_eval` never learns what a "condition" *is*, which is what keeps it
  experiment-agnostic.

**Requirements drive preprocessing.** Each metric declares a `Requirements`
describing the audio it needs — sample rate, mono, level normalisation, whether
it needs a transcript — instead of preparing that audio itself. The runner
groups metrics by requirements, so ten metrics wanting 16 kHz mono cause one
resample and not ten. It also puts level normalisation somewhere visible: ASR,
MOS and speaker embedders are all level-sensitive, and a suite comparing signals
that *differ in level by construction* will measure gain instead of degradation
unless normalisation is declared per metric.

`Requirements.meta_columns` names manifest columns a metric needs in order to
**compute** — `("sex",)` for a pitch tracker whose analysis range is
sex-conditional. It is deliberately not the same list as `carry_columns`, which
says what the results table *shows*, and it is ignored when the runner groups
metrics: a metadata column is not a property of the waveform, so two metrics
differing only in it still share one preparation pass.

## Manifest

One row per audio item. The path and segment columns are the same ones the rest
of these repos already write, so an existing segment index becomes a manifest by
adding three columns rather than by being rewritten.

| column | | meaning |
|---|---|---|
| `utt_id` | required | unique row key |
| `signal_path` | required | audio path, relative to a dataset root |
| `group_id` | required | ties together the rows describing the **same underlying utterance** across conditions |
| `condition` | required | free-form label; `speech_eval` never interprets it |
| `text` | optional | reference transcript |
| `corpus` | optional | selects the dataset root for the row |
| `channel`, `start_s`, `end_s` | optional | as in `audio_utils.AudioDataset` |
| anything else | optional | carried through to the results via `carry_columns` |

`group_id` is what makes pairing a `groupby` rather than a filename join.
Whether a group holds two rows or five is the experiment's decision, taken when
the manifest is written and again when the results are read — never here.

## Running

```bash
uv run --extra cpu scripts/run_eval.py \
    --config     configs/run_eval/local.yaml \
    --output-dir outputs/eval/smoke
```

Outputs, in `--output-dir`:

* `metrics.csv` — one row per utterance, one column per emitted scalar, named
  `<metric name>.<key>`. Long form, so every aggregation is a `groupby`.
* `artifacts.npz` — arrays too large for a cell (embeddings, contours), keyed
  `<utt_id>|<metric name>.<key>`.

Re-running resumes: a metric is skipped for utterances that already carry a
non-null value under its prefix.

The `name` field on a metric config entry overrides its column prefix, so the
same metric type can run twice under different names. Reading two independent
instruments and reporting their disagreement is the intended use, not an edge
case.

## Comparisons

`speech_eval.compare` operates on what the runner already wrote — `metrics.csv`,
and `artifacts.npz` for array-valued features. No audio, no models, no GPU. It is
a package (`core.py`, `wer.py`, `speaker.py`) whose `__init__` re-exports
everything, so `from speech_eval.compare import pooled_wer` works regardless of
which submodule a function lives in.

* **`recovery_rate(source, converted, target)`** — of the distance real speech
  travels between two conditions, what fraction did the model travel?
  `1` reaches the target, `0` changes nothing, `< 0` moves the wrong way.
  Requires content matched across conditions.
* **`slope_vs_target(...)`** — per-speaker least-squares slope of a measure
  against the target level. Fit it on real speech and on converted speech and
  compare: this is the unpaired form of the same question, and it is the
  fallback whenever pairing is impossible.
* **`cluster_bootstrap_ci(...)`** — confidence intervals resampling *speakers*,
  not utterances. A corpus of 50 speakers carries roughly 50 independent units
  however many segments were cut from them.
* **`transcription_errors(...)` + `pooled_wer(...)`** — word error rate from the
  stored ASR hypotheses. Two reference modes: `reference_column="text"` for
  ground truth, `reference_condition="real"` to score against the original's own
  decode. `pooled_wer` is `sum(n_err) / sum(n_ref)`, **not** the mean of
  per-utterance rates — on VAD segments a mean lets a two-word segment weigh as
  much as a thirty-word one, and it is not the definition anyone reports.
* **`paired_similarity(...)`** — cosine between each condition and a named
  reference condition of the same `group_id`, from the stored embeddings.
  Structurally identical to `transcription_errors`' pseudo-reference mode.
* **`verification_trials(...)` + `calibrate(...)`** — build the corpus's own
  target/nontarget distributions from its **real** audio, and reduce them to an
  EER, a threshold and the two means that `calibrated_similarity(...)` rescales
  onto: `0` = as similar as two different speakers, `1` = as similar as two real
  recordings of one speaker. `accept_rate(...)` is the headline.
* **`speaker_centroids(...)` + `similarity_to_centroid(...)`** — enrol each
  speaker from several real utterances and score against the centroid. Needs no
  content matching, so it is the unpaired route, as `slope_vs_target` is for
  `recovery_rate`.

## References

`references.bib` at the repository root holds the primary source for every
measure implemented here, each checked against the source document rather than a
secondary description of it. Where an implementation and a published description
disagree, the disagreement is recorded in the metric's module docstring.

## Metric roadmap

**Implemented**

* `level` — duration, RMS/peak dBFS, crest factor. Needs no model, so it doubles
  as the end-to-end test of the runner and as the data sanity check any
  evaluation should start with.
* `egemaps` — the eGeMAPS low-level descriptors via openSMILE (Eyben et al.,
  *IEEE TAFFC* 7(2):190–202, 2016), aggregated here. Covers **spectral tilt**
  (alpha ratio, Hammarberg index, spectral slopes 0–500 and 500–1500 Hz) and, as
  a by-product, F0, jitter, shimmer, HNR, the harmonic differences and the
  formant tracks — though **read the formants from `formants_praat` instead**,
  for the reasons recorded there. Three things to know before using it in a paper:

  1. **`alphaRatio` is high-over-low, and the GeMAPS prose is the outlier.** The
     2016 text describes it as the "ratio of the summed energy from 50–1000 Hz
     and 1–5 kHz"; openSMILE computes the reciprocal, which is what the original
     definition says — Frøkjær-Jensen & Prytz (1975), energy above over below
     1 kHz. Verified here: a flat spectrum gives +6.14 dB against the analytic
     6.24, and boosting below 1 kHz by 20 dB drives it to −12.3. So it **rises**
     with vocal effort, while `hammarbergIndex` **falls**. Cite the 1975 original
     for the definition.
  2. **Level normalisation is on by default and is load-bearing.** Above full
     scale openSMILE's spectral measures are corrupted by many dB with no
     warning (the same signal reads −12.3 at peak 1.0 and +4.5 at peak 5.0), and
     the two spectral slopes drift with level even in range. Normalising to a
     fixed RMS fixes both; absolute level is the `level` metric's job.
  3. **Voiced-only descriptors are aggregated over voiced frames only.**
     openSMILE writes a literal `0.0` into every `_sma3nz` descriptor on unvoiced
     frames, so an all-frame mean would not thin the sample — it would drag the
     mean toward zero in proportion to the silence in the segment.

  Not covered by eGeMAPS: **CPPS** (no cepstral measure in the set) and the
  **Iseli–Alwan corrected** H1\*–H2\* — openSMILE's `logRelF0-H1-H2` is a plain
  harmonic ratio with no formant correction, so it confounds the glottal source
  with F1.

* `f0_praat`, `f0_opensmile`, `f0_swiftf0` — fundamental frequency, measured by
  three trackers: autocorrelation (Boersma 1993, via parselmouth), subharmonic
  summation (Hermes 1988, the eGeMAPS F0) and a small convolutional network on
  the STFT (SwiftF0). Each emits `median_hz`, `mean_semitones`, `sd_semitones`,
  voicing counts and the contour. Things to know:

  1. **All three are emitted and none is adjudicated in the metric.** Praat and
     openSMILE are both periodicity estimators and share their failure modes —
     octave halving when the fundamental is weak relative to the upper harmonics,
     trouble on creak — so two of them agreeing is not a majority among
     independent witnesses. High vocal effort weakens the fundamental, which is
     precisely the regime where the classical pair may *both* halve and the
     neural tracker be right. The reverse worry, that a neural tracker breaks on
     inaudible out-of-distribution conversion artifacts, is equally live. Both
     are testable rather than assertable: compare inter-tracker disagreement on
     real against converted audio from the same speakers, in `compare.py`.
  2. **The analysis range is sex-conditional, widened, and provenance.** Male
     60–400 Hz, female 90–600 Hz, unknown 60–600 Hz — ceilings above Praat's
     usual 300/500 because high effort raises F0 and a low ceiling *causes*
     octave halving. It depends only on the `sex` column, never on the audio, so
     a speaker is analysed identically in every condition. `fmin_hz`, `fmax_hz`
     and `range_source` are on every row, because a different range means a
     different number. openSMILE is the exception: 55–1000 Hz is fixed inside the
     eGeMAPS configuration, and it reports `range_source = "backend_fixed"`.
  3. **The frame grids differ and are reported, not resampled.** Praat and
     openSMILE run at 10 ms; SwiftF0's hop is fixed by its ONNX graph at 16 ms,
     and the first frame centres differ too. Each row carries `t0_s` and
     `time_step_s`, so frame *k* sits at `t0_s + k·time_step_s` and alignment is
     one explicit, visible step downstream.

  SwiftF0 rather than PENN, which is 0.6 points better on PTDB-TUG but imports
  `torbi`, whose prebuilt Viterbi binaries lag current torch and fail at import.
  SwiftF0 needs only onnxruntime and ships its model in the wheel, so it runs on
  an offline compute node. Note that every published tracker ranking is measured
  on clean read speech: **nobody has benchmarked a pitch tracker on high vocal
  effort or codec-decoded audio**, which is this project's regime.

* `formants_praat` — F1–F3 frequency and bandwidth from Praat's Burg LPC
  analysis, via parselmouth. Emits `f{n}_median_hz`, `f{n}_sd_hz`,
  `f{n}_bandwidth_hz` over voiced frames, plus contours. Four things to know:

  1. **`egemaps` already emits formants, and those columns should not be used.**
     eGeMAPSv02 carries `F1frequency_sma3nz` and eight siblings, so
     `egemaps.F1frequency_sma3nz_voiced_mean` has always been in the table. On
     synthesised vowels with known formants, mean absolute error on F1–F3 was
     **3.6 % for Praat against 48.2 % for openSMILE**. The mean is not the
     reason to switch. Sweeping the *source tilt* on one vowel whose vocal tract
     never moves, openSMILE reads F1 at 336 Hz with a modal source and 1197 Hz
     with a flat one — it loses F1 and promotes F2 below about −8 dB/octave.
     Flattening the source is what vocal effort *does*, so that tracker moves
     with the variable under test and would manufacture a clean, large, entirely
     spurious formant displacement out of a tilt change. Praat's error grows by
     3.3 points across the same sweep, openSMILE's by 59. The `egemaps.` prefix
     is what keeps the two families apart in the table.
  2. **Praat's own residual tilt bias lands on F1, and F2/F3 are the clean
     channels.** "Much better" is not "unbiased": across the same sweep Praat's
     F1 drifts 488 → 556 Hz, and end to end through the runner a 12 → 6 dB/octave
     flattening moves F1 by +28 to +43 Hz on a fixed vocal tract — roughly
     5–10 Hz per dB/octave. It points the *same way* as the real effect (effort
     raises F1 and flattens the source), so a measured F1 displacement is an
     upper bound on the articulatory one, and an F1 shift of only a few tens of
     Hz is not evidence on its own. F2 moved 1469 → 1517 Hz and F3 2444 → 2493 Hz
     over that sweep, with no monotone trend: rest a result on those.
  3. **A formant mean is mostly a statement about phonetic content.** Averaged
     over an utterance, F1 is dominated by which vowels were spoken, not by voice
     quality. These columns are interpretable **only in a content-matched paired
     comparison** — same speaker, same words, which is what `group_id` identifies.
     Aggregated across conditions without that pairing they measure the text. The
     eventual fix is restricting to forced-aligned vowel intervals, which is what
     `Requirements.needs_intervals` is reserved for.
  4. **The ceiling is sex-conditional and is provenance.** Male 5000 Hz, female
     5500 Hz, unknown 5500 Hz, from the `sex` column alone and never from the
     audio — the same guarantee the F0 range makes, so one speaker is analysed
     identically in every condition. It is load-bearing: a 4500 Hz ceiling loses
     F3 entirely (1979 Hz read for a true 2500), while 5000/5500/6000 all land
     within ~1 %. `ceiling_hz` and `ceiling_source` are on every row.
  5. **Viterbi `Track` is on by default, and the probe could not show it
     helping.** On synthesised diphthongs with moving formants and noise, mean
     RMSE was 68 Hz raw against 81 Hz tracked, and on the hardest case (a /w/-like
     onset at 20 dB SNR) 288 Hz against 364 Hz. Praat's own 1/1/1 costs were the
     best tracking setting tried; `frequency_cost=0` collapses it to 866 Hz,
     because the reference is the only thing anchoring a pole to a track slot.
     But the probe synthesises exactly five well-separated poles and hands the
     analyser exactly five to find — the regime where raw Burg barely mislabels
     and tracking has nothing to repair. Real speech (nasals, formant-zero pairs,
     creak, genuine mergers) is the case tracking exists for and the one this
     probe cannot reach. `track: false` is a one-line config change and `tracked`
     is emitted per row, so the two never mix silently in one table.

  Track's reference frequencies default to `(2k+1)·ceiling/10`, which reproduces
  Praat's own published defaults (550, 1650, 2750, 3850, 4950 Hz) exactly at
  Praat's own default 5500 Hz ceiling and scales them coherently when the ceiling
  changes with the speaker — which a fixed reference list would not.

* `asr_whisper`, `asr_wav2vec2` — speech recognition, emitting the **hypothesis
  string** and never a WER. `openai/whisper-large-v3` is autoregressive: its
  decoder is a language model, so it renders degraded audio as fluent English
  and its WER *understates* intelligibility loss. `facebook/wav2vec2-large-960h-lv60-self`
  is CTC decoded by argmax with no language model at all, so a destroyed phoneme
  stays destroyed. The gap between them is the diagnostic; neither is
  adjudicated here. Three things to know:

  1. **Level normalisation is −20 dBFS here, not the −27 the phonetics metrics
     use, and this is a Whisper fact checked against its source.** Its feature
     extraction ends with `log_spec = max(log_spec, log_spec.max() - 8.0)` —
     relative, so level-invariant — followed by `(log_spec + 4.0) / 4.0`, which
     is absolute and is not. Magnitudes are squared, so **a gain of X dB shifts
     every input feature by X/40 units** on a scale whose whole span is 2 units.
     At −27 dBFS RMS with speech's 12–15 dB crest, features land 0.30–0.38 units
     low: 15–19% of the range, systematically. −20 is as close to full scale as
     this suite goes without a peak limiter. wav2vec2 z-scores the raw waveform
     and is exactly level-invariant, but is fed the same audio so that a
     difference between the two columns is a difference between two models. Note
     `do_normalize=True` on `WhisperFeatureExtractor` is *not* the fix: it acts
     on the log-mel features, not the waveform.
  2. **Over-long utterances are flagged, never truncated and never raised.**
     Whisper's feature extractor silently truncates past 30 s, which would turn
     every word after the cut into a deletion and yield a plausible-looking,
     badly inflated WER. Such rows get `status = "too_long"` and an empty
     hypothesis; `compare.py` excludes them and prints the count. `status` is
     also the column that makes resume terminate — an utterance decoding
     legitimately to `""` writes an empty cell that pandas reads back as NaN.
  3. **Language is configured, never detected, and decoding is deterministic.**
     Auto-detection can flip on degraded audio, which would let the instrument
     move with the signal it measures. Greedy, temperature 0, no temperature
     fallback, no initial prompt — a prompt is extra language-model prior, which
     is the variable the two backends exist to separate.

  Checkpoints must be cached before an offline node runs them:
  `scripts/download_asr_models.py`. It also fetches the Whisper **tokenizer**,
  because the British–American spelling map that `EnglishTextNormalizer` needs
  ships there rather than with the model — otherwise *scoring* fails offline
  long after transcription succeeded.

* `speaker_ecapa`, `speaker_wavlm` — speaker identity, emitting the **embedding**
  and never a cosine. `speechbrain/spkrec-ecapa-voxceleb` is an 80-band log-mel
  filterbank into a TDNN with attentive statistics pooling (192-d, VoxCeleb1+2);
  `microsoft/wavlm-base-plus-sv` is a self-supervised transformer over the raw
  waveform with an x-vector head (512-d, VoxCeleb1). They share no frontend, no
  architecture and no training set — two checkpoints differing only in their SSL
  encoder would agree for reasons unrelated to the speech under test. Four things
  to know:

  1. **A raw cosine is not reportable, and the calibration is not optional.**
     Same-speaker ECAPA pairs land anywhere in ~0.5–0.85 depending on corpus,
     microphone and segment length, so `0.62` means nothing on its own.
     `verification_trials` + `calibrate` measure this corpus's own same-speaker
     and different-speaker distributions from its **real** audio, and
     `calibrated_similarity` puts any cosine on that axis. Target trials exclude
     same-`group_id` pairs by default: two recordings of the same content share
     phonetic material and often a take, so counting them as "same speaker"
     inflates the yardstick and flatters every later comparison.
  2. **Neither embedder has ever seen high vocal effort.** Both are
     VoxCeleb-trained — celebrity interview speech — and both were trained with a
     criterion that makes within-speaker style variation something to be
     *invariant to*. So a small effort effect is a plausible, informative result
     rather than a broken measurement, and the invariance may hold on real loud
     speech while failing on a converter's artifacts. Report the raw
     distributions alongside any threshold-based accept rate.
  3. **Level normalisation is −20 dBFS, and the two backends need it unequally.**
     `wavlm-base-plus-sv` ships `do_normalize: false`, so unlike the wav2vec2 ASR
     checkpoint it is genuinely level-sensitive. ECAPA's `hyperparams.yaml` sets
     `mean_var_norm: InputNormalization(norm_type: sentence, std_norm: False)`,
     which subtracts each log-mel bin's own time mean — and a gain of *g* adds the
     constant `2·ln g` to every bin — so ECAPA is analytically level-invariant.
     Both are fed the same audio anyway, so a difference between the columns is a
     difference between two models. The *value* −20 has no embedder-side argument
     behind it: it is the ASR level, so all four metrics share one preparation
     pass. `tests/test_speaker.py` probes both claims rather than trusting them.
  4. **Short utterances are flagged and still embedded.** Unlike ASR's
     `too_long`, where truncation makes the hypothesis wrong, a short embedding is
     only noisier — so `status = "short"` is advisory and filtering is the
     reader's decision (`ok_statuses`, `min_duration_s`). The real trap is
     **duration asymmetry**: embedding similarity rises with length, so scoring a
     2 s conversion against a 10 s original measures length as much as identity.
     Frame-synchronous conversion makes this a non-issue by construction; where it
     does not hold, it is fatal and silent.

  Cache both before an offline node runs them:
  `scripts/download_speaker_models.py`. ECAPA needs `--ecapa-savedir` passed
  there *and* as the metric's `savedir:`, because speechbrain loads from a
  directory of its own rather than from the HuggingFace cache.

**Planned.** The list below is the agreed scope. *The details of each metric —
which backend, which aggregation, which normalisation — are settled at
implementation time, one metric at a time, not here.*

1. ~~**WER distortion**~~ — done, see `asr_whisper` / `asr_wav2vec2` above and
   `transcription_errors` / `pooled_wer` in `compare.py`. Both reference modes
   are implemented. Two caveats recorded where they bite rather than here: the
   pseudo-reference mode measures *divergence from the original decode*, not
   intelligibility, and reads 0 whenever the recogniser makes the identical
   error on both conditions; and the Whisper normaliser strips fillers
   (`hmm|mm|mhm|mmm|uh|um`), so on spontaneous corpora disfluencies are not
   scored at all.
2. ~~**F0**~~ — done, see the three trackers above. Still outstanding from this
   family: **contour correlation** against the original and the **inter-tracker
   agreement** diagnostic, both of which need two utterances or two trackers and
   so belong in `compare.py`, computed from the stored contours.
3. **Formants (F1/F2)** — partly done, see `formants_praat` above. What was
   built departs from this plan in one way that matters and must not be forgotten
   when reading the columns: it aggregates over **whole utterances**, not over
   aligned vowel intervals, because the runner still has no interval support
   (`Requirements.needs_intervals`). A whole-utterance formant mean is dominated
   by which vowels were spoken, so those columns are valid *only* in the
   content-matched paired comparison `group_id` provides. Still outstanding from
   this family: the **vowel-interval restriction** that makes an unpaired reading
   legitimate, a **perceptual scale** (Bark or ERB) rather than raw Hz, and
   **vowel space area** as the one-number summary. Also recorded there: Praat's
   own residual F1 bias tracks the source tilt at roughly 5–10 Hz per dB/octave,
   in the same direction as the effect, so F2/F3 are the clean channels.
4. ~~**Spectral tilt**~~ — done, see `egemaps` above. Still outstanding from this
   family: **CPPS** (Hillenbrand & Houde 1996; Praat's `Get CPPS` via
   parselmouth rather than a reimplementation), the **Iseli–Alwan corrected**
   H1\*–H2\* / H1\*–A3\*, and an **LTAS-based** alpha ratio, which is what the
   vocal-effort literature actually uses (Sundberg & Nordenberg, *JASA*
   120(1):453–457, 2006) rather than a mean over per-frame values.
5. **AutoMOS** — a headline MOS predictor plus a sub-dimensional one, so that
   coloration and discontinuity can be attributed separately. Interpreted as
   differences against the codec round trip, never as absolute scores.
6. ~~**Speaker identity**~~ — done, see `speaker_ecapa` / `speaker_wavlm` above
   and `paired_similarity` / `calibrate` / `speaker_centroids` in
   `compare/speaker.py`. Deliberately left out: **AS-Norm** and **PLDA**
   back-ends. Both are standard in verification systems and both would change the
   numbers, but neither is needed to compare conditions *within* one corpus,
   which is all this suite does — calibrating against the corpus's own
   distributions achieves that with no extra parameters to fit.
7. **Signal-level distances** — MCD with c₀ dropped, log-F0 RMSE, LTAS
   difference.

Each family adds one import in `speech_eval/metrics/__init__.py` and populates
its own extra in `pyproject.toml`, so a node needing only spectral tilt never
installs an ASR stack.

**Not settled yet:** which baseline conditions every evaluation run emits
(gain-only, codec round trip, real-at-target). That decision is deferred until
the metrics exist, and costs nothing to defer because it lives in the manifest
and in `compare.py`, not in any metric.
