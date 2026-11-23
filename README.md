# The Impact of Age and Sex on Spanish ASR Performance: data and code

Supplementary material for "The Impact of Age and Sex on Spanish ASR Performance: A Cross-Model Evaluation", submitted to the Journal of Language and Aging Research. It contains the per-utterance evaluation results of the four model conditions on the two test sets (edit counts, not text) and the code that produces every table and figure of the paper from them. Transcriptions, system outputs, audio and model weights are not included.

## Contents

```
data/            per-utterance test tables (inputs of all analyses)
code/            analysis scripts; run_all.py runs them and checks the results against the paper
results/         outputs of the scripts (tables as CSV, model summaries, figures, run log)
build/           provenance script that created data/ from the raw evaluation outputs (not needed to reproduce)
requirements.txt package versions used
```

## Data

`data/coser_test.csv` has one row for each of the 1,494 informant utterances of the COSER test split (28 recordings, 41 informants, 27 locations in five provinces). `data/coacan_test.csv` has one row for each of the 704 utterances of the HablaCanariaBot test split of CoACan; 256 of them come from individual recordings and form the analytic sample of the regressions.

| Column | Meaning |
|---|---|
| `utt_id` | segment file name |
| `audio_id`, `speaker_id`, `speaker` (COSER) | recording, informant label within the recording, and their combination (the speaker unit of the models) |
| `message_id`, `speaker_code` (CoACan) | voice message (the grouping unit of the CoACan models) and a pseudonymous participant code (S01-S79, individual recordings only) |
| `role`, `province`, `enclave` (COSER) | speaker role (all informants), province and location |
| `source`, `num_speakers` (CoACan) | individual or dyadic recording and number of speakers |
| `age`, `sex` | age in years and sex as recorded in the corpus metadata (CoACan: self-reported; one participant reported "Other") |
| `nisqa` | NISQA overall quality score of the recording (COSER) or voice message (CoACan) |
| `duration_sec`, `n_words_ref` | segment duration and number of reference words as stored by the evaluation script |
| `nref` | number of words of the reference transcription after normalisation |
| `ed_whisper_base`, `ed_qwen3_base`, `ed_whisper_lora`, `ed_qwen3_lora` | word-level edit distance between the normalised reference and the normalised output of each condition; an empty output counts as one deletion per reference word |

The tables contain no reference transcriptions and no system outputs. The corpora themselves are available from their maintainers: COSER through http://www.corpusrural.es/ and CoACan through Zenodo (doi:10.5281/zenodo.17413533). The participant codes cannot be linked to names or platform accounts.

## How the numbers are computed

- **WER.** Reference and hypothesis were normalised as in Whisper's basic normaliser (lower case, punctuation replaced by spaces, diacritics kept) and compared by word-level edit distance (`build/00_build_release_data.py`). Per-utterance WER is `100 * ed / nref`. An empty output counts as a WER of 100 %.
- **Corpus-level WER (Table 4).** Uncapped micro-average, with a 95 % percentile interval from 1,000 utterance resamples (`numpy.random.default_rng(42)`). The CoACan column uses the 255 single-speaker utterances with sex reported as female or male.
- **Mixed models (Tables 5 and 6, Figure 2).** `statsmodels` MixedLM, REML, Wald z-tests: `wer ~ age_c + sex + z_nisqa + z_dur + z_rate` with a random intercept for speaker (COSER) or voice message (CoACan). Per-utterance WER is capped at 100 %, age is centred at 70 (COSER) and 22 (CoACan), and log NISQA, log duration and speaking rate are standardised. Table 6 adds pooled OLS with cluster-robust errors on the same units.
- **Age specifications (Table 2).** The COSER models are refitted by maximum likelihood with linear, quadratic, decadal and three-band age; the scan adds a single step at each age threshold that leaves at least two speakers on either side (53 to 88).
- **Age bands (Table 7).** Descriptive means over speakers (COSER) or voice messages (CoACan) of the mean capped WER.
- **Sensitivity analyses.**
  - **WER cap:** 200 % instead of 100 %.
  - **Age:** log(age) instead of centred age.
  - **Fractional logit (Papke-Wooldridge):** a binomial GLM on WER/100 with cluster-robust errors.
  - **COSER grouping:** a random intercept for recording instead of speaker.
  - **CoACan grouping:** a random intercept for speaker (68 speakers) instead of voice message.
- **Paired bootstrap (Section 5.1).** Qwen3 minus Whisper micro-WER per corpus and condition, with both systems scored on the same 1,000 utterance resamples (seed 42); two-sided p is twice the smaller tail proportion around zero.

## Reproducing

```
pip install -r requirements.txt
cd code
python run_all.py
```

The run takes a few minutes, most of it for the age-threshold scan. At the end, `run_all.py` compares 108 values with the paper (Tables 2 and 4 to 7 and the values quoted in the text) and prints the number of differences. The run in `results/run_log.txt` reports none.

## Files in results/

| File | Content |
|---|---|
| `table2_age_specifications.csv`, `age_threshold_scan.csv` | Table 2 and the threshold scan |
| `table4_corpus_wer.csv` | Table 4, plus the 256- and 704-utterance CoACan samples and relative changes under LoRA |
| `table5_lmm.csv`, `lmm_all_terms.csv`, `lmm_summaries.txt` | Table 5 with SE, CI, ICC and intercept SD; all fixed effects; full model summaries |
| `table6_lmm_vs_ols.csv` | Table 6 |
| `table7_age_bands.csv` | Table 7 |
| `sensitivity_cap200.csv`, `sensitivity_log_age.csv`, `sensitivity_fractional_logit.csv`, `coser_recording_intercept.csv` | sensitivity analyses |
| `paired_bootstrap_qwen3_vs_whisper.csv` | paired bootstrap of the system differences |
| `coacan_speaker_intercept.csv` | CoACan models with a speaker intercept |
| `wer_above_100.csv`, `sample_description.txt` | shares of degenerate outputs and the sample counts quoted in the text |
| `figs/fig-werdist.png`, `figs/fig3-forest.png` | Figures 1 and 2 |
