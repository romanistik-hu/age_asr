"""Runs every script and checks the results against the paper."""
import os
import runpy
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import RESULTS

for s in ("01_corpus_wer.py", "02_mixed_models.py", "03_age_specifications.py", "04_sensitivity.py",
          "05_descriptives.py", "06_figures.py",
          "07_paired_bootstrap.py", "08_coacan_speaker_intercept.py"):
    print(f"\n######## {s}")
    runpy.run_path(os.path.join(HERE, s), run_name="__main__")

R = lambda f: pd.read_csv(os.path.join(RESULTS, f))
fails = []


def check(name, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"{'ok  ' if ok else 'DIFF'} {name}: {got:.4f} (paper {want})")
    if not ok:
        fails.append(name)


t4 = R("table4_corpus_wer.csv")
paper4 = {("COSER", "Whisper baseline"): (31.58, 30.27, 33.08), ("COSER", "Qwen3 baseline"): (30.33, 29.52, 31.18),
          ("COSER", "Whisper LoRA"): (30.43, 28.04, 32.87), ("COSER", "Qwen3 LoRA"): (31.27, 29.69, 32.82),
          ("CoACan (255, female or male)", "Whisper baseline"): (10.17, 8.89, 11.49),
          ("CoACan (255, female or male)", "Qwen3 baseline"): (9.33, 8.18, 10.49),
          ("CoACan (255, female or male)", "Whisper LoRA"): (6.20, 5.27, 7.13),
          ("CoACan (255, female or male)", "Qwen3 LoRA"): (7.73, 6.64, 8.80)}
for (s, c), (w, lo, hi) in paper4.items():
    r = t4[(t4["sample"] == s) & (t4["condition"] == c)].iloc[0]
    check(f"Table 4 {s[:6]} {c} WER", r.micro_wer, w, 0.005)
    check(f"Table 4 {s[:6]} {c} CI low", r.ci_low, lo, 0.005)
    check(f"Table 4 {s[:6]} {c} CI high", r.ci_high, hi, 0.005)

t5 = R("table5_lmm.csv")
paper5 = {("COSER", "Whisper baseline"): (6.32, .045, 0.047, .763), ("COSER", "Qwen3 baseline"): (6.37, .019, 0.066, .622),
          ("COSER", "Whisper LoRA"): (2.92, .350, 0.104, .494), ("COSER", "Qwen3 LoRA"): (6.88, .063, 0.080, .658),
          ("CoACan", "Whisper baseline"): (2.54, .152, 0.127, .314), ("CoACan", "Qwen3 baseline"): (5.08, .009, 0.152, .271),
          ("CoACan", "Whisper LoRA"): (2.41, .118, 0.043, .700), ("CoACan", "Qwen3 LoRA"): (2.66, .121, 0.104, .406)}
for (corp, c), (b, p, a, ap) in paper5.items():
    r = t5[(t5.corpus == corp) & (t5.condition == c)].iloc[0]
    check(f"Table 5 {corp} {c} sex b", r.sex_male_b, b, 0.005)
    check(f"Table 5 {corp} {c} sex p", r.sex_male_p, p, 0.0006)
    check(f"Table 5 {corp} {c} age b", r.age_per_year_b, a, 0.0005)
    check(f"Table 5 {corp} {c} age p", r.age_per_year_p, ap, 0.0006)

t6 = R("table6_lmm_vs_ols.csv")
for (corp, c), b in {("COSER", "Whisper baseline"): 10.04, ("COSER", "Qwen3 baseline"): 9.54, ("COSER", "Whisper LoRA"): 7.08,
                     ("COSER", "Qwen3 LoRA"): 8.85, ("CoACan", "Whisper baseline"): 2.59, ("CoACan", "Qwen3 baseline"): 5.06,
                     ("CoACan", "Whisper LoRA"): 2.41, ("CoACan", "Qwen3 LoRA"): 2.66}.items():
    check(f"Table 6 {corp} {c} OLS b", t6[(t6.corpus == corp) & (t6.condition == c)].iloc[0].ols_cr_b, b, 0.005)

t2 = R("table2_age_specifications.csv")
paper2 = {"Quadratic": (1.59, 1.84, 2.00, 1.92), "Decadal buckets": (4.19, 2.36, 1.25, 2.76),
          "Three-level grouping": (1.04, 2.38, -0.18, 1.92)}
for spec, vals in paper2.items():
    for c, v in zip(("Whisper baseline", "Whisper LoRA", "Qwen3 baseline", "Qwen3 LoRA"), vals):
        check(f"Table 2 {spec} {c}", t2[(t2.specification == spec) & (t2.condition == c)].iloc[0].delta_aic_vs_linear, v, 0.005)

t7 = R("table7_age_bands.csv")
for (corp, band, cond), v in {("COSER", "50-70", "whisper_base"): 36.1, ("COSER", "71-85", "whisper_base"): 38.3,
                              ("COSER", "86+", "whisper_base"): 36.6, ("COSER", "50-70", "qwen3_base"): 34.8,
                              ("COSER", "71-85", "qwen3_base"): 36.1, ("COSER", "86+", "qwen3_base"): 36.2,
                              ("CoACan", "18-25", "whisper_base"): 11.0, ("CoACan", "26+", "whisper_base"): 12.0,
                              ("CoACan", "18-25", "qwen3_base"): 11.0, ("CoACan", "26+", "qwen3_base"): 11.6}.items():
    check(f"Table 7 {corp} {band} {cond}", t7[(t7.corpus == corp) & (t7.band == band) & (t7.condition == cond)].iloc[0].mean_wer, v, 0.05)

pb = R("paired_bootstrap_qwen3_vs_whisper.csv")
for (corp, cond), (d, lo, hi, p) in {("COSER", "base"): (-1.26, -2.58, -0.24, .018), ("CoACan", "base"): (-0.84, -2.06, 0.28, .17),
                                     ("COSER", "lora"): (0.84, -1.49, 3.11, .46), ("CoACan", "lora"): (1.53, 0.62, 2.42, .001)}.items():
    r = pb[(pb.corpus == corp) & (pb.cond == cond)].iloc[0]
    check(f"Paired bootstrap {corp} {cond} delta", r.delta_q_minus_w, d, 0.005)
    check(f"Paired bootstrap {corp} {cond} CI lo", r.ci_lo, lo, 0.005)
    check(f"Paired bootstrap {corp} {cond} CI hi", r.ci_hi, hi, 0.005)
    check(f"Paired bootstrap {corp} {cond} p", max(r.p_two_sided, .001), p, 0.006)

sp = R("coacan_speaker_intercept.csv").set_index("condition")
check("CoACan speaker intercept Qwen3 baseline sex b", sp.loc["Qwen3 baseline", "sex_male_b"], 4.51, 0.005)
check("CoACan speaker intercept Qwen3 baseline sex p", sp.loc["Qwen3 baseline", "sex_male_p"], .037, 0.0006)
for c in ("Whisper baseline", "Whisper LoRA", "Qwen3 LoRA"):
    check(f"CoACan speaker intercept {c} sex p >= .05", float(sp.loc[c, "sex_male_p"] >= .05), 1.0, 0)
check("CoACan speaker intercept min age p >= .05", float(sp["age_p"].min() >= .05), 1.0, 0)

print(f"\n{len(fails)} differences" + (": " + "; ".join(fails) if fails else ""))
