"""Mixed models of sex and age, the comparison with pooled OLS and the recording intercept (Tables 5 and 6)."""
import os

import pandas as pd
import statsmodels.formula.api as smf

from common import CONTROLS, LABELS, RESULTS, SEX, lmm, model_frames

FORM = f"wer ~ age_c + {CONTROLS}"
rows, terms, cmp_rows, rec_rows, summaries = [], [], [], [], []
for corpus, cond, m in model_frames():
    fit = lmm(FORM, m)
    ci = fit.conf_int()
    gv, rv = float(fit.cov_re.iloc[0, 0]), float(fit.scale)
    row = {"corpus": corpus, "condition": LABELS[cond], "n": len(m), "groups": m["group"].nunique(),
           "icc": round(gv / (gv + rv), 3), "sd_intercept": round(gv ** 0.5, 2)}
    for term, name in ((SEX, "sex_male"), ("age_c", "age_per_year")):
        row.update({f"{name}_b": fit.params[term], f"{name}_se": fit.bse[term], f"{name}_ci_low": ci.loc[term, 0],
                    f"{name}_ci_high": ci.loc[term, 1], f"{name}_p": fit.pvalues[term]})
    rows.append(row)
    for term in fit.params.index:
        if term == "Group Var":
            continue
        terms.append({"corpus": corpus, "condition": LABELS[cond], "term": term, "b": fit.params[term],
                      "se": fit.bse[term], "p": fit.pvalues[term]})
    summaries.append(f"==== {corpus} {LABELS[cond]} | n={len(m)} | groups={m['group'].nunique()}\n{fit.summary()}\n")
    ols = smf.ols(FORM, m).fit(cov_type="cluster", cov_kwds={"groups": m["group"]})
    cmp_rows.append({"corpus": corpus, "condition": LABELS[cond], "lmm_b": fit.params[SEX], "lmm_p": fit.pvalues[SEX],
                     "ols_cr_b": ols.params[SEX], "ols_cr_p": ols.pvalues[SEX]})
    if corpus == "COSER":
        rec = lmm(FORM, m, group="audio_id")
        gv2, rv2 = float(rec.cov_re.iloc[0, 0]), float(rec.scale)
        rec_rows.append({"condition": LABELS[cond], "groups": m["audio_id"].nunique(),
                         "sex_male_b": rec.params[SEX], "sex_male_p": rec.pvalues[SEX],
                         "age_b": rec.params["age_c"], "age_p": rec.pvalues["age_c"], "icc": round(gv2 / (gv2 + rv2), 3)})

t = pd.DataFrame(rows)
t.round(4).to_csv(os.path.join(RESULTS, "table5_lmm.csv"), index=False)
pd.DataFrame(terms).round(4).to_csv(os.path.join(RESULTS, "lmm_all_terms.csv"), index=False)
pd.DataFrame(cmp_rows).round(4).to_csv(os.path.join(RESULTS, "table6_lmm_vs_ols.csv"), index=False)
pd.DataFrame(rec_rows).round(4).to_csv(os.path.join(RESULTS, "coser_recording_intercept.csv"), index=False)
open(os.path.join(RESULTS, "lmm_summaries.txt"), "w", encoding="utf8").write("\n".join(summaries))
print(t[["corpus", "condition", "n", "groups", "sex_male_b", "sex_male_p", "age_per_year_b", "age_per_year_p", "icc"]].round(3).to_string(index=False))
