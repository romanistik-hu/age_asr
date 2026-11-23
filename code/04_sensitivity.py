"""Sensitivity analyses for the models of Table 5."""
import os

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from common import CONTROLS, LABELS, RESULTS, SEX, lmm, model_frames

out = {"cap200": [], "log_age": [], "fractional_logit": []}
for corpus, cond, m in model_frames(cap=200):
    f = lmm(f"wer ~ age_c + {CONTROLS}", m)
    out["cap200"].append({"corpus": corpus, "condition": LABELS[cond], "n": len(m), "sex_male_b": f.params[SEX],
                          "sex_male_p": f.pvalues[SEX], "age_b": f.params["age_c"], "age_p": f.pvalues["age_c"]})
for corpus, cond, m in model_frames():
    f = lmm(f"wer ~ log_age + {CONTROLS}", m)
    out["log_age"].append({"corpus": corpus, "condition": LABELS[cond], "n": len(m), "sex_male_b": f.params[SEX],
                           "sex_male_p": f.pvalues[SEX], "log_age_b": f.params["log_age"], "log_age_p": f.pvalues["log_age"]})
    m = m.copy()
    m["y"] = m["wer"].clip(0, 100) / 100
    g = smf.glm(f"y ~ age_c + {CONTROLS}", m, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": m["group"]})
    out["fractional_logit"].append({"corpus": corpus, "condition": LABELS[cond], "n": len(m),
                                    "sex_male_logit": g.params[SEX], "sex_male_p": g.pvalues[SEX],
                                    "age_logit": g.params["age_c"], "age_p": g.pvalues["age_c"]})
names = {"cap200": "sensitivity_cap200.csv", "log_age": "sensitivity_log_age.csv",
         "fractional_logit": "sensitivity_fractional_logit.csv"}
for k, rows in out.items():
    t = pd.DataFrame(rows).round(4)
    t.to_csv(os.path.join(RESULTS, names[k]), index=False)
    print(f"--- {k}\n{t.to_string(index=False)}")
