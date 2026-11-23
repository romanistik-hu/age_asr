"""CoACan models with a random intercept for speaker."""
import os

import pandas as pd

from common import CONTROLS, LABELS, RESULTS, SEX, lmm, model_frames

FORM = f"wer ~ age_c + {CONTROLS}"
rows = []
for corpus, cond, m in model_frames():
    if corpus != "CoACan":
        continue
    fit = lmm(FORM, m, group="speaker_code")
    gv, rv = float(fit.cov_re.iloc[0, 0]), float(fit.scale)
    rows.append({"condition": LABELS[cond], "groups": m["speaker_code"].nunique(), "n": len(m),
                 "sex_male_b": round(fit.params[SEX], 4), "sex_male_se": round(fit.bse[SEX], 4),
                 "sex_male_p": round(fit.pvalues[SEX], 4), "age_b": round(fit.params["age_c"], 4),
                 "age_p": round(fit.pvalues["age_c"], 4), "icc": round(gv / (gv + rv), 3)})
out = pd.DataFrame(rows)
out.to_csv(os.path.join(RESULTS, "coacan_speaker_intercept.csv"), index=False)
print(out.to_string(index=False))
