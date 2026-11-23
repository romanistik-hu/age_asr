"""Age specifications and the single-threshold scan in COSER (Table 2)."""
import os

import numpy as np
import pandas as pd

from common import CONTROLS, LABELS, RESULTS, lmm, model_frames

SPECS = {"Linear": "age_c", "Quadratic": "age_c + age_c2", "Decadal buckets": "C(decade)",
         "Three-level grouping": "C(band3, Treatment('50-70'))"}
rows, scan = [], []
for corpus, cond, m in model_frames():
    if corpus != "COSER":
        continue
    m = m.copy()
    m["age_c2"] = m["age_c"] ** 2
    m["decade"] = (np.floor(m["age"] / 10) * 10).astype(int).astype(str) + "s"
    m["band3"] = pd.cut(m["age"], [40, 70, 85, 200], labels=["50-70", "71-85", "86+"]).astype(str)
    fits = {s: lmm(f"wer ~ {t} + {CONTROLS}", m, reml=False) for s, t in SPECS.items()}
    for s, f in fits.items():
        rows.append({"condition": LABELS[cond], "specification": s, "loglik": f.llf, "aic": f.aic,
                     "delta_aic_vs_linear": f.aic - fits["Linear"].aic})
    lin = fits["Linear"].llf
    for t in range(int(m["age"].min()) + 1, int(m["age"].max()) + 1):
        above = m.loc[m["age"] >= t, "group"].nunique()
        below = m.loc[m["age"] < t, "group"].nunique()
        if above < 2 or below < 2:
            continue
        m["step"] = (m["age"] >= t).astype(int)
        f = lmm(f"wer ~ age_c + step + {CONTROLS}", m, reml=False)
        scan.append({"condition": LABELS[cond], "threshold": t, "speakers_at_or_above": above,
                     "step_b": f.params["step"], "lr_vs_linear": 2 * (f.llf - lin)})
t = pd.DataFrame(rows)
t.round(3).to_csv(os.path.join(RESULTS, "table2_age_specifications.csv"), index=False)
s = pd.DataFrame(scan)
s.round(3).to_csv(os.path.join(RESULTS, "age_threshold_scan.csv"), index=False)
print(t.pivot(index="specification", columns="condition", values="delta_aic_vs_linear").round(2).to_string())
for c, g in s.groupby("condition", sort=False):
    hi = g[(g.threshold >= 80) & (g.threshold <= 88)]
    best = g.loc[g.lr_vs_linear.idxmax()]
    print(f"{c}: max LR at 80-88 = {hi.lr_vs_linear.max():.2f}; best threshold {int(best.threshold)} "
          f"(LR {best.lr_vs_linear:.2f}, step {best.step_b:+.2f}); LR at 71 = {g[g.threshold == 71].lr_vs_linear.iloc[0]:.2f}")
