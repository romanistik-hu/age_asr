"""Paired bootstrap of the difference between Qwen3 and Whisper."""
import os

import numpy as np
import pandas as pd

from common import DATA, RESULTS

B, SEED = 1000, 42
rng = np.random.default_rng(SEED)
rows = []
for corpus, fname, individual in (("COSER", "coser_test.csv", False), ("CoACan", "coacan_test.csv", True)):
    d = pd.read_csv(os.path.join(DATA, fname))
    if individual:
        d = d[(d.num_speakers == 1) & d.sex.isin(["Female", "Male"])]
    n = d["nref"].values
    for cond in ("base", "lora"):
        w, q = d[f"ed_whisper_{cond}"].values, d[f"ed_qwen3_{cond}"].values

        def micro(e, idx):
            return 100 * e[idx].sum() / n[idx].sum()

        full = np.arange(len(d))
        delta = micro(q, full) - micro(w, full)
        boots = np.array([micro(q, i) - micro(w, i) for i in (rng.integers(0, len(d), len(d)) for _ in range(B))])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        p = 2 * min((boots >= 0).mean(), (boots <= 0).mean())
        rows.append({"corpus": corpus, "cond": cond, "n_utt": len(d), "whisper": round(micro(w, full), 2),
                     "qwen3": round(micro(q, full), 2), "delta_q_minus_w": round(delta, 2),
                     "ci_lo": round(lo, 2), "ci_hi": round(hi, 2), "p_two_sided": round(p, 3)})
out = pd.DataFrame(rows)
out.to_csv(os.path.join(RESULTS, "paired_bootstrap_qwen3_vs_whisper.csv"), index=False)
print(out.to_string(index=False))
