"""Corpus-level WER, relative change under LoRA and outputs above 100 % WER (Table 4)."""
import os

import numpy as np
import pandas as pd

from common import CONDITIONS, LABELS, RESULTS, individuals, load


def micro(d, c):
    d = d[d[f"ed_{c}"].notna() & (d["nref"] > 0)]
    return 100 * d[f"ed_{c}"].sum() / d["nref"].sum()


def boot(d, c, seed=42, n=1000):
    d = d[d[f"ed_{c}"].notna() & (d["nref"] > 0)]
    e, r = d[f"ed_{c}"].to_numpy(), d["nref"].to_numpy()
    rng = np.random.default_rng(seed)
    v = [100 * e[i].sum() / r[i].sum() for i in (rng.integers(0, len(e), len(e)) for _ in range(n))]
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


coser = load("coser")
coacan = load("coacan")
ind = individuals(coacan)
samples = {"COSER": coser, "CoACan (255, female or male)": ind[ind["sex"].isin(["Female", "Male"])],
           "CoACan (256 individual)": ind, "CoACan (704, full test split)": coacan}
rows = []
for name, d in samples.items():
    for c in CONDITIONS:
        lo, hi = boot(d, c)
        rows.append({"sample": name, "n": len(d), "condition": LABELS[c], "micro_wer": round(micro(d, c), 2),
                     "ci_low": round(lo, 2), "ci_high": round(hi, 2)})
t = pd.DataFrame(rows)
for name in samples:
    for arch in ("whisper", "qwen3"):
        b = micro(samples[name], f"{arch}_base")
        a = micro(samples[name], f"{arch}_lora")
        t.loc[(t["sample"] == name) & (t["condition"] == LABELS[f"{arch}_lora"]), "relative_change_pct"] = round(100 * (a - b) / b, 1)
t.to_csv(os.path.join(RESULTS, "table4_corpus_wer.csv"), index=False)

rows = []
for name, d in (("COSER", coser), ("CoACan (256 individual)", ind), ("CoACan (704, full test split)", coacan)):
    for c in CONDITIONS:
        w = d[f"wer_{c}"]
        rows.append({"sample": name, "condition": LABELS[c], "n": int(w.notna().sum()),
                     "above_100": int((w > 100).sum()), "share_pct": round(100 * (w > 100).mean(), 2),
                     "median_wer": round(w.median(), 1), "max_wer": round(w.max(), 0)})
pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "wer_above_100.csv"), index=False)
print(t.to_string(index=False))
