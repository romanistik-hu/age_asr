"""Data loading and model fitting shared by all scripts."""
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)

CONDITIONS = ["whisper_base", "qwen3_base", "whisper_lora", "qwen3_lora"]
LABELS = {"whisper_base": "Whisper baseline", "qwen3_base": "Qwen3 baseline",
          "whisper_lora": "Whisper LoRA", "qwen3_lora": "Qwen3 LoRA"}
SEX = "C(sex, Treatment('Female'))[T.Male]"
CONTROLS = "C(sex, Treatment('Female')) + z_nisqa + z_dur + z_rate"


def load(corpus):
    d = pd.read_csv(os.path.join(DATA, f"{corpus}_test.csv"))
    for c in CONDITIONS:
        d[f"wer_{c}"] = 100 * d[f"ed_{c}"] / d["nref"].where(d["nref"] > 0)
    d["speaking_rate"] = d["n_words_ref"] / d["duration_sec"]
    d["group"] = d["speaker"] if corpus == "coser" else d["message_id"]
    return d


def individuals(d):
    return d[d["source"] == "bot_individual"].copy()


def zscore(x):
    return (x - x.mean()) / x.std()


def prep(d, cond, centre, cap=100, log_age=False):
    col = f"wer_{cond}"
    m = d[d[col].notna() & (d["nisqa"] > 0) & (d["duration_sec"] > 0) & d["speaking_rate"].notna()].copy()
    m["wer"] = m[col].clip(upper=cap) if cap is not None else m[col]
    m["age_c"] = m["age"] - centre
    m["log_age"] = np.log(m["age"])
    m["z_nisqa"] = zscore(np.log(m["nisqa"]))
    m["z_dur"] = zscore(np.log(m["duration_sec"]))
    m["z_rate"] = zscore(m["speaking_rate"])
    return m


def lmm(formula, m, reml=True, group="group"):
    import statsmodels.formula.api as smf
    for opt in ("powell", "nm", "cg"):
        try:
            fit = smf.mixedlm(formula, m, groups=m[group]).fit(reml=reml, method=opt)
            if np.all(np.isfinite(fit.bse.values)) and fit.bse.get("age_c", 0) < 1e3:
                return fit
        except Exception:
            pass
    raise RuntimeError("no optimiser converged for " + formula)


CORPORA = {"COSER": ("coser", 70), "CoACan": ("coacan", 22)}


def model_frames(cap=100, log_age=False):
    out = []
    for corpus, (name, centre) in CORPORA.items():
        d = load(name)
        if name == "coacan":
            d = individuals(d)
        for c in CONDITIONS:
            out.append((corpus, c, prep(d, c, centre, cap=cap, log_age=log_age)))
    return out
