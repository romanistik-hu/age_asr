"""Age bands and the sample counts quoted in the text (Table 7)."""
import os

import numpy as np
import pandas as pd

from common import RESULTS, individuals, load

coser = load("coser")
coacan = load("coacan")
ind = individuals(coacan)
rows = []
for c in ("whisper_base", "qwen3_base"):
    d = coser.assign(w=coser[f"wer_{c}"].clip(upper=100))
    u = d.groupby("speaker").agg(m=("w", "mean"), age=("age", "first"), sex=("sex", "first")).reset_index()
    u["band"] = pd.cut(u["age"], [40, 70, 85, 200], labels=["50-70", "71-85", "86+"]).astype(str)
    for b, g in u.groupby("band"):
        rows.append({"corpus": "COSER", "unit": "speaker", "band": b, "condition": c, "n": len(g),
                     "pct_female": round(100 * (g.sex == "Female").mean()), "mean_wer": round(g.m.mean(), 1)})
    d = ind.assign(w=ind[f"wer_{c}"].clip(upper=100))
    u = d.groupby("message_id").agg(m=("w", "mean"), age=("age", "first"), sex=("sex", "first")).reset_index()
    u["band"] = np.where(u["age"] >= 26, "26+", "18-25")
    for b, g in u.groupby("band"):
        rows.append({"corpus": "CoACan", "unit": "voice message", "band": b, "condition": c, "n": len(g),
                     "pct_female": round(100 * (g.sex == "Female").mean()), "mean_wer": round(g.m.mean(), 1)})
pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "table7_age_bands.csv"), index=False)

L = []
g = coser.groupby("speaker").agg(n=("utt_id", "size"), sex=("sex", "first"), age=("age", "first"))
L.append(f"COSER test: {len(coser)} informant utterances, {coser.audio_id.nunique()} recordings, "
         f"{coser.speaker.nunique()} speakers, {coser.enclave.nunique()} locations, {coser.province.nunique()} provinces "
         f"({', '.join(sorted(coser.province.unique()))})")
L.append(f"  utterances per speaker: mean {g.n.mean():.1f}, range {g.n.min()}-{g.n.max()}; "
         + "; ".join(f"{s}: {int((g.sex == s).sum())} speakers, mean {g[g.sex == s].n.mean():.1f}" for s in ("Male", "Female")))
L.append(f"  age range {g.age.min():.0f}-{g.age.max():.0f}; 86+: {int((g.age >= 86).sum())} speakers "
         f"({', '.join(str(int(a)) for a in sorted(g[g.age >= 86].age))}), sexes {g[g.age >= 86].sex.unique().tolist()}")
r = coser.groupby("audio_id").speaker.nunique()
L.append(f"  recordings with more than one informant: {int((r > 1).sum())}")
L.append(f"  segment duration: median {coser.duration_sec.median():.1f} s, range {coser.duration_sec.min():.1f}-{coser.duration_sec.max():.1f} s, "
         f"{coser.duration_sec.sum() / 3600:.2f} h; NISQA median {coser.nisqa.median():.2f}")
L.append(f"CoACan test: {len(coacan)} utterances; individual {len(ind)}, dyadic {int((coacan.source != 'bot_individual').sum())} "
         f"(from {coacan[coacan.source != 'bot_individual'].message_id.nunique()} dyadic messages)")
L.append(f"  individual: {ind.message_id.nunique()} voice messages by {ind.speaker_code.nunique()} speakers; "
         f"utterances per message {len(ind) / ind.message_id.nunique():.2f} (range {ind.groupby('message_id').size().min()}-"
         f"{ind.groupby('message_id').size().max()}), per speaker {len(ind) / ind.speaker_code.nunique():.2f} "
         f"(range {ind.groupby('speaker_code').size().min()}-{ind.groupby('speaker_code').size().max()})")
L.append(f"  sex (utterances): {ind.sex.value_counts().to_dict()}; age range {ind.age.min():.0f}-{ind.age.max():.0f}, median {ind.age.median():.0f}")
o = ind[ind.age >= 26]
L.append(f"  26+ tail: {len(o)} utterances, {o.message_id.nunique()} voice messages, {o.speaker_code.nunique()} speakers")
oth = ind[~ind.sex.isin(["Female", "Male"])]
L.append(f"  sex 'Other': {len(oth)} utterance(s), message(s) {oth.message_id.unique().tolist()}; "
         f"without it: {len(ind) - len(oth)} utterances, {ind[ind.sex.isin(['Female', 'Male'])].message_id.nunique()} messages, "
         f"{ind[ind.sex.isin(['Female', 'Male'])].speaker_code.nunique()} speakers")
L.append(f"  segment duration (individual): median {ind.duration_sec.median():.1f} s, range {ind.duration_sec.min():.1f}-{ind.duration_sec.max():.1f} s; "
         f"all 704: median {coacan.duration_sec.median():.1f} s, range {coacan.duration_sec.min():.1f}-{coacan.duration_sec.max():.1f} s, "
         f"{coacan.duration_sec.sum() / 3600:.2f} h")
L.append(f"  NISQA median: individual {ind.nisqa.median():.2f}, all 704 {coacan.nisqa.median():.2f}")
open(os.path.join(RESULTS, "sample_description.txt"), "w", encoding="utf8").write("\n".join(L) + "\n")
print("\n".join(L))
