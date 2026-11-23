"""Builds the release tables from the raw evaluation outputs (not needed to reproduce the paper)."""
import json
import os
import re
import sys

import pandas as pd

RAW, XLSX, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
S = os.path.join(RAW, "sociolinguistic")
Q = os.path.join(RAW, "{}", "results-qwen-{}-{}", "transcripts", "{}")
QWEN = {
    ("coser", "base"): Q.format("qwen_coser", "coser", "BASE", "qwen3_base_coser_test_utterance_prediction.json"),
    ("coser", "lora"): Q.format("qwen_coser", "coser", "FT-LORA-V2", "transcripts_qwenlorafull_coser_test.json"),
    ("coacan", "base"): Q.format("qwen_bot", "bot", "BASE", "qwen3_base_bot_test_utterance_prediction.json"),
    ("coacan", "lora"): Q.format("qwen_bot", "bot", "FT-LORA-V2", "transcripts_qwenlorafull__bot_test.json"),
}
CONDITIONS = ["whisper_base", "qwen3_base", "whisper_lora", "qwen3_lora"]


def normalize(t):
    if not isinstance(t, str):
        return ""
    t = t.lower()
    t = re.sub(r'[¿¡!?.,;:()"\'\-—–\[\]{}…«»]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


def edits(ref, hyp):
    r, h = normalize(ref).split(), normalize(hyp).split()
    prev = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        cur = [i] + [0] * len(h)
        for j in range(1, len(h) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
        prev = cur
    return prev[len(h)]


def qwen(path, name):
    df = pd.DataFrame(json.load(open(path, encoding="utf8")))
    if "condition" in df and "no_metadata" in set(df["condition"]):
        df = df[df["condition"] == "no_metadata"]
    df["k"] = df["reference_raw"].apply(normalize)
    return df[["k", "prediction_clean"]].rename(columns={"prediction_clean": name}).drop_duplicates("k")


def base_table(prefix, corpus):
    zs = pd.read_csv(os.path.join(S, f"{prefix}_zero_shot.csv"))
    ft = pd.read_csv(os.path.join(S, f"{prefix}_normal_finetuned.csv"))
    zs["k"] = zs["label_text"].apply(normalize)
    ft["k"] = ft["label_text"].apply(normalize)
    b = zs[["k", "label_text", "pred_text", "audio_quality", "audio_duration_sec", "n_words_ref"]].rename(
        columns={"label_text": "reference", "pred_text": "hyp_whisper_base", "audio_quality": "nisqa",
                 "audio_duration_sec": "duration_sec"})
    b = b.merge(ft[["k", "pred_text"]].rename(columns={"pred_text": "hyp_whisper_lora"}).drop_duplicates("k"), on="k", how="left")
    b = b.merge(qwen(QWEN[(corpus, "base")], "hyp_qwen3_base"), on="k", how="left")
    b = b.merge(qwen(QWEN[(corpus, "lora")], "hyp_qwen3_lora"), on="k", how="left")
    return b


def counts(b):
    b["nref"] = [len(normalize(r).split()) for r in b["reference"]]
    for c in CONDITIONS:
        b[f"ed_{c}"] = [edits(r, h if isinstance(h, str) else "") for r, h in zip(b["reference"], b[f"hyp_{c}"])]
    return b


NUM = ["nisqa", "duration_sec", "n_words_ref", "nref"] + [f"ed_{c}" for c in CONDITIONS]
report = []

c = counts(base_table("coser", "coser"))
tm = pd.read_csv(os.path.join(RAW, "coser_test_metadata.csv"))
tm["k"] = tm["sentence"].apply(normalize)
tm = tm[["k", "file_name", "audio_id", "speaker_id", "role", "age", "sex", "province", "enclave"]].drop_duplicates("k")
c = c.merge(tm, on="k", how="left")
c["speaker"] = c["audio_id"].astype(str) + "_" + c["speaker_id"].astype(str)
c = c[["file_name", "audio_id", "speaker_id", "speaker", "role", "province", "enclave", "age", "sex"] + NUM]
c = c.rename(columns={"file_name": "utt_id"})
c.to_csv(os.path.join(OUT, "coser_test.csv"), index=False)
report.append(f"COSER: {len(c)} utterances, {c.audio_id.nunique()} recordings, {c.speaker.nunique()} speakers, "
              f"missing ids {int(c.utt_id.isna().sum())}")

b = counts(base_table("bot", "coacan"))
bm = pd.read_csv(os.path.join(RAW, "bot_test_metadata.csv"))
bm["k"] = bm["sentence"].apply(normalize)
bm = bm[["k", "file_name", "id", "source", "num_speakers", "age", "sex"]].drop_duplicates("k")
b = b.merge(bm, on="k", how="left")
ids = pd.read_excel(XLSX, usecols=["id", "usuario_id"])
ind_ids = ids[ids["id"].astype(str).str.startswith("IND_")].sort_values("id")
order = list(dict.fromkeys(ind_ids["usuario_id"]))
code = {u: f"S{i + 1:02d}" for i, u in enumerate(order)}
b["speaker_code"] = b["id"].map({m: code[u] for m, u in zip(ind_ids["id"], ind_ids["usuario_id"])})
b = b[["file_name", "id", "speaker_code", "source", "num_speakers", "age", "sex"] + NUM]
b = b.rename(columns={"file_name": "utt_id", "id": "message_id"})
b.to_csv(os.path.join(OUT, "coacan_test.csv"), index=False)
ind = b[b["source"] == "bot_individual"]
report.append(f"CoACan: {len(b)} utterances ({len(ind)} individual), individual messages {ind.message_id.nunique()}, "
              f"individual speakers {ind.speaker_code.nunique()} of {len(code)} coded, missing ids {int(b.utt_id.isna().sum())}")
print("\n".join(report))
