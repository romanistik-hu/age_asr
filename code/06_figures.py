"""Figures 1 and 2."""
import os

import matplotlib
import numpy as np
import pandas as pd

from common import LABELS, RESULTS, SEX, individuals, lmm, load, model_frames, CONTROLS

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = os.path.join(RESULTS, "figs")
os.makedirs(FIGS, exist_ok=True)

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": INK, "font.family": "DejaVu Sans"})

rows = []
for corpus, cond, m in model_frames():
    f = lmm(f"wer ~ age_c + {CONTROLS}", m)
    ci = f.conf_int()
    rows.append({"corpus": corpus, "cond": LABELS[cond], "sex": f.params[SEX], "sex_lo": ci.loc[SEX, 0], "sex_hi": ci.loc[SEX, 1],
                 "age": 10 * f.params["age_c"], "age_lo": 10 * ci.loc["age_c", 0], "age_hi": 10 * ci.loc["age_c", 1]})
R = pd.DataFrame(rows)
ypos = [8.2, 7.2, 6.2, 5.2, 3.0, 2.0, 1.0, 0.0]
fig, axs = plt.subplots(1, 2, figsize=(5.6, 3.3), sharex=True, sharey=True, gridspec_kw={"wspace": 0.06})
panels = (("sex", "Sex (male)", BLUE, "o", 6.5), ("age", "Age (per decade)", ORANGE, "s", 6.0))
for ax, (key, title, colour, marker, ms) in zip(axs, panels):
    ax.axvline(0, color=MUTED, lw=0.9, zorder=1)
    for y, (_, r) in zip(ypos, R.iterrows()):
        lo, hi, est = r[f"{key}_lo"], r[f"{key}_hi"], r[key]
        excludes_zero = lo > 0 or hi < 0
        ax.plot([lo, hi], [y, y], color=colour, lw=1.6, solid_capstyle="round", zorder=2)
        ax.plot(est, y, marker=marker, ms=ms, mfc=colour if excludes_zero else SURFACE, mec=colour, mew=1.6, zorder=3)
    ax.set_title(title, fontsize=8.8, color=INK, loc="center", pad=6)
    ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("WER (pp)")
axs[0].set_yticks(ypos)
axs[0].set_yticklabels([])
axs[0].set_ylim(-0.8, 9.9)
LABEL_X = -0.62
for y, name in ((9.15, "COSER"), (3.95, "CoACan")):
    axs[0].text(LABEL_X, y, name, transform=axs[0].get_yaxis_transform(), ha="left", va="center", fontsize=8.5,
                fontweight="bold", color=INK)
for y, name in zip(ypos, R["cond"]):
    axs[0].text(LABEL_X + 0.04, y, name, transform=axs[0].get_yaxis_transform(), ha="left", va="center", fontsize=8.5,
                color=INK)
from matplotlib.lines import Line2D
handles = [Line2D([], [], marker="o", ls="", ms=6, mfc=MUTED, mec=MUTED, label="95% CI excludes zero"),
           Line2D([], [], marker="o", ls="", ms=6, mfc=SURFACE, mec=MUTED, mew=1.4, label="95% CI includes zero")]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.6, -0.01), labelcolor=MUTED)
fig.subplots_adjust(left=0.235, right=0.985, top=0.92, bottom=0.215)
plt.savefig(os.path.join(FIGS, "fig3-forest.png"), dpi=300, facecolor=SURFACE)
plt.close()

fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
for a, d, t in ((axs[0], load("coser"), "COSER"), (axs[1], individuals(load("coacan")), "CoACan (individuals)")):
    w = d["wer_whisper_base"].dropna()
    a.hist(w.clip(upper=200), bins=40, color="#4472C4", alpha=0.85, edgecolor="white")
    a.axvline(100, color="crimson", ls="--", lw=1)
    a.set_title(f"{t}: median {w.median():.0f}%, >100%: {100 * (w > 100).mean():.1f}%")
    a.set_xlabel("per-utterance WER (%)")
    a.set_ylabel("count")
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig-werdist.png"), dpi=200)
plt.close()
print(R.round(2).to_string(index=False))
