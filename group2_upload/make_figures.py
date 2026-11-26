# make_figures.py
"""Group 2: results table + figures for the 4 Dec meeting and the essay.

Reads results/*.csv from run_all.py and channel_study.py. Skips any figure whose CSV
doesn't exist yet, so it can be run at any point.

Outputs:
  results/summary_table.csv / .md   mean +/- std over S1..S5 per exercise x model x class_weight
  figures/fig1_within_subject.png   balanced accuracy: logreg vs SVM per exercise (+ cross-subject)
  figures/fig2_channel_count.png    balanced accuracy vs number of electrodes
  figures/fig3_pair_heatmap.png     every 2-electrode pair (and single electrodes on the diagonal)
  figures/fig4_class_weight.png     what class_weight="balanced" trades: rest vs movement errors

Usage: python make_figures.py [--class-weight none|balanced] [--exercise 1]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
RES, FIG = ROOT / "results", ROOT / "figures"

# Colour = model, the same in every figure. Text stays dark grey, never the series colour.
COLORS = {"logreg": "#257de0", "svm": "#f47230"}
NAMES = {"logreg": "Logistic regression", "svm": "RBF SVM"}
TEXT, TEXT_2, GRID = "#191919", "#666666", "#e6e6e6"
SEQ_BLUE = ["#cce3fd", "#9dc8f7", "#6aabf1", "#338ced", "#2370cd", "#1757ac", "#0e3d7e"]

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": TEXT_2, "axes.labelcolor": TEXT, "xtick.color": TEXT_2,
    "ytick.color": TEXT_2, "text.color": TEXT, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def pct(ax):
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


def summary_table(within, cross):
    metrics = ["balanced_acc", "macro_f1", "rest_false_activation", "move_missed_as_rest"]
    g = within.groupby(["exercise", "model", "class_weight"])
    table = g[metrics].agg(["mean", "std"])
    table.columns = [f"{m}_{s}" for m, s in table.columns]
    table["ms_per_window"] = g["ms_per_window"].median()
    table["n_subjects"] = g.size()
    if cross is not None:
        c = cross.set_index(["exercise", "model", "class_weight"])
        table["cross_subject_balanced_acc"] = c["balanced_acc"]
        table["cross_subject_rest_false_activation"] = c["rest_false_activation"]
    table = table.reset_index()
    table.to_csv(RES / "summary_table.csv", index=False)

    lines = ["| Exercise | Model | class_weight | Balanced acc | Macro F1 | Rest false-act | Move->rest | ms/window | Cross-subject bal acc |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        cross_txt = f"{r['cross_subject_balanced_acc']:.1%}" if "cross_subject_balanced_acc" in r and pd.notna(r.get("cross_subject_balanced_acc")) else "-"
        lines.append(
            f"| E{r.exercise} | {NAMES[r.model]} | {r.class_weight} "
            f"| {r.balanced_acc_mean:.1%} ± {r.balanced_acc_std:.1%} | {r.macro_f1_mean:.1%} ± {r.macro_f1_std:.1%} "
            f"| {r.rest_false_activation_mean:.1%} | {r.move_missed_as_rest_mean:.1%} "
            f"| {r.ms_per_window:.2f} | {cross_txt} |")
    (RES / "summary_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def fig1_within(within, cross, cw):
    d = within[within.class_weight == cw]
    exercises = sorted(d.exercise.unique())
    models = [m for m in ["logreg", "svm"] if m in d.model.unique()]
    width, gap = 0.36, 0.04
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, m in enumerate(models):
        offset = (i - (len(models) - 1) / 2) * (width + gap)
        for j, e in enumerate(exercises):
            vals = d[(d.model == m) & (d.exercise == e)].balanced_acc
            x = j + offset
            ax.bar(x, vals.mean(), width, color=COLORS[m], label=NAMES[m] if j == 0 else None)
            ax.errorbar(x, vals.mean(), yerr=vals.std(), color=TEXT_2, capsize=3, lw=1)
            ax.scatter(np.full(len(vals), x), vals, s=12, color="white", edgecolor=TEXT, lw=0.6, zorder=3)
            ax.text(x, max(vals.max(), vals.mean() + vals.std()) + 0.02, f"{vals.mean():.0%}",
                    ha="center", fontsize=9)
            if cross is not None:
                c = cross[(cross.model == m) & (cross.exercise == e) & (cross.class_weight == cw)]
                if len(c):
                    ax.scatter(x, c.balanced_acc.iloc[0], marker="D", s=42, color="white",
                               edgecolor=TEXT, lw=1.4, zorder=4,
                               label="Cross-subject (train S1-S4, test S5)" if (i == 0 and j == 0) else None)
    ax.set_xticks(range(len(exercises)), [f"E{e}" for e in exercises])
    ax.set_ylabel("Balanced accuracy (test reps 3, 5)")
    ax.set_ylim(0, 1.05)
    pct(ax)
    ax.set_title(f"Within-subject balanced accuracy, mean ± std over S1-S5 (dots = subjects)\nclass_weight = {cw}",
                 loc="left", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8.5)
    fig.savefig(FIG / "fig1_within_subject.png")
    plt.close(fig)


def fig2_channels(greedy, subsets, svm, within, exercise, cw):
    g = greedy.groupby("n_electrodes")["balanced_acc"].agg(["mean", "min", "max"])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.fill_between(g.index, g["min"], g["max"], color=COLORS["logreg"], alpha=0.15, lw=0)
    ax.plot(g.index, g["mean"], color=COLORS["logreg"], lw=2, marker="o", ms=6,
            label=f"{NAMES['logreg']}: best set of k (greedy), band = range over subjects")
    rings = subsets[subsets.kind == "ring4"].groupby("electrodes")["balanced_acc"].mean()
    for n, (el, v) in enumerate(rings.items()):
        ax.scatter(4, v, marker="s", s=30, color="white", edgecolor=COLORS["logreg"], lw=1.5, zorder=3,
                   label="Logistic regression: evenly spaced ring of 4" if n == 0 else None)
        ax.annotate(f"electrodes {el}: {v:.0%}", (4, v), xytext=(14, -14 - 12 * n),
                    textcoords="offset points", fontsize=8, color=TEXT_2,
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8))
    if svm is not None:
        pts = {2: svm[svm.which == "general_pair"].svm_balanced_acc.mean(),
               4: svm[svm.which == "greedy_best4"].svm_balanced_acc.mean()}
        if within is not None:
            w = within[(within.model == "svm") & (within.exercise == exercise) & (within.class_weight == cw)]
            if len(w):
                pts[8] = w.balanced_acc.mean()
        ax.plot(list(pts), list(pts.values()), color=COLORS["svm"], lw=0, marker="D", ms=7,
                label=f"{NAMES['svm']}: chosen pair / best 4 / all 8")
    ax.axvline(2, color=TEXT_2, lw=1, ls="--")
    ax.text(2.08, 1.0, "real TBP hand: 2 electrodes", fontsize=8, color=TEXT_2, va="top")
    for k, v in g["mean"].items():
        if k in (1, 2, 4, 8):
            ax.annotate(f"{v:.0%}", (k, v), xytext=(-8, 8), textcoords="offset points", ha="right", fontsize=9)
    ax.set_xticks(range(1, 9))
    ax.set_xlabel("Number of electrodes used")
    ax.set_ylabel("Balanced accuracy (test reps 3, 5)")
    ax.set_ylim(0, 1.05)
    pct(ax)
    ax.set_title(f"How accuracy falls with fewer electrodes (E{exercise}, mean of S1-S5)", loc="left", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1, fontsize=8)
    fig.savefig(FIG / "fig2_channel_count.png")
    plt.close(fig)


def fig3_pairs(subsets, exercise):
    m = np.full((8, 8), np.nan)
    for _, r in subsets[subsets.kind.isin(["pair", "single"])].groupby("electrodes")["balanced_acc"].mean().reset_index().iterrows():
        idx = [int(c) - 1 for c in r.electrodes.split("+")]
        i, j = (idx[0], idx[0]) if len(idx) == 1 else (max(idx), min(idx))
        m[i, j] = r.balanced_acc
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ_BLUE)
    fig, ax = plt.subplots(figsize=(6, 5.2))
    ax.grid(False)
    im = ax.imshow(m, cmap=cmap, vmin=np.nanmin(m), vmax=np.nanmax(m))
    best = np.nanmax(np.where(np.eye(8, dtype=bool), np.nan, m))
    pairs_cv = subsets[subsets.kind == "pair"].groupby("electrodes")["cv_balanced_acc"].mean()
    chosen = [int(c) - 1 for c in pairs_cv.idxmax().split("+")]
    ci, cj = max(chosen), min(chosen)
    for i in range(8):
        for j in range(i + 1):
            if np.isnan(m[i, j]):
                continue
            light = m[i, j] > np.nanmin(m) + 0.6 * (np.nanmax(m) - np.nanmin(m))
            ax.text(j, i, f"{m[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if light else TEXT, fontweight="bold" if (i, j) == (ci, cj) else None)
            if (i, j) == (ci, cj):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=TEXT, lw=1.5))
    ax.set_xticks(range(8), range(1, 9))
    ax.set_yticks(range(8), range(1, 9))
    ax.set_xlabel("Electrode")
    ax.set_ylabel("Electrode")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    cb.outline.set_visible(False)
    ax.set_title(f"Logistic regression balanced accuracy for each electrode pair\n"
                 f"(E{exercise}, mean of S1-S5; diagonal = one electrode alone;\nbox = pair chosen by cross-validation on training reps)",
                 loc="left", fontsize=10)
    fig.savefig(FIG / "fig3_pair_heatmap.png")
    plt.close(fig)


def fig4_class_weight(within):
    metrics = [("balanced_acc", "Balanced accuracy\n(higher = better)"),
               ("rest_false_activation", "Rest read as a movement\n(hand moves by itself)"),
               ("move_missed_as_rest", "Movement read as rest\n(hand ignores you)")]
    models = [m for m in ["logreg", "svm"] if m in within.model.unique()]
    cws = ["none", "balanced"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    width, gap = 0.36, 0.04
    for ax, (col, title) in zip(axes, metrics):
        for i, m in enumerate(models):
            offset = (i - (len(models) - 1) / 2) * (width + gap)
            for j, cw in enumerate(cws):
                v = within[(within.model == m) & (within.class_weight == cw)][col].mean()
                ax.bar(j + offset, v, width, color=COLORS[m], label=NAMES[m] if (j == 0 and col == "balanced_acc") else None)
                ax.annotate(f"{v:.1%}" if v < 0.1 else f"{v:.0%}", (j + offset, v), xytext=(0, 3),
                            textcoords="offset points", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(cws)), [f"class_weight\n{c}" for c in cws], fontsize=8.5)
        ax.set_title(title, fontsize=9.5, loc="left")
        pct(ax)
        ax.set_ylim(0, max(ax.get_ylim()[1], 0.1) * 1.08)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("What class_weight='balanced' trades (all subjects and exercises, mean)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_class_weight.png")
    plt.close(fig)


def read(name):
    p = RES / name
    return pd.read_csv(p) if p.exists() else None


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--class-weight", choices=["none", "balanced"], default="none",
                   help="which class_weight setting figs 1 and 2 show")
    p.add_argument("--exercise", type=int, default=1, help="exercise used in channel_study.py")
    args = p.parse_args()
    FIG.mkdir(exist_ok=True)

    within, cross = read("within_subject.csv"), read("cross_subject.csv")
    tag = f"E{args.exercise}" + ("" if args.class_weight == "none" else "_balanced")
    subsets, greedy, svm = (read(f"channel_subsets_{tag}.csv"), read(f"channel_greedy_{tag}.csv"),
                            read(f"channel_svm_{tag}.csv"))
    made = []
    if within is not None:
        summary_table(within, cross); made += ["results/summary_table.md"]
        fig1_within(within, cross, args.class_weight); made.append("fig1")
        if within.class_weight.nunique() > 1:
            fig4_class_weight(within); made.append("fig4")
    if greedy is not None:
        fig2_channels(greedy, subsets, svm, within, args.exercise, args.class_weight); made.append("fig2")
        fig3_pairs(subsets, args.exercise); made.append("fig3")
    print("\nmade:", made or "nothing yet (run run_all.py / channel_study.py first)")
