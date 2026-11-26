# channel_study.py
"""Group 2 original extension: how much accuracy is lost with fewer electrodes?

The real TBP hand has only 2 EMG channels, so: which 2 of the 8 ring electrodes work best,
and how does accuracy fall from 8 -> 4 -> 2 -> 1?

Rule: electrode sets are CHOSEN using cross-validation on training reps only
(LeaveOneGroupOut over reps 1,2,4,6). Test reps 3,5 are only used to report the result,
never to pick the winner, otherwise the "best pair" would be flattered by luck.

Logistic regression (C=1, the value tuned on all 8 channels) does the big search because it's fast;
the RBF SVM then confirms the winning sets.

Outputs (results/):
  channel_subsets.csv  every single electrode, all 28 pairs, two 4-electrode rings, all 8 (logreg)
  channel_greedy.csv   greedy forward selection 1..8 electrodes (logreg)
  channel_svm.csv      SVM on the chosen pair / best 4 vs logreg

Usage:
  python channel_study.py                # E1, S1..S5
  python channel_study.py --exercise 2
  python channel_study.py --subjects 1   # quick test
"""
import argparse
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score

from evaluate import evaluate
from load_data import N_CHANNELS, load_csv, select_channels, split_by_rep
from models import make_logreg, make_svm, tune

RESULTS_DIR = Path(__file__).parent / "results"
SVM_PAIR_GRID = {"clf__C": [1, 10, 100], "clf__gamma": ["scale"]}   # small grid: SVM is slow
RINGS_4 = [(0, 2, 4, 6), (1, 3, 5, 7)]   # every other electrode, evenly spaced round the forearm


def label(channels):
    """Electrode numbers as people say them (1..8), e.g. (0, 4) -> '1+5'."""
    return "+".join(str(c + 1) for c in sorted(channels))


class SubsetScorer:
    """Scores electrode subsets for one subject, caching so nothing is fitted twice."""

    def __init__(self, X_train, y_train, X_test, y_test, groups, class_weight):
        self.data = (X_train, y_train, X_test, y_test, groups)
        self.cw = class_weight
        self.cache = {}

    def cv(self, channels):
        """Mean balanced accuracy over LeaveOneGroupOut folds (training reps only)."""
        key = tuple(sorted(channels))
        if key not in self.cache:
            X_train, y_train, _, _, groups = self.data
            scores = cross_val_score(make_logreg(class_weight=self.cw),
                                     select_channels(X_train, key), y_train, groups=groups,
                                     cv=LeaveOneGroupOut(), scoring="balanced_accuracy", n_jobs=-1)
            self.cache[key] = scores.mean()
        return self.cache[key]

    def test(self, channels):
        """Fit on all training reps, score on test reps 3,5."""
        X_train, y_train, X_test, y_test, _ = self.data
        key = sorted(channels)
        model = make_logreg(class_weight=self.cw).fit(select_channels(X_train, key), y_train)
        return evaluate(model, select_channels(X_test, key), y_test, fig_dir=None)


def greedy_forward(scorer):
    """Start empty; each step add the electrode that most improves CV score. Returns order added."""
    chosen, remaining, steps = [], list(range(N_CHANNELS)), []
    while remaining:
        best = max(remaining, key=lambda c: scorer.cv(chosen + [c]))
        chosen.append(best)
        remaining.remove(best)
        steps.append((list(chosen), scorer.cv(chosen)))
    return steps


def run(subjects, exercise, cw_name):
    cw = None if cw_name == "none" else "balanced"
    subset_rows, greedy_rows, scorers, splits = [], [], {}, {}

    # --- 1) logreg on every subset, per subject ---
    for s in subjects:
        t0 = time.perf_counter()
        X, y, rep = load_csv(s, exercise)
        X_train, X_test, y_train, y_test, groups = split_by_rep(X, y, rep)
        splits[s] = (X_train, X_test, y_train, y_test, groups)
        sc = scorers[s] = SubsetScorer(X_train, y_train, X_test, y_test, groups, cw)

        subsets = ([("single", (c,)) for c in range(N_CHANNELS)]
                   + [("pair", p) for p in combinations(range(N_CHANNELS), 2)]
                   + [("ring4", r) for r in RINGS_4]
                   + [("all8", tuple(range(N_CHANNELS)))])
        for kind, chans in subsets:
            r = sc.test(chans)
            subset_rows.append({"subject": s, "exercise": exercise, "class_weight": cw_name,
                                "kind": kind, "n_electrodes": len(chans), "electrodes": label(chans),
                                "cv_balanced_acc": sc.cv(chans), **r})

        for k, (chans, cv) in enumerate(greedy_forward(sc), start=1):
            r = sc.test(chans)
            greedy_rows.append({"subject": s, "exercise": exercise, "class_weight": cw_name,
                                "n_electrodes": k, "electrodes": label(chans),
                                "added": chans[-1] + 1, "cv_balanced_acc": cv, **r})
        best_pair = max(combinations(range(N_CHANNELS), 2), key=sc.cv)
        print(f"S{s} E{exercise}: best pair by CV {label(best_pair)} "
              f"(CV {sc.cv(best_pair):.3f}); greedy order "
              f"{[c + 1 for c in greedy_forward(sc)[-1][0]]}  ({time.perf_counter() - t0:.0f} s)")

    subsets_df = pd.DataFrame(subset_rows)
    greedy_df = pd.DataFrame(greedy_rows)

    # One pair for everyone (what you'd wire into the real hand): best MEAN CV across subjects
    pairs = subsets_df[subsets_df.kind == "pair"]
    general = pairs.groupby("electrodes")["cv_balanced_acc"].mean().idxmax()
    general_pair = tuple(int(c) - 1 for c in general.split("+"))
    print(f"\nbest pair across subjects (mean CV): electrodes {general}")

    # --- 2) SVM confirms the chosen sets ---
    svm_rows = []
    for s in subjects:
        X_train, X_test, y_train, y_test, groups = splits[s]
        sc = scorers[s]
        own_pair = max(combinations(range(N_CHANNELS), 2), key=sc.cv)
        best4 = tuple(greedy_df[(greedy_df.subject == s) & (greedy_df.n_electrodes == 4)]
                      .electrodes.iloc[0].split("+"))
        best4 = tuple(int(c) - 1 for c in best4)
        for which, chans in [("general_pair", general_pair), ("own_best_pair", own_pair),
                             ("greedy_best4", best4)]:
            t0 = time.perf_counter()
            search = tune(make_svm(class_weight=cw), SVM_PAIR_GRID,
                          select_channels(X_train, chans), y_train, groups)
            r = evaluate(search.best_estimator_, select_channels(X_test, chans), y_test, fig_dir=None)
            lr = sc.test(chans)
            svm_rows.append({"subject": s, "exercise": exercise, "class_weight": cw_name,
                             "which": which, "n_electrodes": len(chans), "electrodes": label(chans),
                             "svm_C": search.best_params_["clf__C"],
                             "svm_cv_balanced_acc": search.best_score_,
                             **{f"svm_{k}": v for k, v in r.items()},
                             "logreg_balanced_acc": lr["balanced_acc"],
                             "logreg_rest_false_activation": lr["rest_false_activation"]})
            print(f"S{s} {which:>13} {label(chans):>8}: SVM bal acc {r['balanced_acc']:.3f}  "
                  f"logreg {lr['balanced_acc']:.3f}  ({time.perf_counter() - t0:.0f} s)")

    RESULTS_DIR.mkdir(exist_ok=True)
    tag = f"E{exercise}" + ("" if cw_name == "none" else "_balanced")
    subsets_df.to_csv(RESULTS_DIR / f"channel_subsets_{tag}.csv", index=False)
    greedy_df.to_csv(RESULTS_DIR / f"channel_greedy_{tag}.csv", index=False)
    pd.DataFrame(svm_rows).to_csv(RESULTS_DIR / f"channel_svm_{tag}.csv", index=False)

    # --- summary ---
    print("\nmean test balanced acc across subjects (logreg):")
    print(greedy_df.groupby("n_electrodes")["balanced_acc"].agg(["mean", "std"]).round(3)
          .rename(index=lambda k: f"best {k} (greedy)"))
    for kind in ["single", "pair", "ring4", "all8"]:
        d = subsets_df[subsets_df.kind == kind]["balanced_acc"]
        print(f"  {kind:>6}: mean {d.mean():.3f}  range {d.min():.3f} to {d.max():.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exercise", type=int, default=1)
    p.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    args = p.parse_args()
    t = time.perf_counter()
    run(args.subjects, args.exercise, args.class_weight)
    print(f"\ndone in {(time.perf_counter() - t) / 60:.1f} min")
