# run_all.py
"""Group 2 (EMG classification): full benchmark.

1. Within-subject: S1..S5 x E1/E2/E3 x {logreg, svm} x class_weight {none, balanced}.
   Train reps 1,2,4,6 (tuned with LeaveOneGroupOut over reps), test reps 3,5.
   -> results/within_subject.csv
2. Cross-subject: train on S1..S4, test on S5, per exercise.
   -> results/cross_subject.csv

Resumable: each finished run is appended to the CSV straight away, and runs already in
the CSV are skipped. Stop with Ctrl+C at any time and re-run to carry on.

Usage:
  python run_all.py                 # everything
  python run_all.py --quick         # S1 E1 only (smoke test, a few minutes)
  python run_all.py --models logreg # logreg only (fast)
  python run_all.py --skip-cross    # within-subject only
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import evaluate, print_results
from load_data import load_csv, split_by_rep
from models import MODELS, make_svm, tune

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
CM_DIR = ROOT / "figures" / "confusion_matrices"
SUBJECTS = [1, 2, 3, 4, 5]
EXERCISES = [1, 2, 3]
CLASS_WEIGHTS = {"none": None, "balanced": "balanced"}
ALL_REPS = (1, 2, 3, 4, 5, 6)   # rep 0 dropped, as in the within-subject split
SVM_CROSS_MAX_TRAIN = 25000     # SVM training time grows ~ n^2, so cross-subject SVM uses a subsample


def already_done(csv_path, **key):
    """True if a row matching every key=value is already in the CSV."""
    if not csv_path.exists():
        return False
    df = pd.read_csv(csv_path)
    mask = np.ones(len(df), dtype=bool)
    for k, v in key.items():
        mask &= df[k].astype(str) == str(v)
    return bool(mask.any())


def append_row(csv_path, row):
    """Append one result row to the CSV (writes the header the first time)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


def within_subject(subjects, exercises, models, out_csv):
    for e in exercises:
        for s in subjects:
            X, y, rep = load_csv(s, e)
            X_train, X_test, y_train, y_test, groups = split_by_rep(X, y, rep)
            for model_name in models:
                make, grid = MODELS[model_name]
                for cw_name, cw in CLASS_WEIGHTS.items():
                    key = dict(subject=s, exercise=e, model=model_name, class_weight=cw_name)
                    if already_done(out_csv, **key):
                        print(f"skip (done): S{s} E{e} {model_name} {cw_name}")
                        continue
                    t0 = time.perf_counter()
                    search = tune(make(class_weight=cw), grid, X_train, y_train, groups)
                    tune_s = time.perf_counter() - t0
                    name = f"S{s}_E{e}_{model_name}_{cw_name}"
                    r = evaluate(search.best_estimator_, X_test, y_test, name=name, fig_dir=CM_DIR)
                    print_results(name, r)
                    params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
                    print(f"{'':>16}  best {params}  CV bal acc {search.best_score_:.3f}  ({tune_s:.0f} s)")
                    append_row(out_csv, {
                        **key,
                        "n_train": len(y_train), "n_test": len(y_test),
                        "n_classes": len(np.unique(y_train)),
                        "best_C": params.get("C"), "best_gamma": params.get("gamma", ""),
                        "cv_balanced_acc": search.best_score_,
                        **r,
                        "tune_seconds": tune_s,
                    })


def svm_params_from_within(within_csv, exercise, cw_name):
    """Most common best (C, gamma) for this exercise in the within-subject results.

    Cross-subject SVM is too slow to grid-search on ~70k windows, so it reuses the
    setting chosen most often by the within-subject tuning (training reps only, so no test peeking).
    """
    default = (10.0, "scale")   # Joe's choice on S1 E1
    if not within_csv.exists():
        return default
    df = pd.read_csv(within_csv)
    df = df[(df.model == "svm") & (df.exercise == exercise) & (df.class_weight == cw_name)]
    if df.empty:
        return default
    C, gamma = df.groupby(["best_C", "best_gamma"]).size().idxmax()
    gamma = gamma if gamma == "scale" else float(gamma)
    return float(C), gamma


def cross_subject(exercises, models, out_csv, within_csv, train_subjects=(1, 2, 3, 4), test_subject=5):
    for e in exercises:
        # Train on every rep of S1..S4; groups = subject, so CV holds out one whole person
        Xs, ys, gs = [], [], []
        for s in train_subjects:
            X, y, rep = load_csv(s, e)
            keep = np.isin(rep, ALL_REPS)
            Xs.append(X[keep]); ys.append(y[keep]); gs.append(np.full(keep.sum(), s))
        X_train, y_train, groups = np.vstack(Xs), np.concatenate(ys), np.concatenate(gs)

        X, y, rep = load_csv(test_subject, e)
        keep = np.isin(rep, ALL_REPS)
        X_test, y_test = X[keep], y[keep]

        for model_name in models:
            for cw_name, cw in CLASS_WEIGHTS.items():
                key = dict(exercise=e, model=model_name, class_weight=cw_name,
                           train_subjects="+".join(map(str, train_subjects)), test_subject=test_subject)
                if already_done(out_csv, **key):
                    print(f"skip (done): cross E{e} {model_name} {cw_name}")
                    continue
                t0 = time.perf_counter()
                if model_name == "logreg":
                    make, grid = MODELS["logreg"]
                    search = tune(make(class_weight=cw), grid, X_train, y_train, groups)
                    model, cv_score = search.best_estimator_, search.best_score_
                    C, gamma, n_fit = search.best_params_["clf__C"], "", len(y_train)
                else:
                    C, gamma = svm_params_from_within(within_csv, e, cw_name)
                    # Stratified subsample keeps every gesture's share the same
                    rng = np.random.default_rng(0)
                    n = min(SVM_CROSS_MAX_TRAIN, len(y_train))
                    idx = np.sort(np.concatenate([
                        rng.choice(np.flatnonzero(y_train == c),
                                   size=max(1, round(n * np.mean(y_train == c))), replace=False)
                        for c in np.unique(y_train)]))
                    model = make_svm(class_weight=cw, C=C, gamma=gamma).fit(X_train[idx], y_train[idx])
                    cv_score, n_fit = np.nan, len(idx)
                fit_s = time.perf_counter() - t0
                name = f"cross_E{e}_{model_name}_{cw_name}"
                r = evaluate(model, X_test, y_test, name=name, fig_dir=CM_DIR)
                print_results(name, r)
                print(f"{'':>16}  C={C} gamma={gamma}  trained on {n_fit} windows  ({fit_s:.0f} s)")
                append_row(out_csv, {
                    **key, "n_train": n_fit, "n_test": len(y_test),
                    "best_C": C, "best_gamma": gamma, "cv_balanced_acc": cv_score,
                    **r, "fit_seconds": fit_s,
                })


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="S1 E1 only, no cross-subject")
    p.add_argument("--models", nargs="+", default=["logreg", "svm"], choices=list(MODELS))
    p.add_argument("--skip-cross", action="store_true")
    args = p.parse_args()

    within_csv = RESULTS_DIR / ("within_subject_quick.csv" if args.quick else "within_subject.csv")
    cross_csv = RESULTS_DIR / "cross_subject.csv"
    subjects, exercises = ([1], [1]) if args.quick else (SUBJECTS, EXERCISES)

    t_start = time.perf_counter()
    print("=== within-subject ===")
    within_subject(subjects, exercises, args.models, within_csv)
    if not (args.quick or args.skip_cross):
        print("\n=== cross-subject (train S1-S4, test S5) ===")
        cross_subject(exercises, args.models, cross_csv, within_csv)
    print(f"\ndone in {(time.perf_counter() - t_start) / 60:.1f} min. Results in {RESULTS_DIR}")
