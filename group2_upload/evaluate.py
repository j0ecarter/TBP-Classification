# evaluate.py
"""Group 2 (EMG classification): score a fitted model on the test reps.

Metrics (team V2 rules: plain accuracy is not reported):
  balanced_acc          mean of per-gesture recall
  macro_f1              mean of per-gesture F1
  rest_false_activation fraction of true-rest windows predicted as a movement (hand moves on its own)
  move_missed_as_rest   fraction of true-movement windows predicted as rest (hand ignores you)
  ms_per_window         median time to classify ONE window (what the real-time hand would see)
  ms_per_window_batch   time per window when predicting the whole test set at once
"""
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save PNGs without opening a window
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, balanced_accuracy_score,
                             confusion_matrix, f1_score)

REST = 0
FIG_DIR = Path(__file__).parent / "figures"


def time_per_window(model, X, n=200, seed=0):
    """Median ms to predict a single window, over n randomly chosen test windows."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    model.predict(X[:1])  # warm-up call, not timed
    times = []
    for i in idx:
        t0 = time.perf_counter()
        model.predict(X[i:i + 1])
        times.append(time.perf_counter() - t0)
    return 1000 * float(np.median(times))


def save_confusion_matrix(y_true, y_pred, path, title=""):
    """Row-normalised confusion matrix: row = true gesture, each row sums to 1.

    The diagonal is each gesture's recall; column 0 off the diagonal = movements misread as rest.
    """
    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    size = max(6, 0.45 * len(labels))
    fig, ax = plt.subplots(figsize=(size, size))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, cmap="Blues", values_format=".2f", colorbar=False, text_kw={"fontsize": 6})
    ax.set_title(title)
    ax.set_xlabel("predicted label (0 = rest)")
    ax.set_ylabel("true label (0 = rest)")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def evaluate(model, X_test, y_test, name="model", fig_dir=FIG_DIR):
    """Score a fitted model. Returns a dict of metrics; saves a confusion matrix PNG if fig_dir is set."""
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    batch_ms = 1000 * (time.perf_counter() - t0) / len(X_test)

    is_rest = y_test == REST
    results = {
        "balanced_acc": balanced_accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "rest_false_activation": float(np.mean(y_pred[is_rest] != REST)),
        "move_missed_as_rest": float(np.mean(y_pred[~is_rest] == REST)),
        "ms_per_window": time_per_window(model, X_test),
        "ms_per_window_batch": batch_ms,
    }
    if fig_dir is not None:
        b, f = results["balanced_acc"], results["macro_f1"]
        save_confusion_matrix(y_test, y_pred, Path(fig_dir) / f"cm_{name}.png",
                              title=f"{name}  (balanced acc {b:.3f}, macro F1 {f:.3f})")
    return results


def print_results(name, r):
    print(f"{name:>16}: bal acc {r['balanced_acc']:.3f}  macro F1 {r['macro_f1']:.3f}  "
          f"rest false-act {r['rest_false_activation']:.3f}  move->rest {r['move_missed_as_rest']:.3f}  "
          f"{r['ms_per_window']:.3f} ms/window (batch {r['ms_per_window_batch']:.4f})")


if __name__ == "__main__":
    from load_data import load_csv, split_by_rep
    from models import LOGREG_GRID, make_logreg, tune

    X, y, rep = load_csv(subject=1, exercise=1)
    X_train, X_test, y_train, y_test, groups = split_by_rep(X, y, rep)

    # 1) Default logistic regression (C = 1)
    default = make_logreg().fit(X_train, y_train)
    print_results("logreg C=1", evaluate(default, X_test, y_test, name="S1_E1_logreg_default"))

    # 2) Tune C with LeaveOneGroupOut over training reps 1, 2, 4, 6 (test reps untouched)
    print("\ntuning C (4 folds x 5 values)...")
    search = tune(make_logreg(), LOGREG_GRID, X_train, y_train, groups)
    for C, mean, std in zip(search.cv_results_["param_clf__C"],
                            search.cv_results_["mean_test_score"],
                            search.cv_results_["std_test_score"]):
        print(f"  C={C:<6} CV balanced acc {mean:.3f} +/- {std:.3f}")
    best_C = search.best_params_["clf__C"]
    print("best C:", best_C)

    tuned = search.best_estimator_  # already refitted on all training reps
    print_results(f"logreg C={best_C}",
                  evaluate(tuned, X_test, y_test, name="S1_E1_logreg_tuned"))
    print(f"\nconfusion matrices saved in {FIG_DIR}")