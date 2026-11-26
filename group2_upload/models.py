# models.py
"""Group 2 (EMG classification): model builders and grouped-CV tuning.

Every model is a Pipeline(StandardScaler, classifier), so inside cross-validation the
scaler is refitted on the training part of each fold and never sees the held-out rep.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Logistic regression: C = how much to trust the training data (small C = cautious)
LOGREG_GRID = {"clf__C": [0.01, 0.1, 1, 10, 100]}

# RBF SVM: Joe's grid from classify_rbf.py (C 0.1/1/10, gamma scale/0.01/0.1) so results
# are comparable, plus C=100 because his best C (10) sat at the top edge of his grid.
# gamma = how far each training window's influence reaches (big gamma = very local, wiggly border).
# "scale" = 1 / (n_features * variance) = 1/40 = 0.025 after scaling with all 8 channels.
SVM_GRID = {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.1]}


def make_logreg(class_weight=None, C=1.0):
    """StandardScaler + multinomial logistic regression (flat borders between gestures).

    class_weight=None treats every window equally; "balanced" up-weights rare gestures
    so the 59% rest windows don't dominate training.
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=C, class_weight=class_weight, max_iter=5000)),
    ])


def make_svm(class_weight=None, C=10.0, gamma="scale"):
    """StandardScaler + RBF-kernel SVM (curved borders between gestures).

    Adapted from Joe's classify_rbf.py. cache_size (MB) only speeds up training.
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=C, gamma=gamma, class_weight=class_weight,
                    cache_size=1000)),
    ])


MODELS = {
    "logreg": (make_logreg, LOGREG_GRID),
    "svm": (make_svm, SVM_GRID),
}


def tune(model, grid, X, y, groups, n_jobs=-1):
    """Grid search with LeaveOneGroupOut, scored by balanced accuracy.

    Within-subject: groups = rep of each training row (1, 2, 4, 6), so each fold holds
    out one whole rep. Cross-subject: groups = subject, so each fold holds out one person.
    Test data is never touched. Returns the fitted GridSearchCV; use .best_estimator_
    (refitted on all training data), .best_params_, .best_score_, .cv_results_.
    """
    search = GridSearchCV(
        model, grid,
        cv=LeaveOneGroupOut(),
        scoring="balanced_accuracy",
        n_jobs=n_jobs,
        refit=True,
    )
    search.fit(X, y, groups=groups)
    return search
