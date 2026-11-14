"""RBF-SVM baseline with grouped tuning for NinaPro DB2 classification."""

import argparse

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from read_ninapro import load_ninapro


def make_features(recording):
    """Return one feature row, gesture ID and repetition ID per valid window."""
    emg = recording.emg[:, :8]  # First eight recorded channels.
    gestures = recording.gesture
    repetitions = recording.repetition
    window_size = round(0.200 * recording.sample_rate_hz)
    step_size = round(0.050 * recording.sample_rate_hz)
    features, labels, groups = [], [], []

    for start in range(0, len(emg) - window_size + 1, step_size):
        stop = start + window_size
        gesture = gestures[start]
        repetition = repetitions[start]

        # Label 0 is rest. Skip only samples without a usable repetition ID.
        # Never allow a window to mix gesture labels or repetitions.
        if repetition == 0:
            continue
        if not np.all(gestures[start:stop] == gesture):
            continue
        if not np.all(repetitions[start:stop] == repetition):
            continue

        window = np.asarray(emg[start:stop], dtype=np.float64)
        rms = np.sqrt(np.mean(window ** 2, axis=0))
        mav = np.mean(np.abs(window), axis=0)
        waveform_length = np.sum(np.abs(np.diff(window, axis=0)), axis=0)

        features.append(np.concatenate([rms, mav, waveform_length]))
        labels.append(gesture)
        groups.append(repetition)

    if not features:
        raise ValueError("No complete rest or movement windows found.")
    X = np.asarray(features)
    if not np.isfinite(X).all():
        raise ValueError("Features contain missing or infinite values.")
    return X, np.asarray(labels), np.asarray(groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to the extracted DB2 S1_E1_A1.mat file")
    args = parser.parse_args()

    # 1. Load the recording using your existing reader.
    recording = load_ninapro(args.path, db=2)
    if recording.exercise != 1:
        raise ValueError("This starter expects DB2 E1 (Exercise B).")
    print(f"Subject {recording.subject}; EMG shape: {recording.emg.shape}")
    print(f"Labels used: {recording.gesture_field}")
    if recording.gesture_field == "stimulus":
        print("Using cue labels: these describe prompts, not refined movement timing.")

    # 2. Turn short signal windows into 24-number feature vectors.
    X, y, repetitions = make_features(recording)

    # 3. Hold out complete repetitions, including all their overlapping windows.
    train = np.isin(repetitions, [1, 3, 4, 6])
    test = np.isin(repetitions, [2, 5])
    if not np.all(train | test) or not train.any() or not test.any():
        raise ValueError("Expected nonempty training/test sets with repetition IDs 1–6.")
    if len(np.unique(y[train])) < 2 or not np.array_equal(
        np.unique(y[train]), np.unique(y[test])
    ):
        raise ValueError("Training and test sets must contain the same classes.")
    print(f"X: {X.shape}; y: {y.shape}; classes: {np.unique(y)}")
    print(f"Training windows: {train.sum()}; test windows: {test.sum()}")
    print(
        f"Rest windows: {(y[train] == 0).sum()} train; "
        f"{(y[test] == 0).sum()} test"
    )

    X_train = X[train]
    y_train = y[train]
    training_groups = repetitions[train]
    X_test = X[test]
    y_test = y[test]
    print(f"Cross-validation repetitions: {np.unique(training_groups)}")

    # 4. Tune C and gamma using whole training repetitions as validation folds.
    model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            class_weight="balanced",
        ),
    )
    parameter_grid = {
        "svc__C": [0.1, 1, 10],
        "svc__gamma": ["scale", 0.01, 0.1],
    }
    splitter = LeaveOneGroupOut()

    for fold, (fit_indices, validation_indices) in enumerate(
        splitter.split(X_train, y_train, training_groups),
        start=1,
    ):
        fit_repetitions = np.unique(training_groups[fit_indices])
        validation_repetitions = np.unique(training_groups[validation_indices])
        print(
            f"Fold {fold}: train {fit_repetitions}; "
            f"validate {validation_repetitions}"
        )

    search = GridSearchCV(
        estimator=model,
        param_grid=parameter_grid,
        cv=splitter,
        scoring="balanced_accuracy",
        refit=True,
        error_score="raise",
    )
    search.fit(X_train, y_train, groups=training_groups)
    print(f"Best parameters: {search.best_params_}")
    print(f"Mean CV balanced accuracy: {search.best_score_:.1%}")

    # 5. Evaluate the selected model once on held-out repetitions 2 and 5.
    best_model = search.best_estimator_
    predicted = best_model.predict(X_test)
    rest_test = y_test == 0
    if not rest_test.any():
        raise ValueError("The test set contains no rest windows.")
    false_activation_rate = np.mean(predicted[rest_test] != 0)

    print(f"Accuracy: {accuracy_score(y_test, predicted):.1%}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_test, predicted):.1%}")
    print(f"Rest false-activation rate: {false_activation_rate:.1%}")
    print(classification_report(y_test, predicted, digits=3, zero_division=0))


if __name__ == "__main__":
    main()
