"""First RBF-SVM baseline: one NinaPro DB2 Exercise B recording."""

import argparse

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
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

        # Start with movement periods only. Never mix labels or repetitions.
        if gesture == 0 or repetition == 0:
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
        raise ValueError("No complete movement windows found.")
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
        raise ValueError("Training and test sets must contain the same movement classes.")
    print(f"X: {X.shape}; y: {y.shape}; classes: {np.unique(y)}")
    print(f"Training windows: {train.sum()}; test windows: {test.sum()}")

    # 4. Learn scaling and the classifier using training data only.
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, gamma="scale"),
    )
    model.fit(X[train], y[train])

    # 5. Evaluate once on the held-out repetitions.
    predicted = model.predict(X[test])
    print(f"Accuracy: {accuracy_score(y[test], predicted):.1%}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y[test], predicted):.1%}")
    print(classification_report(y[test], predicted, digits=3, zero_division=0))


if __name__ == "__main__":
    main()
