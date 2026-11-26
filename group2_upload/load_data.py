# load_data.py
"""Group 2 (EMG classification): load, split and channel-select NinaPro DB2 feature CSVs.

CSV layout: columns 0..39 = 5 features x 8 electrodes, feature-major
(MAV ch1..8, RMS ch1..8, WL ch1..8, ZC ch1..8, SSC ch1..8), then label, repetition.
"""
from pathlib import Path

import numpy as np
import pandas as pd

N_CHANNELS = 8   # electrodes in the ring
N_FEATURES = 5   # MAV, RMS, WL, ZC, SSC
FEATURE_NAMES = ["MAV", "RMS", "WL", "ZC", "SSC"]


def load_csv(subject, exercise, data_dir=None):
    """Load one NinaPro DB2 feature CSV. Returns X (n, 40), y (n,), rep (n,)."""
    if data_dir is None:
        data_dir = Path(__file__).parent / "data"
    path = Path(data_dir) / f"S{subject}_E{exercise}_A1_features.csv"
    df = pd.read_csv(path)

    # Guard: the 40 feature columns must be named 0..39 in order (feature-major layout)
    n_cols = N_CHANNELS * N_FEATURES
    assert list(df.columns[:n_cols]) == [str(i) for i in range(n_cols)], "unexpected column order"

    X = df.iloc[:, :n_cols].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    rep = df["repetition"].to_numpy(dtype=int)
    return X, y, rep


def split_by_rep(X, y, rep, train=(1, 2, 4, 6), test=(3, 5)):
    """Split windows by repetition. Returns X_train, X_test, y_train, y_test, groups.

    Rows whose rep is in neither list (e.g. rep 0) are dropped.
    groups = rep of each training row, used later for grouped cross-validation.
    """
    assert not set(train) & set(test), "a rep cannot be in both train and test"

    train_mask = np.isin(rep, train)   # True where this row's rep is in train
    test_mask = np.isin(rep, test)     # True where this row's rep is in test

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    groups = rep[train_mask]
    return X_train, X_test, y_train, y_test, groups


def channel_columns(channels):
    """Column indices for the given electrodes (0-based: electrode 1 = channel 0).

    Channel c lives in columns c, c+8, c+16, c+24, c+32. Output keeps feature-major order.
    """
    channels = list(channels)
    assert all(0 <= c < N_CHANNELS for c in channels), "channels must be 0..7"
    assert len(set(channels)) == len(channels), "duplicate channel"
    return [f * N_CHANNELS + c for f in range(N_FEATURES) for c in channels]


def select_channels(X, channels):
    """Keep only the 5 feature columns of each chosen electrode. Returns (n, 5 * len(channels))."""
    return X[:, channel_columns(channels)]


if __name__ == "__main__":
    from sklearn.preprocessing import StandardScaler

    X, y, rep = load_csv(subject=1, exercise=1)

    # --- Step 2a: split by repetition ---
    print("rest vs movement, by rep:")
    print(pd.crosstab(np.where(y == 0, "rest", "move"), rep))

    X_train, X_test, y_train, y_test, groups = split_by_rep(X, y, rep)

    print("\ntrain:", X_train.shape, " test:", X_test.shape)
    print("rows in neither (dropped):", len(y) - len(y_train) - len(y_test))
    print("train reps:", np.unique(groups))
    print("rest fraction  train:", round((y_train == 0).mean(), 3),
          " test:", round((y_test == 0).mean(), 3))
    print("labels missing from train:", set(np.unique(y)) - set(np.unique(y_train)))
    print("labels missing from test: ", set(np.unique(y)) - set(np.unique(y_test)))

    # --- Step 2b: scaling (fit on train only) ---
    raw_std = X_train.std(axis=0)
    print("\nraw spread per column (std): smallest", f"{raw_std.min():.3g}",
          " largest", f"{raw_std.max():.3g}")
    print("raw std per feature type (averaged over electrodes):")
    for f, name in enumerate(FEATURE_NAMES):
        print(f"  {name:>3}: {raw_std[f * N_CHANNELS:(f + 1) * N_CHANNELS].mean():.3g}")

    scaler = StandardScaler().fit(X_train)    # learns mean and std from TRAIN only
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)       # test uses the train mean/std

    print("scaled train: |mean| max", round(np.abs(X_train_s.mean(axis=0)).max(), 3),
          " std range", round(X_train_s.std(axis=0).min(), 3), "to",
          round(X_train_s.std(axis=0).max(), 3))
    print("scaled test:  |mean| max", round(np.abs(X_test_s.mean(axis=0)).max(), 3),
          " std range", round(X_test_s.std(axis=0).min(), 3), "to",
          round(X_test_s.std(axis=0).max(), 3))

    # --- Step 2c: channel selection ---
    print("\nelectrode 1 columns:", channel_columns([0]))
    print("electrodes 1 + 5 columns:", channel_columns([0, 4]))
    for chans in ([0, 1, 2, 3, 4, 5, 6, 7], [0, 2, 4, 6], [0, 4]):
        print(f"{len(chans)} channels {chans}: X_train shape {select_channels(X_train, chans).shape}")