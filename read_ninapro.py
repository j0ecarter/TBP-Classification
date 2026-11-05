"""Load NinaPro DB1–DB5 EMG from an extracted .mat file or dataset folder.

Install dependencies with `python3 -m pip install numpy scipy`.
Run `python3 read_ninapro.py ~/Downloads/s1` for DB1.
For another database, pass its number and a folder containing that database
only, for example `python3 read_ninapro.py --db 2 ~/Downloads/NinaPro/DB2`.
"""

import argparse
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.io.matlab import MatReadError


DB1_SAMPLE_RATE_HZ = 100
DB2_SAMPLE_RATE_HZ = 2000
DB3_SAMPLE_RATE_HZ = 2000
DB4_SAMPLE_RATE_HZ = 2000
DB5_SAMPLE_RATE_HZ = 200

# Stored EMG channels and sampling rates from https://ninapro.hevs.ch/
# DB1 rate: https://pmc.ncbi.nlm.nih.gov/articles/PMC4421935/
DB_FORMATS = {
    1: (10, DB1_SAMPLE_RATE_HZ),
    2: (12, DB2_SAMPLE_RATE_HZ),
    3: (12, DB3_SAMPLE_RATE_HZ),
    4: (12, DB4_SAMPLE_RATE_HZ),
    5: (16, DB5_SAMPLE_RATE_HZ),
}


@dataclass
class Recording:
    emg: np.ndarray  # Rows are samples, columns are EMG channels.
    gesture: np.ndarray  # Exercise-local gesture ID per sample (0 means rest).
    repetition: np.ndarray  # Repetition ID per sample.
    subject: int
    exercise: int
    sample_rate_hz: int = DB1_SAMPLE_RATE_HZ
    database: int = 1
    gesture_field: str = "restimulus"


def load_ninapro(path, *, db: int) -> Recording:
    """Read EMG and aligned gesture/repetition labels for the selected database."""
    if db not in DB_FORMATS:
        raise ValueError("Choose a NinaPro database from DB1 through DB5.")

    path = Path(path).expanduser()
    if path.suffix.lower() != ".mat":
        raise ValueError(f"Pass an extracted NinaPro DB{db} .mat file.")

    expected_channels, sample_rate_hz = DB_FORMATS[db]
    metadata_fields = ("emg", "subject", "exercise")
    refined_labels = ("restimulus", "rerepetition")
    cue_labels = ("stimulus", "repetition")
    fields = metadata_fields + refined_labels
    allow_cue_labels = db in (2, 3)
    if allow_cue_labels:
        fields += cue_labels

    data = loadmat(path, variable_names=fields)
    missing = [name for name in metadata_fields if name not in data]
    if missing:
        raise ValueError(f"Missing DB{db} fields: {', '.join(missing)}")

    emg = data["emg"]
    if emg.ndim != 2 or emg.shape[0] == 0 or emg.shape[1] != expected_channels:
        raise ValueError(
            f"Expected a nonempty DB{db} EMG matrix with {expected_channels} channels."
        )

    labels = refined_labels
    label_shape = (emg.shape[0], 1)
    # Some DB2/DB3 files lack a complete, aligned refined pair.
    if allow_cue_labels and not all(
        name in data and data[name].shape == label_shape for name in refined_labels
    ):
        labels = cue_labels

    for name in labels:
        if name not in data:
            raise ValueError(f"Missing DB{db} label field: {name}")
        if data[name].shape != label_shape:
            raise ValueError(f"{name} must have one entry per EMG sample.")

    gesture_field, repetition_field = labels
    return Recording(
        emg=emg,
        gesture=data[gesture_field][:, 0],
        repetition=data[repetition_field][:, 0],
        subject=int(data["subject"].item()),
        exercise=int(data["exercise"].item()),
        sample_rate_hz=sample_rate_hz,
        database=db,
        gesture_field=gesture_field,
    )


def load_db1(path) -> Recording:
    """Read one NinaPro DB1 recording."""
    return load_ninapro(path, db=1)


def iter_ninapro(path, *, db: int) -> Iterator[Recording]:
    """Yield recordings from a file or directory tree for one database."""
    path = Path(path).expanduser()
    if path.is_dir():
        files = sorted(
            file for file in path.rglob("*")
            if file.suffix.lower() == ".mat" and file.is_file()
        )
        if not files:
            raise ValueError(f"No extracted .mat files found in {path}")
    else:
        files = [path]

    for file in files:
        try:
            recording = load_ninapro(file, db=db)
        except (OSError, ValueError, MatReadError, NotImplementedError) as error:
            raise ValueError(f"{file}: {error}") from error
        yield recording


def iter_db1(path) -> Iterator[Recording]:
    """Yield NinaPro DB1 recordings from a file or directory tree."""
    yield from iter_ninapro(path, db=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", type=Path, help="Extracted .mat file or dataset folder")
    parser.add_argument("--db", type=int, choices=DB_FORMATS, default=1,
                        help="NinaPro database number (default: 1)")
    args = parser.parse_args()

    try:
        for recording in iter_ninapro(args.path, db=args.db):
            samples, channels = recording.emg.shape
            gesture_ids = ", ".join(map(str, np.unique(recording.gesture)))
            print(f"Subject {recording.subject}, exercise {recording.exercise}")
            print(f"{samples:,} samples × {channels} channels at {recording.sample_rate_hz} Hz")
            print(f"Gesture IDs: {gesture_ids} (0 = rest)")
            if recording.database != 1:
                print(f"NinaPro DB{recording.database} label field: {recording.gesture_field}")
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
