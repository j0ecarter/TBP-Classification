"""Print a small table of samples from one extracted NinaPro recording."""

import argparse
from pathlib import Path

from scipy.io.matlab import MatReadError

from read_ninapro import DB_FORMATS, load_ninapro


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to one extracted .mat file")
    parser.add_argument("--db", type=int, choices=DB_FORMATS, default=1,
                        help="NinaPro database number (default: 1)")
    parser.add_argument("--start", type=float, default=0, metavar="SECONDS",
                        help="Time to start the preview (default: 0)")
    parser.add_argument("--rows", type=int,
                        help="Maximum rows to show (default: 10, or all with --every)")
    parser.add_argument("--every", type=int, metavar="N",
                        help="Show every Nth sample starting at --start")
    args = parser.parse_args()
    if args.rows is not None and args.rows < 1:
        parser.error("--rows must be at least 1.")
    if args.every is not None and args.every < 1:
        parser.error("--every must be at least 1.")

    try:
        recording = load_ninapro(args.path, db=args.db)
    except (OSError, ValueError, MatReadError, NotImplementedError) as error:
        parser.error(f"{args.path}: {error}")

    samples, channels = recording.emg.shape
    rate = recording.sample_rate_hz
    if not 0 <= args.start < samples / rate:
        parser.error(f"--start must be at least 0 and less than {samples / rate:g} seconds.")
    start = min(round(args.start * rate), samples - 1)
    step = args.every or 1
    row_limit = args.rows
    if row_limit is None and args.every is None:
        row_limit = 10
    stop = samples if row_limit is None else min(start + row_limit * step, samples)

    print(f"NinaPro DB{recording.database} | Subject {recording.subject} | Exercise {recording.exercise}")
    print(f"{samples:,} samples, {channels} EMG channels, {rate:,} Hz")
    repetition_field = "rerepetition" if recording.gesture_field == "restimulus" else "repetition"
    print(f"MATLAB fields: emg, {recording.gesture_field}, {repetition_field}. Gesture 0 = rest.\n")

    headers = ("sample", "time_s", "gesture_id", "repetition_id",
               *(f"emg_{channel:02d}" for channel in range(1, channels + 1)))
    widths = (max(6, len(str(samples - 1))), 12, 10, 13, *(11 for _ in range(channels)))
    row_format = " ".join(f"{{:>{width}}}" for width in widths)
    print(row_format.format(*headers))
    for i in range(start, stop, step):
        print(row_format.format(
            i, f"{i / rate:.6f}", int(recording.gesture[i]), int(recording.repetition[i]),
            *(f"{value:.4g}" for value in recording.emg[i]),
        ))


if __name__ == "__main__":
    main()
