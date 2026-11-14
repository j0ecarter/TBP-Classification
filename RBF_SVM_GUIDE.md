# NinaPro RBF-SVM starter

A small, commented baseline using your existing NinaPro reader. Open
`classify_rbf.py` and follow its five numbered stages.

## Run on macOS or Linux

Download and extract the recording into `data/DB2/` as described in
[README.md](README.md). Open a terminal in the project folder, then run:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python classify_rbf.py data/DB2/S1_E1_A1.mat
```

You can also pass a path to the recording wherever it is stored on your computer.
This script prints results without modifying the data or saving a trained model.
It uses the existing `read_ninapro.py` in this project.

## What each stage does

1. **Load:** `recording.emg` contains samples in rows and channels in columns.
   `recording.gesture` and `recording.repetition` give the aligned labels.
2. **Extract features:** take 200 ms windows, advancing 50 ms at a time.
   At 2,000 Hz, that is 400 samples per window and 100 samples per step.
   For each of the first eight channels, calculate RMS (signal magnitude),
   MAV (mean absolute magnitude), and waveform length (total absolute change
   between neighbouring samples). Each window becomes 24 numbers: eight RMS
   values, eight MAV values, then eight waveform lengths.
   `X` holds these feature rows; `y` holds their movement labels.
3. **Split:** repetitions 1, 3, 4 and 6 train; repetitions 2 and 5 test.
   All overlapping windows from one repetition stay together. Randomly
   splitting windows could put nearly identical signals into both sets.
4. **Train:** `StandardScaler` learns each feature's mean and scale from
   training data. `SVC` then learns to separate the movements with an RBF kernel.
   The pipeline automatically applies the same learned scaling at prediction.
   `C=1.0` and `gamma="scale"` are starting settings, not tuned parameters.
5. **Evaluate:** accuracy is the fraction of test windows classified correctly.
   Balanced accuracy averages recall across movement classes, giving each
   movement equal weight. The printed report shows per-class precision,
   recall and F1; support is the number of windows, not independent trials.

## Scope of this first experiment

- One DB2 E1 (Exercise B) file, one participant, first eight channels.
- Movement-only classification: rest (label 0) is excluded. The classifier
  cannot decide whether someone is resting.
- Windows crossing a gesture or repetition boundary are excluded. This does
  not evaluate transitions or continuous movement detection.
- No additional signal filtering is applied: features use the stored EMG.
- The reader prefers complete refined labels. This run uses `restimulus` and
  `rerepetition`. If those arrays are incomplete in another file, the reader
  falls back to cue labels (`stimulus`/`repetition`), which describe the
  requested action rather than refined movement timing. The script prints
  which labels it uses.
- The score describes held-out repetitions from the same recording, not
  generalisation to a new person, day or device. Adjacent test windows overlap.

After understanding this baseline, add a confusion matrix and a rest class
with an explicit rest-splitting policy. Tune C and gamma only using grouped
validation inside the training repetitions; reserve the test repetitions for
final evaluation. Do not repeatedly choose settings using their test score.

References: [NinaPro DB2](https://ninapro.hevs.ch/instructions/DB2.html),
[scikit-learn SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html),
[scikit-learn pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html).

## Verified first run — 14 November 2026

The local subject-1 recording produced 7,028 windows with 24 features each:
4,707 training windows and 2,321 test windows across movement IDs 1–17.
Accuracy was 70.5%; balanced accuracy was 69.0%. These are results of this
fixed baseline, with no parameter search.
