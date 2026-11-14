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
   `X` holds these feature rows; `y` holds their rest or movement labels.
3. **Split:** repetitions 1, 3, 4 and 6 form the development set;
   repetitions 2 and 5 remain held out for final testing. All overlapping
   windows from one repetition stay together. This commonly used NinaPro
   benchmark split supports comparison with prior work; it is not claimed to
   be the uniquely best split.
4. **Tune and train:** `LeaveOneGroupOut` validates on each complete development
   repetition in turn. `GridSearchCV` compares three `C` values and three
   `gamma` values using mean balanced accuracy. `StandardScaler` and `SVC`
   remain in one pipeline, so scaling is learned only from each fold's training
   portion. The selected pipeline is then refitted on all development windows.
5. **Evaluate:** accuracy is the fraction of test windows classified correctly.
   Balanced accuracy averages recall across movement classes, giving each
   movement equal weight. The printed report shows per-class precision,
   recall and F1. The rest false-activation rate reports how often a held-out
   rest window is incorrectly predicted as movement.

## Scope of this first experiment

- One DB2 E1 (Exercise B) file, one participant, first eight channels.
- Rest is included as class 0. Balanced class weights reduce the effect of its
  larger number of windows.
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

Next, add a saved confusion matrix and repeat the fixed procedure for additional
participants. Keep repetitions 2 and 5 out of parameter selection, and do not
choose further settings using their test results.

References: [NinaPro DB2](https://ninapro.hevs.ch/instructions/DB2.html),
[scikit-learn SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html),
[scikit-learn pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html).

## Verified first run — 14 November 2026

The local subject-1 recording produced 17,233 windows with 24 features each:
11,592 development windows and 5,641 test windows across rest plus movement
IDs 1–17. Four-fold repetition-grouped validation selected `C=10` and
`gamma="scale"`, with mean validation balanced accuracy of 69.2%. Final held-out
accuracy was 81.5%, balanced accuracy was 79.5%, and the rest false-activation
rate was 17.0%.
