# Group 2: margin models (RBF SVM + logistic regression)

EMG gesture classification on NinaPro DB2, Team Bath Smart Prosthetics (EMG Classification sub-team).
Code and results by Pranshu Sheel (Brev). All results below were run on 25 Nov 2026 using the team's processed DB2 feature CSVs.

## What this does

- **Benchmark:** logistic regression vs RBF SVM on S1 to S5, exercises E1, E2 and E3, each run with `class_weight` none and balanced.
- **Cross-subject test:** train on S1 to S4, test on S5.
- **Channel-count study (E1):** how accuracy changes with 8, 6, 4, 2 and 1 electrodes, which pair of electrodes works best, and whether smart placement beats an evenly spaced ring. This matters because the TBP hand will have far fewer than 8 electrodes.

Setup (same as the team V2 guide):
- Train reps 1, 2, 4, 6. Test reps 3, 5. Rep 0 dropped (0.2% of windows).
- Features: 40 columns = 5 features (MAV, RMS, WL, ZC, SSC) x 8 electrodes, StandardScaler fitted on train only.
- Hyperparameters tuned with LeaveOneGroupOut over the training reps, scored by balanced accuracy. Test reps are never used for tuning.
- SVM grid: C 0.1 / 1 / 10 / 100, gamma scale / 0.01 / 0.1. Logistic regression grid: C 0.01 to 100.
- Metrics: balanced accuracy, macro F1, rest false-activation (rest predicted as a movement), move->rest (movement predicted as rest), confusion matrix, ms per window. Plain accuracy is not reported.

## Files

| File | What it does |
|---|---|
| `load_data.py` | Loads one CSV, splits by repetition, selects electrodes |
| `evaluate.py` | All metrics, confusion matrix PNGs, time per window |
| `models.py` | Logistic regression and SVM pipelines (scaler + model), grids, tuning |
| `run_all.py` | Full benchmark: within-subject and cross-subject. Resumable |
| `channel_study.py` | Electrode-count and electrode-pair study on E1 |
| `make_figures.py` | Summary table and the 4 figures from the results CSVs |
| `results/` | Output CSVs and `summary_table.md` |
| `figures/` | `fig1` to `fig4` and `confusion_matrices/` (one per run) |

## How to run

1. Install Python 3.10+ and the packages:
   ```
   pip install -r requirements.txt
   ```
2. Make a `data/` folder next to the scripts and copy in the 15 DB2 CSVs from
   `VIP Projects/processed_data/DB2/DB2_s1 ... DB2_s5` (files named `S1_E1_A1_features.csv` etc.). Put them all in `data/` directly, not in subfolders.
3. Run in this order:
   ```
   python run_all.py --quick      # S1 E1 only, about 2 minutes. Check this works first
   python run_all.py              # everything, about 40 minutes
   python channel_study.py        # about 10 minutes
   python make_figures.py         # table + figures
   ```
   `run_all.py` saves each finished run straight away, so you can stop it with Ctrl+C and re-run to carry on.
   Other options: `--models logreg` (fast, logreg only), `--skip-cross` (within-subject only).

Electrode numbering: in results and figures electrodes are 1 to 8. In the code (`select_channels`) they are 0 to 7.

## Key results

Within-subject, mean over S1 to S5, balanced accuracy (class_weight none / balanced):

| Exercise | Logistic regression | RBF SVM | Chance |
|---|---|---|---|
| E1 (17 movements + rest) | 63.3% / 67.5% | 65.8% / 71.0% | 5.6% |
| E2 (23 grasps + rest) | 49.4% / 52.6% | 55.0% / 59.4% | 4.2% |
| E3 (9 movements + rest) | 82.6% / 84.0% | 83.1% / 86.0% | 10% |

Full table with macro F1, false activations and timing: `results/summary_table.md`.

1. **SVM beats logistic regression** in 22 of 30 cases, by 3.5 points on average. Both are fast enough for real time (SVM 0.4 to 3.2 ms per window vs a 50 ms window step).
2. **Cross-subject collapses:** training on S1 to S4 and testing on S5 gives 7 to 27%, just above chance, while S5 trained on itself gets about 71%. A real prosthesis will need per-user calibration.
3. **class_weight balanced is a trade-off:** balanced accuracy goes up 3 to 5 points, but rest false-activation rises from about 5 to 10% to 25 to 50% (the hand moves on its own), while move->rest falls from about 14 to 25% to 4 to 8%. Which is better depends on what is safer for the user.
4. **Electrode count matters a lot (E1, logistic regression):** 8 electrodes 62%, best 4 50%, best 2 33%, best 1 21%. With SVM, 2 electrodes give 35 to 38%. Every subject's best pair is different (best shared pair: electrodes 1 + 3). Smart placement of 4 beats an evenly spaced ring by only about 3 points.

## Limitations

- Only 5 subjects, all intact-limb (DB2). Not tested on amputee data (DB3).
- Offline classification of pre-computed windows, not real-time control.
- One fixed rep split. Small hyperparameter grids. Cross-subject SVM reuses the within-subject settings on a 25k-window subsample (a full grid search was too slow).
- Channel study only on E1.
- Not directly comparable with other groups unless the same split, features and metrics are used.

## Suggested next steps

- Best 2 electrodes on a reduced gesture set (4 to 6 movements), to match the real hand.
- Per-user normalisation or a short calibration session for new users.
- Test on DB3 (amputees), then real time on the TBP hand.
- Agree one shared split and metric set across all 4 groups.
