# TBP-Classification

Python tools for inspecting NinaPro surface EMG recordings and running an initial RBF-kernel SVM classifier. `read_ninapro.py` summarises one recording or a folder. `preview_ninapro_table.py` shows the channel values and gesture and repetition IDs for individual samples.

[ninapro-db2-full.png](ninapro-db2-full.png) shows channel 7 and the gesture timeline for DB2 subject 1, Exercise B.

## Get the recording

1. Download [DB2 subject 1 from NinaPro](https://ninapro.hevs.ch/files/DB2_Preproc/DB2_s1.zip).
2. Extract the archive and put `S1_E1_A1.mat` in `data/DB2/` inside this project. This is the Exercise B file used for the plot.
3. Keep recordings in your local `data/` folder. Additional subject archives are available from [NinaPro DB2](https://ninapro.hevs.ch/instructions/DB2.html), [NinaPro DB3](https://ninapro.hevs.ch/instructions/DB3.html), and the [Dryad dataset archive](https://doi.org/10.5061/dryad.1k84r).

## Run the scripts

On macOS or Linux, run these commands from the project folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python read_ninapro.py --db 2 data/DB2/S1_E1_A1.mat
python preview_ninapro_table.py --db 2 data/DB2/S1_E1_A1.mat --start 5.412 --rows 5
```

The example table shows the first change from rest (`0`) to thumbs-up (`1`). Each row contains all 12 simultaneous EMG readings, the time in seconds, and the gesture and repetition IDs. Add `--every 10000` to preview one sample every five seconds at the DB2 sampling rate of 2,000 Hz.

See the [NinaPro movement chart](https://elib.dlr.de/94692/1/article.2014.Atzori.Electromyography%20data.Nature%20Scientific%20Data.pdf#page=6) for the exercise definitions.

## Run the RBF-SVM starter

After downloading the recording and installing the requirements above:

```sh
python classify_rbf.py data/DB2/S1_E1_A1.mat
```

Open [classify_rbf.py](classify_rbf.py) to follow the five stages: load the
recording, extract window features, split repetitions, train and evaluate.
The [step-by-step guide](RBF_SVM_GUIDE.md) explains each stage and the baseline
results. The starter uses the first eight channels, 200 ms windows, RMS/MAV/
waveform-length features, training-only scaling and an RBF SVM. Repetitions
1, 3, 4 and 6 form the grouped cross-validation set; repetitions 2 and 5 are
reserved for final testing. Rest is included as class 0, and the classifier
reports its rest false-activation rate.

Recordings, virtual environments and Python caches are excluded by `.gitignore`.
Download the data separately; do not add the `.mat` recording to GitHub.
