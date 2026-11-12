# Engine Remaining Useful Life (RUL) and Health-State Monitoring

## Problem description

Aircraft engines are expensive to take out of service and catastrophic if they fail in flight.
Maintenance teams want to know **how many flight cycles an engine has left** and, more simply,
**what to do about it now**. This project predicts the *remaining useful life* (RUL) of a turbofan engine from
its sensor readings and converts it into one of three health states:

| State | RUL (cycles left) | Meaning |
|---|---|---|
| `healthy` | > 100 | normal operation |
| `warning` | 51 – 100 | schedule maintenance |
| `failure` | ≤ 50 | failure or check immediately: inspect / take out of service now |

How it is used: an engine reports its sensors every flight cycle. Each reading is sent to the web service
(`POST /predict`), which answers with the predicted RUL, the health state and the probability of each state.
`stream_client.py` replays a test engine cycle by cycle to simulate this live feed.

Approach: (1) regression of RUL, evaluated with RMSE; (2) the predicted RUL is binned into the three states and
evaluated as a 3-class problem with macro F1, one-vs-rest ROC AUC and a confusion matrix.

## Data

NASA C-MAPSS turbofan degradation simulation, subset **FD001** (1 operating condition, 21 sensors, 3 operating
settings). All files are committed in `data/`:

| File | Content |
|---|---|
| `train_FD001.txt` | 100 engines run **to failure** (20,631 rows). No RUL column: RUL = last cycle − current cycle |
| `test_FD001.txt` | 100 other engines whose history **stops before failure** (13,096 rows) |
| `RUL_FD001.txt` | true RUL of each test engine after its last recorded cycle (line *i* = engine *i*) |

So the training RUL is derived from the run-to-failure data, while the test RUL comes from `RUL_FD001.txt`
(`RUL = RUL_FD001 + last cycle − cycle` for earlier cycles; see `health.load_test`).

**Split (60 / 20 / 20 by engine, same recipe as the course projects):** the 100 training engines are first split into
`full_train` (80) and `test` (20), then `full_train` into `train` (60) and `validation` (20). Engines, not rows, are
split because rows of one engine are near-duplicates and would leak. Models are tuned on train/validation, then checked with **5-fold cross-validation** on `full_train`
(folds also split by engine; 5 folds of ~16 engines, because 10 would leave only ~8 engines per fold) to get the mean and
standard deviation of RMSE, macro F1 and ROC AUC. The final model is refit on `full_train`. It is evaluated on the held-out `test` engines and, as a second fully external check, on the
100 official test engines, which are also the ones `stream_client.py` replays. Original source: NASA Prognostics
Center of Excellence data repository.

## Repository contents

| File | Purpose |
|---|---|
| `notebook.ipynb` | data prep, EDA (target, correlation, mutual information), linear regression / decision tree / random forest / XGBoost (tuned), F1 / AUC / confusion matrix |
| `train.py` | runs 5-fold CV (mean ± std), trains the final XGBoost model on full_train, evaluates on the held-out and official test engines, writes `model.bin` and `metrics.json` |
| `predict.py` | FastAPI service: `POST /predict`, `GET /health` |
| `health.py` | shared helpers: data loading, RUL → state, state probabilities |
| `stream_client.py` | streams a test engine into the API cycle by cycle |
| `pyproject.toml`, `uv.lock` | dependencies (uv) |
| `Dockerfile`, `fly.toml` | container and Fly.io deployment |

## 1. Set up the environment (pip + uv)

```bash
# install uv itself, once
pip install uv

# from the project folder: creates .venv and installs every package in pyproject.toml
# (also writes uv.lock - commit it). uv picks the Python in .python-version (3.14).
uv sync

# activate the environment
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
```

Note: the command is `uv sync` (there is no `uv install`). To run something without activating, use `uv run <command>`.
If a package has no Python 3.14 wheel yet, run `echo 3.12 > .python-version`, change the Dockerfile base image to
`python:3.12-slim-bookworm`, delete `.venv` and run `uv sync` again.

## 2. Run the project

```bash
uv run jupyter notebook notebook.ipynb     # EDA + model selection

python train.py                            # -> model.bin, metrics.json
uvicorn predict:app --host 0.0.0.0 --port 9696    # API on http://localhost:9696  (docs: /docs)

# second terminal (environment activated): stream a test engine into the API
python stream_client.py                    # default: test engine closest to failure
python stream_client.py --unit 7 --delay 0.5
```

The stream prints the predicted RUL, the true RUL (from `RUL_FD001.txt`) and the state, e.g.
`cycle 160  predicted RUL  88.4  true RUL  50  [WARN] warning   (true: failure)`.

Single request:

```bash
curl -X POST http://localhost:9696/predict -H "Content-Type: application/json" -d '{
  "unit": 1, "cycle": 150, "op1": 0.0019, "op2": -0.0003,
  "s2": 643.1, "s3": 1600.2, "s4": 1420.5, "s7": 550.9, "s8": 2388.1, "s9": 9060.0,
  "s11": 47.9, "s12": 520.0, "s13": 2388.1, "s14": 8150.0, "s15": 8.52,
  "s17": 395, "s20": 38.6, "s21": 23.2}'
```
```json
{"unit": 1, "cycle": 150, "rul_cycles": 41.3, "state": "failure",
 "state_probabilities": {"failure": 0.60, "warning": 0.38, "healthy": 0.02}}
```
(values illustrative; constant sensors have defaults, so only the varying ones are required.)

## 3. Docker

```bash
python train.py                                  # model.bin must exist before building
docker build -t rul-health-monitor .
docker run -it --rm -p 9696:9696 rul-health-monitor
python stream_client.py                          # talks to http://localhost:9696/predict
```

The image installs the dependencies with uv (`uv sync`), copies `predict.py`, `health.py` and `model.bin`, and
starts `uvicorn` on port 9696.

## 4. Cloud deployment (Fly.io)

```bash
fly auth login
fly launch --no-deploy --copy-config --name <your-unique-app-name>   # uses fly.toml + Dockerfile
fly deploy
python stream_client.py --url https://<your-unique-app-name>.fly.dev/predict
```
I don't have a deployment url or image, because it is asking for my card info, and I don't have one that works. 
The website keeps rejecting the ones that I add.


## Results

Run `python train.py` and copy the numbers from `metrics.json` here: the 5-fold CV mean ± std, RMSE, macro F1, ROC AUC and confusion matrix on
the held-out test engines and the official test engines (all cycles, plus the standard "last cycle of each engine" benchmark), plus the model
comparison table from the notebook.

## Limitations

- The model sees one cycle at a time, so it tends to react late to degradation; smoothing over the last few cycles
  of an engine would help.
- FD001 has a single operating condition; the model is not expected to transfer to FD002–FD004 unchanged.
- RUL is capped at 125 during training, so predictions above ~125 are not meaningful (they are all `healthy`).
