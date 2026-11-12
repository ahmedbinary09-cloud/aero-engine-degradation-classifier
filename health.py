"""Shared helpers: column names, RUL -> health-state mapping, soft class probabilities."""

import numpy as np
import pandas as pd
from scipy.stats import norm

COLUMNS = (
    ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
)

# Sensors / settings that never change in FD001 (std == 0) carry no information.
CONSTANT_COLUMNS = ["op3", "s1", "s5", "s10", "s16", "s18", "s19"]

FEATURES = ["cycle", "op1", "op2"] + [
    f"s{i}" for i in range(1, 22) if f"s{i}" not in CONSTANT_COLUMNS
]

# Training target is capped: early in life the engine is "healthy" and the exact
# RUL is not predictable from the sensors (standard practice for C-MAPSS).
RUL_CAP = 125

# Health states by remaining cycles: upper edge (inclusive) of each state.
STATES = ["failure", "warning", "healthy"]
EDGES = [50, 100]  # failure (incl. "check immediately") <=50 < warning <=100 < healthy


def rul_to_state_id(rul):
    """Map RUL value(s) to class ids 0..2 (0=failure, 1=warning, 2=healthy)."""
    return np.digitize(np.asarray(rul, dtype=float), EDGES, right=True)


def rul_to_state(rul):
    return [STATES[i] for i in np.atleast_1d(rul_to_state_id(rul))]


def state_probabilities(rul_pred, sigma):
    """Turn a point RUL prediction into class probabilities.

    Assumes the prediction error is roughly Normal(0, sigma), where sigma is the
    validation RMSE. P(state) is the normal mass falling inside that state's RUL range.
    Returns an array of shape (n, 3) whose columns follow STATES.
    """
    r = np.atleast_1d(np.asarray(rul_pred, dtype=float))[:, None]
    bounds = np.array([-np.inf] + EDGES + [np.inf])[None, :]
    cdf = norm.cdf((bounds - r) / sigma)
    proba = np.diff(cdf, axis=1)
    return proba / proba.sum(axis=1, keepdims=True)


def load_train(path="data/train_FD001.txt"):
    """Train engines run to failure, so RUL = last cycle - current cycle."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    df["rul"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    df["rul_capped"] = df["rul"].clip(upper=RUL_CAP)
    return df


def load_test(path="data/test_FD001.txt", rul_path="data/RUL_FD001.txt"):
    """Test engines stop BEFORE failure. RUL_FD001.txt gives the true RUL after each
    engine's last recorded cycle (line i = engine i), so for any earlier cycle:
    RUL = RUL_at_last_cycle + (last cycle - current cycle)."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    final_rul = pd.read_csv(rul_path, header=None)[0]
    final_rul.index = range(1, len(final_rul) + 1)
    last = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = df["unit"].map(final_rul) + (last - df["cycle"])
    df["rul_capped"] = df["rul"].clip(upper=RUL_CAP)
    return df
