#!/usr/bin/env python
"""Train the RUL regression model and save it (model.bin) for predict.py.

Pipeline: load C-MAPSS FD001 train (run-to-failure) -> RUL target (capped)
-> split ENGINES 60/20/20 (train / validation / test; full_train = train + validation)
-> pick XGBoost params on validation (early stopping) -> 5-fold CV on full_train
(folds split by engine; mean and std) -> refit on full_train
-> evaluate on the held-out test engines AND on the official test_FD001 + RUL_FD001
(RMSE, F1, ROC AUC, confusion matrix) -> save bundle.
"""

import json
import pickle
from itertools import product

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from health import (
    FEATURES,
    RUL_CAP,
    STATES,
    load_test,
    load_train,
    rul_to_state_id,
    state_probabilities,
)

MODEL_PATH = "model.bin"
SEED = 1
N_SPLITS = 5   # 5 folds of ~16 engines each; 10 folds would leave only ~8 engines per fold

print(f"pandas=={pd.__version__}")
print(f"numpy=={np.__version__}")
print(f"sklearn=={sklearn.__version__}")
print(f"xgboost=={xgb.__version__}")


def split_by_engine(df, test_size):
    """Split ENGINES (not rows), so no engine appears on both sides."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    idx_train, idx_val = next(gss.split(df, groups=df["unit"]))
    return df.iloc[idx_train].copy(), df.iloc[idx_val].copy()


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def make_model(params, early_stopping=False):
    return xgb.XGBRegressor(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        min_child_weight=params["min_child_weight"],
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=30 if early_stopping else None,
        random_state=SEED,
        n_jobs=-1,
    )


def select_params(df_train, df_val):
    """Small grid over XGBoost params, scored on validation RMSE.

    Early stopping on the validation engines limits the number of trees, which is
    the main guard against over-fitting.
    """
    best = None
    for max_depth, eta, mcw in product([3, 4, 6], [0.03, 0.1], [1, 10]):
        params = dict(n_estimators=1000, learning_rate=eta, max_depth=max_depth,
                      min_child_weight=mcw)
        model = make_model(params, early_stopping=True)
        model.fit(
            df_train[FEATURES],
            df_train["rul_capped"],
            eval_set=[(df_val[FEATURES], df_val["rul_capped"])],
            verbose=False,
        )
        score = rmse(df_val["rul_capped"], model.predict(df_val[FEATURES]))
        print(f"max_depth={max_depth} eta={eta} mcw={mcw} "
              f"trees={model.best_iteration + 1} val_rmse={score:.3f}")
        if best is None or score < best["rmse"]:
            best = dict(params, rmse=score, n_estimators=int(model.best_iteration + 1))
    return best


def train_final(df_fit, best):
    model = make_model(best)
    model.fit(df_fit[FEATURES], df_fit["rul_capped"])
    return model


def cross_validate(df_full_train, best, n_splits=N_SPLITS):
    """K-fold CV on full_train. GroupKFold keeps all cycles of an engine in the same fold."""
    scores = {"rmse_capped": [], "f1_macro": [], "roc_auc_ovr_macro": []}
    gkf = GroupKFold(n_splits=n_splits)
    for fold, (idx_tr, idx_va) in enumerate(gkf.split(df_full_train, groups=df_full_train["unit"])):
        d_tr, d_va = df_full_train.iloc[idx_tr], df_full_train.iloc[idx_va]
        model = train_final(d_tr, best)
        pred = np.clip(model.predict(d_va[FEATURES]), 0, RUL_CAP)
        fold_rmse = rmse(d_va["rul_capped"], pred)
        m = classification_metrics(d_va["rul"], pred, sigma=fold_rmse)
        scores["rmse_capped"].append(fold_rmse)
        scores["f1_macro"].append(m["f1_macro"])
        scores["roc_auc_ovr_macro"].append(m["roc_auc_ovr_macro"])
        print(f"fold {fold}: rmse={fold_rmse:.2f} f1_macro={m['f1_macro']:.3f} auc={m['roc_auc_ovr_macro']:.3f}")
    summary = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "folds": [float(x) for x in v]}
               for k, v in scores.items()}
    for k, v in summary.items():
        print(f"CV {k}: {v['mean']:.3f} +- {v['std']:.3f}")
    return summary


def classification_metrics(rul_true, rul_pred, sigma):
    y_state = rul_to_state_id(rul_true)
    p_state = rul_to_state_id(rul_pred)
    labels = list(range(len(STATES)))
    return {
        "f1_macro": float(f1_score(y_state, p_state, average="macro")),
        "f1_weighted": float(f1_score(y_state, p_state, average="weighted")),
        "roc_auc_ovr_macro": float(roc_auc_score(
            y_state, state_probabilities(rul_pred, sigma),
            multi_class="ovr", average="macro", labels=labels)),
        "confusion_matrix": confusion_matrix(y_state, p_state, labels=labels).tolist(),
    }


def evaluate_all_cycles(model, df, sigma):
    """Every recorded cycle of every engine in df."""
    pred = np.clip(model.predict(df[FEATURES]), 0, RUL_CAP)
    return {"rmse_capped": rmse(df["rul_capped"], pred),
            **classification_metrics(df["rul"], pred, sigma)}


def evaluate_last_cycle(model, df, sigma):
    """Standard C-MAPSS benchmark: one row per engine (its last recorded cycle).
    Only meaningful for the official test engines, which stop before failure."""
    last = df.groupby("unit").tail(1)
    pred = np.clip(model.predict(last[FEATURES]), 0, RUL_CAP)
    return {"rmse_capped": rmse(last["rul_capped"], pred),
            "rmse_raw": rmse(last["rul"], pred),
            **classification_metrics(last["rul"], pred, sigma)}


def save_model(filename, bundle):
    with open(filename, "wb") as f_out:
        pickle.dump(bundle, f_out)
    print(f"model saved to {filename}")


def main():
    df = load_train()
    df_official = load_test()      # official test_FD001 + RUL_FD001 (also used for streaming)

    # 60% train / 20% validation / 20% test (by engine); full_train = train + validation
    df_full_train, df_test = split_by_engine(df, test_size=0.2)
    df_train, df_val = split_by_engine(df_full_train, test_size=0.25)
    print(f"engines: train={df_train.unit.nunique()} val={df_val.unit.nunique()} "
          f"test={df_test.unit.nunique()} | official test={df_official.unit.nunique()}")

    best = select_params(df_train, df_val)
    print("best params:", best)

    # sigma = typical prediction error on unseen engines, used for class probabilities
    val_model = train_final(df_train, best)
    sigma = rmse(df_val["rul_capped"], val_model.predict(df_val[FEATURES]))

    # k-fold check of the chosen params on full_train: mean and std across folds
    cv = cross_validate(df_full_train, best)

    # final model: full_train (train + validation engines)
    model = train_final(df_full_train, best)
    metrics = {
        "states_order": STATES,
        f"cv_{N_SPLITS}fold_full_train": cv,
        "holdout_test": evaluate_all_cycles(model, df_test, sigma),
        "official_test_all_cycles": evaluate_all_cycles(model, df_official, sigma),
        "official_test_last_cycle": evaluate_last_cycle(model, df_official, sigma),
    }
    print(json.dumps(metrics, indent=2))

    bundle = {
        "model": model,
        "features": FEATURES,
        "sigma": sigma,
        "rul_cap": RUL_CAP,
        "metrics": metrics,
        "params": best,
    }
    save_model(MODEL_PATH, bundle)
    with open("metrics.json", "w") as f_out:
        json.dump({"params": best, **metrics}, f_out, indent=2)


if __name__ == "__main__":
    main()
