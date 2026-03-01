#!/usr/bin/env python3
# Tree baselines (LightGBM / XGBoost) for NPZ datasets.
# - Uses provided train/val/test splits
# - Per-sample regression on flattened (time, port) rows
# - Early stopping on val, then refit on train+val with best_iteration
#
# Examples:
#   python3 src/tree_baseline_pnz.py --data data/npz/dataset_sparse_10.npz --model lgbm
#   python3 src/tree_baseline_pnz.py --data data/npz/dataset_sparse_10.npz --model xgb
#   python3 src/tree_baseline_pnz.py --data data/npz/dataset_sparse_10.npz --model lgbm --tune
#   python3 src/tree_baseline_pnz.py --data data/npz/dataset_sparse_10.npz --model xgb --lead 1

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import ParameterGrid

try:
    import lightgbm as lgb
except Exception:
    lgb = None  # type: ignore

try:
    import xgboost as xgb
except Exception:
    xgb = None  # type: ignore


def _decode(arr) -> List[str]:
    out: List[str] = []
    for x in arr:
        if isinstance(x, (bytes, bytearray)):
            out.append(x.decode(errors="ignore"))
        else:
            out.append(str(x))
    return out


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float32).reshape(-1)
    b = np.asarray(b, np.float32).reshape(-1)
    if a.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def macro_rmse(Yhat: np.ndarray, Y: np.ndarray) -> float:
    Yhat = np.asarray(Yhat, np.float32)
    Y = np.asarray(Y, np.float32)
    N = Y.shape[1]
    return float(np.nanmean([rmse(Yhat[:, j], Y[:, j]) for j in range(N)]))


def make_xy(
    X: np.ndarray, Y: np.ndarray, tidx: np.ndarray, lead: int
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Flatten split into rows: (t,port) -> features X[t,port,:], target Y[t+lead,port]."""
    T = X.shape[0]
    tidx = np.asarray(tidx, int).tolist()
    t_valid = [t for t in tidx if (t + lead) < T]
    if not t_valid:
        return np.empty((0, X.shape[2]), np.float32), np.empty((0,), np.float32), []

    Xt = X[t_valid]  # [Ts,N,F]
    Yt = np.stack([Y[t + lead] for t in t_valid], axis=0)  # [Ts,N]
    Ts, N, F = Xt.shape
    return Xt.reshape(Ts * N, F), Yt.reshape(Ts * N), t_valid


def reshape_pred(yhat_flat: np.ndarray, n_ports: int) -> np.ndarray:
    yhat_flat = np.asarray(yhat_flat, np.float32).reshape(-1)
    if yhat_flat.size % n_ports != 0:
        raise ValueError(f"Pred size {yhat_flat.size} not divisible by N={n_ports}")
    return yhat_flat.reshape(-1, n_ports)


def compute_metrics(
    Yhat: np.ndarray,
    Y: np.ndarray,
    busy_thr: float,
    is_sensor: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    Yhat = np.asarray(Yhat, np.float32)
    Y = np.asarray(Y, np.float32)
    out: Dict[str, float] = {}

    # overall
    out["micro_RMSE"] = rmse(Yhat, Y)
    out["macro_RMSE"] = macro_rmse(Yhat, Y)

    # element-wise busy/idle
    idle = Y < busy_thr
    busy = ~idle
    out["idle_RMSE"] = rmse(Yhat[idle], Y[idle]) if np.any(idle) else float("nan")
    out["busy_RMSE"] = rmse(Yhat[busy], Y[busy]) if np.any(busy) else float("nan")
    out["idle_FP_rate"] = float(np.mean(Yhat[idle] >= busy_thr)) if np.any(idle) else float("nan")
    out["n_idle"] = int(np.sum(idle))
    out["n_busy"] = int(np.sum(busy))

    # frame-wise busy: any port busy in the frame
    busy_frame = (np.max(Y, axis=1) >= busy_thr)  # [Tframes]
    out["busyframe_ratio"] = float(np.mean(busy_frame))
    out["n_busyframes"] = int(np.sum(busy_frame))
    out["n_frames"] = int(Y.shape[0])

    if is_sensor is not None:
        is_sensor = np.asarray(is_sensor).astype(bool)
        hidden = ~is_sensor
        sens = is_sensor

        # sensor-only metrics
        if sens.any():
            out["sensor_micro_RMSE"] = rmse(Yhat[:, sens], Y[:, sens])
            sens_busy = (Y[:, sens] >= busy_thr)
            out["sensor_busy_RMSE"] = rmse(Yhat[:, sens][sens_busy], Y[:, sens][sens_busy]) if np.any(sens_busy) else float("nan")
        else:
            out["sensor_micro_RMSE"] = float("nan")
            out["sensor_busy_RMSE"] = float("nan")

        # hidden-only metrics (often a false-propagation check)
        if hidden.any():
            out["hidden_micro_RMSE"] = rmse(Yhat[:, hidden], Y[:, hidden])
            out["hidden_macro_RMSE"] = float(np.nanmean([rmse(Yhat[:, j], Y[:, j]) for j in np.where(hidden)[0]]))

            # element-wise busy on hidden (may be NaN if hidden never congests)
            hidden_busy = (Y[:, hidden] >= busy_thr)
            out["hidden_busy_RMSE"] = rmse(Yhat[:, hidden][hidden_busy], Y[:, hidden][hidden_busy]) if np.any(hidden_busy) else float("nan")

            # busy-frame hidden metrics (never NaN as long as there are any busy frames)
            if np.any(busy_frame):
                Yh = Y[busy_frame][:, hidden]
                Ph = Yhat[busy_frame][:, hidden]
                out["hidden_busyframe_micro_RMSE"] = rmse(Ph, Yh)
                out["hidden_busyframe_FP_rate"] = float(np.mean(Ph >= busy_thr))  # key when Yh is all-zero
            else:
                out["hidden_busyframe_micro_RMSE"] = float("nan")
                out["hidden_busyframe_FP_rate"] = float("nan")
        else:
            out["hidden_micro_RMSE"] = float("nan")
            out["hidden_macro_RMSE"] = float("nan")
            out["hidden_busy_RMSE"] = float("nan")
            out["hidden_busyframe_micro_RMSE"] = float("nan")
            out["hidden_busyframe_FP_rate"] = float("nan")

    return out


def nan_handle(X: np.ndarray, nan_mode: str) -> np.ndarray:
    if nan_mode == "keep":
        return X
    if nan_mode == "zero":
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    raise ValueError(f"Unknown nan_mode: {nan_mode}")


def fit_lgbm(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xval: np.ndarray,
    yval: np.ndarray,
    params: Dict[str, object],
    seed: int,
    n_estimators: int,
    early_stop: int,
    sw_tr: Optional[np.ndarray],
    sw_val: Optional[np.ndarray],
) -> Tuple[object, int, float]:
    if lgb is None:
        raise RuntimeError("lightgbm not installed. pip install lightgbm")

    base = dict(
        n_estimators=int(n_estimators),
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        num_leaves=63,
        min_child_samples=20,
        force_row_wise=True,
        verbosity=-1,
        random_state=seed,
        n_jobs=-1,
    )
    base.update(params)

    model = lgb.LGBMRegressor(**base)

    callbacks = []
    if early_stop > 0:
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stop, first_metric_only=True))

    fit_kwargs = dict(
        X=Xtr,
        y=ytr,
        sample_weight=sw_tr,
        eval_set=[(Xval, yval)],
        eval_metric="rmse",
        callbacks=callbacks,
    )
    # LightGBM is picky about eval_sample_weight=None in some versions
    if sw_val is not None:
        fit_kwargs["eval_sample_weight"] = [sw_val]

    model.fit(**fit_kwargs)

    best_it = int(getattr(model, "best_iteration_", 0) or base["n_estimators"])
    best_rmse = float(getattr(model, "best_score_", {}).get("valid_0", {}).get("rmse", np.nan))
    if np.isnan(best_rmse):
        best_rmse = rmse(model.predict(Xval), yval)

    return model, best_it, best_rmse


def fit_xgb(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xval: np.ndarray,
    yval: np.ndarray,
    params: Dict[str, object],
    seed: int,
    n_estimators: int,
    early_stop: int,
    sw_tr: Optional[np.ndarray],
    sw_val: Optional[np.ndarray],
) -> Tuple[object, int, float]:
    if xgb is None:
        raise RuntimeError("xgboost not installed. pip install xgboost")

    base = dict(
        n_estimators=int(n_estimators),
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=1.0,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        verbosity=0,
        random_state=seed,
        n_jobs=-1,
    )
    # XGBoost 2.1+ expects early_stopping_rounds in the constructor (fit() arg removed)
    if early_stop and early_stop > 0:
        base["early_stopping_rounds"] = int(early_stop)

    base.update(params)

    # Newer XGBoost: early_stopping_rounds must be in ctor
    try:
        model = xgb.XGBRegressor(**base)
        fit_kwargs = dict(
            X=Xtr,
            y=ytr,
            sample_weight=sw_tr,
            eval_set=[(Xval, yval)],
            verbose=False,
        )

        if sw_val is not None:
            # some versions support sample_weight_eval_set, some don't
            try:
                model.fit(**fit_kwargs, sample_weight_eval_set=[sw_val])
            except TypeError:
                model.fit(**fit_kwargs)
        else:
            model.fit(**fit_kwargs)

    except TypeError:
        # Older XGBoost fallback: ctor may not accept early_stopping_rounds -> pass via fit()
        base2 = dict(base)
        es = base2.pop("early_stopping_rounds", None)
        model = xgb.XGBRegressor(**base2)

        fit_kwargs = dict(
            X=Xtr,
            y=ytr,
            sample_weight=sw_tr,
            eval_set=[(Xval, yval)],
            verbose=False,
        )
        if es is not None:
            fit_kwargs["early_stopping_rounds"] = es

        if sw_val is not None:
            try:
                model.fit(**fit_kwargs, sample_weight_eval_set=[sw_val])
            except TypeError:
                model.fit(**fit_kwargs)
        else:
            model.fit(**fit_kwargs)

    # best_iteration is 0-based in xgboost; convert to "number of trees"
    bi = getattr(model, "best_iteration", None)
    if bi is None:
        best_it = int(base.get("n_estimators", n_estimators))
    else:
        best_it = int(bi) + 1

    best_rmse = float(getattr(model, "best_score", np.nan))
    if np.isnan(best_rmse):
        best_rmse = rmse(model.predict(Xval), yval)

    return model, best_it, best_rmse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="NPZ dataset path")
    ap.add_argument("--model", choices=["lgbm", "xgb"], default="lgbm")
    ap.add_argument("--lead", type=int, default=0, help="Predict y_{t+lead} from X_t (0=nowcast)")
    ap.add_argument("--busy_thr", type=float, default=50.0)
    ap.add_argument("--busy_weight", type=float, default=1.0, help="If >1, upweight samples with y>=busy_thr")
    ap.add_argument("--nan_mode", choices=["zero", "keep"], default="zero")
    ap.add_argument("--n_estimators", type=int, default=2000)
    ap.add_argument("--early_stop", type=int, default=100)
    ap.add_argument("--tune", action="store_true", help="Small grid search on val (kept modest)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_npz", default=None, help="Optional: save {y_test,pred,meta} npz")
    args = ap.parse_args()

    Z = np.load(args.data, allow_pickle=True)
    X = Z["X"].astype(np.float32)  # [T,N,F]
    Y = Z["Y"].astype(np.float32)  # [T,N]
    train_idx = Z["train_idx"].astype(int)
    val_idx = Z["val_idx"].astype(int)
    test_idx = Z["test_idx"].astype(int)
    is_sensor = Z.get("is_sensor", None)

    feat_names = _decode(Z.get("feat_names", np.arange(X.shape[2])))
    label_name = _decode(Z.get("label_name", np.array(["Y"])))[0]

    # splits
    Xtr, ytr, t_tr = make_xy(X, Y, train_idx, lead=args.lead)
    Xval, yval, t_val = make_xy(X, Y, val_idx, lead=args.lead)
    Xte, _, t_te = make_xy(X, Y, test_idx, lead=args.lead)
    if Xtr.size == 0 or Xval.size == 0 or Xte.size == 0:
        raise RuntimeError(f"Empty split after lead={args.lead}.")

    Xtr = nan_handle(Xtr, args.nan_mode)
    Xval = nan_handle(Xval, args.nan_mode)
    Xte = nan_handle(Xte, args.nan_mode)

    # sample weights (optional): focus busy targets
    sw_tr = sw_val = None
    if args.busy_weight and args.busy_weight != 1.0:
        sw_tr = np.where(ytr >= args.busy_thr, args.busy_weight, 1.0).astype(np.float32)
        sw_val = np.where(yval >= args.busy_thr, args.busy_weight, 1.0).astype(np.float32)

    # --- pick params + best_iter on (train,val) ---
    if args.tune:
        if args.model == "lgbm":
            grid = ParameterGrid(
                {
                    "num_leaves": [31, 63],
                    "min_child_samples": [20, 40],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                    "reg_lambda": [0.0, 1.0],
                }
            )
            fit_one = fit_lgbm
        else:
            grid = ParameterGrid(
                {
                    "max_depth": [4, 8],
                    "min_child_weight": [1.0, 5.0],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                    "reg_lambda": [1.0, 10.0],
                }
            )
            fit_one = fit_xgb

        best = None  # (rmse, best_it, params)
        for p in grid:
            _, best_it, val_rmse = fit_one(
                Xtr, ytr, Xval, yval, p, args.seed, args.n_estimators, args.early_stop, sw_tr, sw_val
            )
            if best is None or val_rmse < best[0]:
                best = (val_rmse, best_it, p)
        assert best is not None
        best_val_rmse, best_iter, best_params = best
    else:
        # single run with early stopping => get best_iter
        if args.model == "lgbm":
            _, best_iter, best_val_rmse = fit_lgbm(
                Xtr, ytr, Xval, yval, {}, args.seed, args.n_estimators, args.early_stop, sw_tr, sw_val
            )
        else:
            _, best_iter, best_val_rmse = fit_xgb(
                Xtr, ytr, Xval, yval, {}, args.seed, args.n_estimators, args.early_stop, sw_tr, sw_val
            )
        best_params = {}

    # --- refit on train+val using best_iter ---
    trval_idx = np.sort(np.concatenate([train_idx, val_idx]))
    Xtrv, ytrv, _ = make_xy(X, Y, trval_idx, lead=args.lead)
    Xtrv = nan_handle(Xtrv, args.nan_mode)
    sw_trv = None
    if args.busy_weight and args.busy_weight != 1.0:
        sw_trv = np.where(ytrv >= args.busy_thr, args.busy_weight, 1.0).astype(np.float32)

    # disable early stopping in final fit, lock n_estimators=best_iter
    if args.model == "lgbm":
        model, _, _ = fit_lgbm(
            Xtrv, ytrv, Xval, yval,
            {**best_params, "n_estimators": int(max(1, best_iter))},
            args.seed, int(max(1, best_iter)), 0, sw_trv, sw_val
        )
    else:
        model, _, _ = fit_xgb(
            Xtrv, ytrv, Xval, yval,
            {**best_params, "n_estimators": int(max(1, best_iter))},
            args.seed, int(max(1, best_iter)), 0, sw_trv, sw_val
        )

    # --- test ---
    N = Y.shape[1]
    yhat_flat = model.predict(Xte).astype(np.float32)
    Yhat = reshape_pred(yhat_flat, N)
    Yref = np.stack([Y[t + args.lead] for t in t_te], axis=0).astype(np.float32)

    out = {
        "model": args.model,
        "data": args.data,
        "label": label_name,
        "lead": int(args.lead),
        "nan_mode": args.nan_mode,
        "busy_thr": float(args.busy_thr),
        "busy_weight": float(args.busy_weight),
        "n_estimators_max": int(args.n_estimators),
        "early_stop": int(args.early_stop),
        "tuned": bool(args.tune),
        "best_val_rmse": float(best_val_rmse),
        "best_iteration": int(best_iter),
        "ports": int(N),
        "features": int(X.shape[2]),
        "feat_names": feat_names,
        "n_train_times": int(len(t_tr)),
        "n_val_times": int(len(t_val)),
        "n_test_times": int(len(t_te)),
        "metrics_test": compute_metrics(Yhat, Yref, args.busy_thr, is_sensor=is_sensor),
    }

    print(json.dumps(out, indent=2))

    if args.save_npz:
        np.savez_compressed(
            args.save_npz,
            y_test=Yref,
            pred=Yhat,
            meta=json.dumps({k: v for k, v in out.items() if k != "feat_names"}, ensure_ascii=False),
        )
        print(f"[OK] saved {args.save_npz}")


if __name__ == "__main__":
    main()
