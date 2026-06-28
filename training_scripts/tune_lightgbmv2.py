"""
tune_lightgbm_v2.py
====================
Optuna hyperparameter search for the v2 LightGBM residual corrector.
Uses 37 features (train_lightgbm_v2.py feature set).

Objective
---------
Weighted combination targeting the full rainfall spectrum:
    score = 0.20 * overall_RMSE
          + 0.10 * overall_MAE
          + 0.25 * heavy_RMSE_25   (y > 25 mm/h)
          + 0.25 * heavy_RMSE_50   (y > 50 mm/h)
          + 0.20 * heavy_RMSE_75   (y > 75 mm/h)
          + mid_bias_weight * max(0, mean_bias_10_50)   (overcorrection guard)
          + mid_bias_weight * 0.3 * max(0, -mean_bias_25)  (undercorrection nudge)

Search space
------------
    w_min             : [0.5,  2.0]       smooth exp weight curve minimum
    w_max             : [25.0, 80.0]      smooth exp weight curve maximum
    gamma             : [0.02, 0.12]      smooth exp weight curve growth rate
    use_huber         : [True, False]      loss function
    huber_delta       : [0.3,  2.0]       Huber delta (log scale)
    lr                : [5e-4, 5e-3]      learning rate (log scale)
    num_leaves        : [48,   128]
    max_depth         : [5,    9]
    reg_alpha         : [0.05, 1.0]       L1 regularisation (log scale)
    reg_lambda        : [5.0,  30.0]      L2 regularisation (log scale)
    min_child_samples : [5,    20]
    subsample         : [0.6,  0.9]
    colsample         : [0.7,  1.0]

Fixed
-----
    n_estimators = 3000
    early_stop   = 200
    use_dart     = False

Warm start
----------
Enqueues reasonable starting params so trial 0 is already a strong baseline.

Outputs
-------
    <output_dir>/optuna_study_v2.pkl    full Optuna study object
    <output_dir>/best_params_v2.json    best hyperparameters
    <output_dir>/tune_v2_log.txt        full log

Usage
-----
    python tune_lightgbm_v2.py

    python tune_lightgbm_v2.py \\
        --n_trials 60 \\
        --output_dir ../models/qpe_cnn_kan/optuna_v2
"""

import sys
from pathlib import Path

_ROOT          = Path(__file__).resolve().parent.parent
_AI_SCRIPT_DIR = _ROOT / "AI_Script"
if str(_AI_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_AI_SCRIPT_DIR))

import argparse
import json
import logging
import pickle
import warnings

import numpy as np
from scipy.stats import skew, kurtosis
import torch
import torch.nn as nn
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    from cnn_feature_extractor import RadarCNN
    from kan_rainfall_predictor import RainfallKAN
except ImportError as e:
    sys.exit(f"[FATAL] Could not import from {_AI_SCRIPT_DIR}\n{e}")

try:
    from optuna_integration.lightgbm import LightGBMPruningCallback
except ImportError:
    try:
        from optuna.integration.lightgbm import LightGBMPruningCallback
    except ImportError:
        LightGBMPruningCallback = None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(output_dir: Path) -> logging.Logger:
    log = logging.getLogger("lgb_tune_v2")
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    log.addHandler(ch)
    fh = logging.FileHandler(output_dir / "tune_v2_log.txt", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log


# ---------------------------------------------------------------------------
# CNN+KAN model
# ---------------------------------------------------------------------------

class QPEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = RadarCNN(in_channels=4, feature_dim=32)
        self.kan = RainfallKAN(input_dim=32, hidden1=64, hidden2=32,
                               grid_size=5, spline_order=3)
    def forward(self, x):
        return self.kan(self.cnn(x))


def load_cnn_kan(model_path, device, log):
    ckpt       = torch.load(model_path, map_location=device)
    chan_means  = np.array(ckpt["chan_means"], dtype=np.float32)
    chan_stds   = np.array(ckpt["chan_stds"],  dtype=np.float32)
    model       = QPEModel().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    log.info("Loaded CNN+KAN  epoch=%d  val_loss=%.5f",
             ckpt.get("epoch", -1), ckpt.get("val_loss", -1))
    return model, chan_means, chan_stds


def normalise(X, means, stds):
    X = X.astype(np.float32).copy()
    for c in range(4):
        X[:, c] = (X[:, c] - means[c]) / (stds[c] + 1e-8)
    return X


@torch.no_grad()
def get_kan_predictions(model, X_norm, device, batch=512):
    preds = []
    for s in range(0, len(X_norm), batch):
        x_t = torch.from_numpy(X_norm[s:s+batch]).to(device)
        preds.append(model(x_t).squeeze(1).cpu().numpy())
    pred_log1p = np.concatenate(preds)
    return np.clip(np.expm1(pred_log1p), 0.0, None), pred_log1p


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def apply_lgb_correction(pred_mmh, epsilon_log):
    """R_final = clamp(expm1(log1p(R_hat) + epsilon_log), min=0)"""
    return np.clip(
        np.expm1(np.log1p(np.clip(pred_mmh, 0.0, None)) + epsilon_log), 0.0, None)


# ---------------------------------------------------------------------------
# Features (37) — identical to train_lightgbm_v2.py
# ---------------------------------------------------------------------------

CENTER = 4

FEATURE_NAMES = [
    "Z_low_norm", "ZDR_low_norm", "Z_high_norm", "ZDR_high_norm",
    "kan_pred_mmh", "kan_pred_log1p", "kan_pred_sq",
    "Z_diff", "ZDR_diff",
    "Z_low_raw", "ZDR_low_raw",
    "Z_low_std", "Z_low_max", "Z_low_min", "Z_low_skew", "Z_low_kurt",
    "Z_conv_frac",
    "ZDR_low_std", "ZDR_low_mean", "ZDR_low_max",
    "Z_high_std", "Z_high_max", "Z_high_conv_frac",
    "ZDR_high_std",
    "Z_max_ratio", "Z_mean_ratio", "Z_high_min",
    "ZDR_low_skew", "Z_ZDR_consistency", "Z_low_center_vs_mean",
    "Z_high_center_vs_mean",
    "Z_x_kan", "Zdiff_x_kan", "Zhigh_x_kan", "ZDR_x_kan",
    "Zratio_x_kan", "ZDRmax_x_kan",
]

MONOTONE_CONSTRAINTS = [
    0, 0, 0, 0,
    1, 1, 1,
    0, 0,
    0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0,
    0, 0, 0,
    0, 0, 0,
    0,
    0, 0, 0, 0,
    0, 0,
]

assert len(MONOTONE_CONSTRAINTS) == len(FEATURE_NAMES)


def build_features(X_raw, X_norm, pred_mmh):
    N = len(X_raw)
    Z_l   = X_raw[:, 0, CENTER, CENTER]
    ZDR_l = X_raw[:, 1, CENTER, CENTER]
    Z_h   = X_raw[:, 2, CENTER, CENTER]
    ZDR_h = X_raw[:, 3, CENTER, CENTER]
    Z_diff   = (Z_l   - Z_h).astype(np.float32)
    ZDR_diff = (ZDR_l - ZDR_h).astype(np.float32)
    Z_win        = X_raw[:, 0].reshape(N, -1)
    ZDR_win      = X_raw[:, 1].reshape(N, -1)
    Z_high_win   = X_raw[:, 2].reshape(N, -1)
    ZDR_high_win = X_raw[:, 3].reshape(N, -1)
    Z_low_std    = Z_win.std(axis=1).astype(np.float32)
    Z_low_max    = Z_win.max(axis=1).astype(np.float32)
    Z_low_min    = Z_win.min(axis=1).astype(np.float32)
    Z_low_mean   = Z_win.mean(axis=1).astype(np.float32)
    Z_conv_frac  = (Z_win > 40.0).mean(axis=1).astype(np.float32)
    Z_low_skew   = skew(Z_win,     axis=1).astype(np.float32)
    Z_low_kurt   = kurtosis(Z_win, axis=1).astype(np.float32)
    ZDR_low_std  = ZDR_win.std(axis=1).astype(np.float32)
    ZDR_low_mean = ZDR_win.mean(axis=1).astype(np.float32)
    ZDR_low_max  = ZDR_win.max(axis=1).astype(np.float32)
    ZDR_low_skew = skew(ZDR_win, axis=1).astype(np.float32)
    Z_high_std       = Z_high_win.std(axis=1).astype(np.float32)
    Z_high_max       = Z_high_win.max(axis=1).astype(np.float32)
    Z_high_min       = Z_high_win.min(axis=1).astype(np.float32)
    Z_high_mean      = Z_high_win.mean(axis=1).astype(np.float32)
    Z_high_conv_frac = (Z_high_win > 35.0).mean(axis=1).astype(np.float32)
    ZDR_high_std = ZDR_high_win.std(axis=1).astype(np.float32)
    Z_max_ratio  = (Z_high_max  / (Z_low_max  + 1e-8)).astype(np.float32)
    Z_mean_ratio = (Z_high_mean / (Z_low_mean + 1e-8)).astype(np.float32)
    ZDR_l_pos             = np.clip(ZDR_l, 0.0, None)
    Z_ZDR_consistency     = (Z_l - 20.0 * np.log10(ZDR_l_pos + 1.0)).astype(np.float32)
    Z_low_center_vs_mean  = (Z_l - Z_low_mean).astype(np.float32)
    Z_high_center_vs_mean = (Z_h - Z_high_mean).astype(np.float32)
    pred_mmh_pos   = np.clip(pred_mmh, 0.0, None).astype(np.float32)
    pred_log1p_pos = np.log1p(pred_mmh_pos)
    kan_pred_sq    = (pred_mmh_pos ** 2).astype(np.float32)
    Z_x_kan      = (Z_l         * pred_log1p_pos).astype(np.float32)
    Zdiff_x_kan  = (Z_diff      * pred_log1p_pos).astype(np.float32)
    Zhigh_x_kan  = (Z_h         * pred_log1p_pos).astype(np.float32)
    ZDR_x_kan    = (ZDR_l       * pred_log1p_pos).astype(np.float32)
    Zratio_x_kan = (Z_max_ratio * pred_log1p_pos).astype(np.float32)
    ZDRmax_x_kan = (ZDR_low_max * pred_log1p_pos).astype(np.float32)
    return np.column_stack([
        X_norm[:, 0, CENTER, CENTER], X_norm[:, 1, CENTER, CENTER],
        X_norm[:, 2, CENTER, CENTER], X_norm[:, 3, CENTER, CENTER],
        pred_mmh_pos, pred_log1p_pos, kan_pred_sq,
        Z_diff, ZDR_diff, Z_l, ZDR_l,
        Z_low_std, Z_low_max, Z_low_min, Z_low_skew, Z_low_kurt, Z_conv_frac,
        ZDR_low_std, ZDR_low_mean, ZDR_low_max,
        Z_high_std, Z_high_max, Z_high_conv_frac,
        ZDR_high_std,
        Z_max_ratio, Z_mean_ratio, Z_high_min,
        ZDR_low_skew, Z_ZDR_consistency, Z_low_center_vs_mean,
        Z_high_center_vs_mean,
        Z_x_kan, Zdiff_x_kan, Zhigh_x_kan, ZDR_x_kan,
        Zratio_x_kan, ZDRmax_x_kan,
    ]).astype(np.float32)


# ---------------------------------------------------------------------------
# Sample weights
# ---------------------------------------------------------------------------

def compute_sample_weights(y_true, w_min, w_max, gamma):
    y     = np.clip(y_true, 0.0, None).astype(np.float64)
    denom = np.expm1(gamma * 150.0)
    if denom < 1e-8:
        w = w_min + (w_max - w_min) * (y / 150.0)
    else:
        w = w_min + (w_max - w_min) * np.expm1(gamma * y) / denom
    return w.astype(np.float32)


# ---------------------------------------------------------------------------
# Objective factory
# ---------------------------------------------------------------------------

def make_objective(
    X_train_raw, y_train, X_val_raw, y_val,
    X_train_norm, X_val_norm,
    train_pred_mmh, val_pred_mmh,
    rmse_weight, mae_weight,
    heavy_weight_25, heavy_weight_50, heavy_weight_75,
    mid_bias_weight,
    n_estimators, early_stop,
    log,
):
    # Build feature matrices once — shared across all trials
    log.info("Building feature matrices once for all trials...")
    X_lgb_train = build_features(X_train_raw, X_train_norm, train_pred_mmh)
    X_lgb_val   = build_features(X_val_raw,   X_val_norm,   val_pred_mmh)
    log.info("  Train features: %s  Val features: %s",
             X_lgb_train.shape, X_lgb_val.shape)

    # Log1p residuals
    train_resid_log = (
        np.log1p(np.clip(y_train, 0.0, None)) -
        np.log1p(np.clip(train_pred_mmh, 0.0, None))
    ).astype(np.float32)

    val_resid_log = (
        np.log1p(np.clip(y_val, 0.0, None)) -
        np.log1p(np.clip(val_pred_mmh, 0.0, None))
    ).astype(np.float32)

    # Masks
    heavy_mask_25 = y_val > 25.0
    heavy_mask_50 = y_val > 50.0
    heavy_mask_75 = y_val > 75.0
    mid_mask      = (y_val >= 10.0) & (y_val <= 50.0)
    n_heavy_25    = heavy_mask_25.sum()
    n_heavy_50    = heavy_mask_50.sum()
    n_heavy_75    = heavy_mask_75.sum()
    n_mid         = mid_mask.sum()

    def objective(trial: optuna.Trial) -> float:

        # ── Sample hyperparameters ─────────────────────────────────────────
        w_min             = trial.suggest_float("w_min",             0.5,   2.0)
        w_max             = trial.suggest_float("w_max",            25.0,  80.0)
        gamma             = trial.suggest_float("gamma",             0.02,  0.12)
        use_huber         = trial.suggest_categorical("use_huber", [True, False])
        huber_delta       = trial.suggest_float("huber_delta",       0.3,   2.0, log=True)
        lr                = trial.suggest_float("lr",               5e-4,  5e-3, log=True)
        num_leaves        = trial.suggest_int(  "num_leaves",        48,   128)
        max_depth         = trial.suggest_int(  "max_depth",          5,     9)
        reg_alpha         = trial.suggest_float("reg_alpha",         0.05,  1.0, log=True)
        reg_lambda        = trial.suggest_float("reg_lambda",        5.0,  30.0, log=True)
        min_child_samples = trial.suggest_int(  "min_child_samples",  5,    20)
        subsample         = trial.suggest_float("subsample",          0.6,   0.9)
        colsample         = trial.suggest_float("colsample",          0.7,   1.0)

        weights = compute_sample_weights(y_train, w_min, w_max, gamma)

        objective_fn = "huber" if use_huber else "regression_l2"
        # Internal metric name LightGBM reports (for pruning callback)
        metric_internal = "huber" if use_huber else "l2"
        # Param value LightGBM accepts
        metric_param    = "huber" if use_huber else "mse"
        alpha_val       = huber_delta if use_huber else 0.9

        lgb_train_ds = lgb.Dataset(
            X_lgb_train, label=train_resid_log,
            weight=weights, feature_name=FEATURE_NAMES, free_raw_data=False)
        lgb_val_ds = lgb.Dataset(
            X_lgb_val, label=val_resid_log,
            reference=lgb_train_ds, feature_name=FEATURE_NAMES, free_raw_data=False)

        params = {
            "boosting_type":        "gbdt",
            "objective":            objective_fn,
            "alpha":                alpha_val,
            "metric":               metric_param,
            "learning_rate":        lr,
            "max_depth":            max_depth,
            "num_leaves":           num_leaves,
            "subsample":            subsample,
            "subsample_freq":       1,
            "colsample_bytree":     colsample,
            "reg_alpha":            reg_alpha,
            "reg_lambda":           reg_lambda,
            "min_child_samples":    min_child_samples,
            "monotone_constraints": MONOTONE_CONSTRAINTS,
            "verbose":              -1,
            "n_jobs":               -1,
        }

        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stop, verbose=False),
            lgb.log_evaluation(period=-1),
        ]
        if LightGBMPruningCallback is not None:
            callbacks.append(
                LightGBMPruningCallback(trial, metric_internal, valid_name="valid_0"))

        try:
            lgb_model = lgb.train(
                params, lgb_train_ds,
                num_boost_round=n_estimators,
                valid_sets=[lgb_val_ds],
                callbacks=callbacks,
            )
        except optuna.exceptions.TrialPruned:
            raise

        num_iter = (lgb_model.best_iteration
                    if lgb_model.best_iteration > 0
                    else lgb_model.num_trees())

        epsilon_log = lgb_model.predict(X_lgb_val, num_iteration=num_iter)
        final_pred  = apply_lgb_correction(val_pred_mmh, epsilon_log)

        overall_rmse = float(np.sqrt(np.mean((final_pred - y_val) ** 2)))
        overall_mae  = float(np.mean(np.abs(final_pred - y_val)))

        heavy_rmse_25 = (float(np.sqrt(np.mean(
            (final_pred[heavy_mask_25] - y_val[heavy_mask_25]) ** 2)))
            if n_heavy_25 >= 5 else overall_rmse)

        heavy_rmse_50 = (float(np.sqrt(np.mean(
            (final_pred[heavy_mask_50] - y_val[heavy_mask_50]) ** 2)))
            if n_heavy_50 >= 5 else heavy_rmse_25)

        heavy_rmse_75 = (float(np.sqrt(np.mean(
            (final_pred[heavy_mask_75] - y_val[heavy_mask_75]) ** 2)))
            if n_heavy_75 >= 5 else heavy_rmse_50)

        # Two-sided bias terms
        mid_overfit_penalty = 0.0
        tail_under_penalty  = 0.0
        if n_mid >= 5:
            mid_bias = float(np.mean(final_pred[mid_mask] - y_val[mid_mask]))
            mid_overfit_penalty = max(0.0,  mid_bias)   # penalise overcorrection
        if n_heavy_25 >= 5:
            tail_bias = float(np.mean(final_pred[heavy_mask_25] - y_val[heavy_mask_25]))
            tail_under_penalty = max(0.0, -tail_bias)   # penalise undercorrection

        score = (rmse_weight     * overall_rmse       +
                 mae_weight      * overall_mae        +
                 heavy_weight_25 * heavy_rmse_25      +
                 heavy_weight_50 * heavy_rmse_50      +
                 heavy_weight_75 * heavy_rmse_75      +
                 mid_bias_weight * mid_overfit_penalty +
                 mid_bias_weight * 0.3 * tail_under_penalty)

        trial.set_user_attr("overall_rmse",  round(overall_rmse,  4))
        trial.set_user_attr("overall_mae",   round(overall_mae,   4))
        trial.set_user_attr("heavy_rmse_25", round(heavy_rmse_25, 4))
        trial.set_user_attr("heavy_rmse_50", round(heavy_rmse_50, 4))
        trial.set_user_attr("heavy_rmse_75", round(heavy_rmse_75, 4))
        trial.set_user_attr("mid_bias",      round(mid_overfit_penalty, 4))
        trial.set_user_attr("tail_under",    round(tail_under_penalty,  4))
        trial.set_user_attr("n_trees",       num_iter)

        return score

    return objective


# ---------------------------------------------------------------------------
# Main tune function
# ---------------------------------------------------------------------------

def tune(
    data_dir, model_dir, output_dir, n_trials,
    rmse_weight, mae_weight,
    heavy_weight_25, heavy_weight_50, heavy_weight_75,
    mid_bias_weight,
    n_estimators, early_stop,
    log,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Load data ──────────────────────────────────────────────────────────
    log.info("Loading dataset from %s", data_dir)
    X_train_raw = np.load(data_dir / "X_train.npy")
    y_train     = np.load(data_dir / "y_train.npy")
    X_val_raw   = np.load(data_dir / "X_val.npy")
    y_val       = np.load(data_dir / "y_val.npy")
    log.info("Train: %d  Val: %d", len(y_train), len(y_val))
    for thr in [25, 50, 75]:
        mask = y_val > thr
        log.info("  Val >%d mm/h : %d (%.1f%%)", thr, mask.sum(), 100*mask.mean())

    # ── Load CNN+KAN ───────────────────────────────────────────────────────
    log.info("Loading CNN+KAN from %s", model_dir / "best_model.pt")
    model, chan_means, chan_stds = load_cnn_kan(
        model_dir / "best_model.pt", device, log)

    X_train_norm = normalise(X_train_raw, chan_means, chan_stds)
    X_val_norm   = normalise(X_val_raw,   chan_means, chan_stds)

    log.info("Running CNN+KAN inference (once, shared across all trials)...")
    train_pred_mmh, _ = get_kan_predictions(model, X_train_norm, device)
    val_pred_mmh,   _ = get_kan_predictions(model, X_val_norm,   device)

    # CNN+KAN baseline metrics
    kan_rmse = float(np.sqrt(np.mean((val_pred_mmh - y_val) ** 2)))
    kan_mae  = float(np.mean(np.abs(val_pred_mmh - y_val)))
    kan_r    = float(np.corrcoef(y_val, val_pred_mmh)[0, 1])
    log.info("CNN+KAN baseline — RMSE=%.4f  MAE=%.4f  r=%.4f",
             kan_rmse, kan_mae, kan_r)
    for thr in [25, 50, 75]:
        mask = y_val > thr
        if mask.sum() >= 5:
            rmse_thr = float(np.sqrt(np.mean((val_pred_mmh[mask] - y_val[mask]) ** 2)))
            bias_thr = float(np.mean(val_pred_mmh[mask] - y_val[mask]))
            log.info("  >%d mm/h : RMSE=%.4f  Bias=%+.4f  N=%d",
                     thr, rmse_thr, bias_thr, mask.sum())

    # ── Build objective ────────────────────────────────────────────────────
    objective = make_objective(
        X_train_raw, y_train, X_val_raw, y_val,
        X_train_norm, X_val_norm,
        train_pred_mmh, val_pred_mmh,
        rmse_weight=rmse_weight, mae_weight=mae_weight,
        heavy_weight_25=heavy_weight_25,
        heavy_weight_50=heavy_weight_50,
        heavy_weight_75=heavy_weight_75,
        mid_bias_weight=mid_bias_weight,
        n_estimators=n_estimators, early_stop=early_stop,
        log=log,
    )

    log.info("=" * 65)
    log.info("Starting v2 search: n_trials=%d", n_trials)
    log.info("Objective = %.2f*RMSE + %.2f*MAE + %.2f*h25 + %.2f*h50 + %.2f*h75 "
             "+ %.2f*MidBias + %.2f*TailUnder",
             rmse_weight, mae_weight,
             heavy_weight_25, heavy_weight_50, heavy_weight_75,
             mid_bias_weight, mid_bias_weight * 0.3)
    log.info("Features  = 37 (v2 expanded set)")
    log.info("Weights   = smooth exp curve: w(y) = w_min + (w_max-w_min)*expm1(g*y)/expm1(g*150)")
    log.info("=" * 65)

    # ── Create study ───────────────────────────────────────────────────────
    sampler = TPESampler(seed=42, multivariate=True)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=50)

    study = optuna.create_study(
        direction  = "minimize",
        sampler    = sampler,
        pruner     = pruner,
        study_name = "lgb_qpe_corrector_v2",
    )

    # Warm-start: reasonable starting point
    study.enqueue_trial({
        "w_min":              1.0,
        "w_max":             35.0,
        "gamma":              0.06,
        "use_huber":         True,
        "huber_delta":        0.64,
        "lr":                 0.0037,
        "num_leaves":        63,
        "max_depth":          6,
        "reg_alpha":          0.12,
        "reg_lambda":        18.8,
        "min_child_samples": 20,
        "subsample":          0.69,
        "colsample":          0.88,
    })

    # ── Callback ───────────────────────────────────────────────────────────
    def trial_callback(study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            log.info(
                "Trial %3d | score=%8.4f | RMSE=%7.4f | MAE=%7.4f | "
                "h25=%7.4f | h50=%7.4f | h75=%7.4f | "
                "mbias=%+5.2f | tunder=%5.2f | trees=%4d | best=%8.4f",
                trial.number, trial.value,
                trial.user_attrs.get("overall_rmse",  -1),
                trial.user_attrs.get("overall_mae",   -1),
                trial.user_attrs.get("heavy_rmse_25", -1),
                trial.user_attrs.get("heavy_rmse_50", -1),
                trial.user_attrs.get("heavy_rmse_75", -1),
                trial.user_attrs.get("mid_bias",      0.0),
                trial.user_attrs.get("tail_under",    0.0),
                trial.user_attrs.get("n_trees",       -1),
                study.best_value,
            )
        elif trial.state == optuna.trial.TrialState.PRUNED:
            log.info("Trial %3d | PRUNED", trial.number)

    # ── Run ────────────────────────────────────────────────────────────────
    study.optimize(
        objective,
        n_trials          = n_trials,
        callbacks         = [trial_callback],
        show_progress_bar = False,
    )

    # ── Results ────────────────────────────────────────────────────────────
    best = study.best_trial
    log.info("=" * 65)
    log.info("Search complete.")
    log.info("Best trial         : #%d", best.number)
    log.info("Best score         : %.4f", best.value)
    log.info("Best overall RMSE  : %.4f", best.user_attrs.get("overall_rmse",  -1))
    log.info("Best overall MAE   : %.4f", best.user_attrs.get("overall_mae",   -1))
    log.info("Best heavy RMSE 25 : %.4f", best.user_attrs.get("heavy_rmse_25", -1))
    log.info("Best heavy RMSE 50 : %.4f", best.user_attrs.get("heavy_rmse_50", -1))
    log.info("Best heavy RMSE 75 : %.4f", best.user_attrs.get("heavy_rmse_75", -1))
    log.info("Best n_trees       : %d",   best.user_attrs.get("n_trees",       -1))
    log.info("Best params:")
    for k, v in best.params.items():
        log.info("  %-22s : %s", k, v)

    # ── Save best params ───────────────────────────────────────────────────
    best_params = {
        **best.params,
        "n_estimators":        n_estimators,
        "early_stop":          early_stop,
        "use_dart":            False,
        "best_score":          round(best.value, 5),
        "best_overall_rmse":   best.user_attrs.get("overall_rmse",  -1),
        "best_overall_mae":    best.user_attrs.get("overall_mae",   -1),
        "best_heavy_rmse_25":  best.user_attrs.get("heavy_rmse_25", -1),
        "best_heavy_rmse_50":  best.user_attrs.get("heavy_rmse_50", -1),
        "best_heavy_rmse_75":  best.user_attrs.get("heavy_rmse_75", -1),
        "kan_baseline_rmse":   round(kan_rmse, 5),
        "kan_baseline_mae":    round(kan_mae,  5),
        "objective_weights": {
            "overall_rmse":  rmse_weight,
            "overall_mae":   mae_weight,
            "heavy_rmse_25": heavy_weight_25,
            "heavy_rmse_50": heavy_weight_50,
            "heavy_rmse_75": heavy_weight_75,
            "mid_bias":      mid_bias_weight,
        },
        "n_features":     len(FEATURE_NAMES),
        "feature_version": "v2",
        "search_pass":     1,
    }

    params_path = output_dir / "best_params_v2.json"
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    log.info("Best params saved : %s", params_path)

    study_path = output_dir / "optuna_study_v2.pkl"
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    log.info("Full study saved  : %s", study_path)

    # ── Top-10 table ───────────────────────────────────────────────────────
    completed = sorted(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: t.value)
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    log.info("")
    log.info("Top 10 trials:")
    log.info("  %4s  %8s  %8s  %8s  %8s  %8s  %8s  %7s",
             "Rank", "Score", "RMSE", "MAE", "H25", "H50", "H75", "Trees")
    for rank, t in enumerate(completed[:10], 1):
        log.info("  %4d  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f  %7d",
                 rank, t.value,
                 t.user_attrs.get("overall_rmse",  -1),
                 t.user_attrs.get("overall_mae",   -1),
                 t.user_attrs.get("heavy_rmse_25", -1),
                 t.user_attrs.get("heavy_rmse_50", -1),
                 t.user_attrs.get("heavy_rmse_75", -1),
                 t.user_attrs.get("n_trees",        -1))

    log.info("")
    log.info("Trials complete: %d  |  Pruned: %d  |  Total: %d",
             len(completed), len(pruned), len(study.trials))

    # ── Retrain command ────────────────────────────────────────────────────
    log.info("")
    log.info("To retrain with best params:")
    cmd = "  python train_lightgbm_v2.py"
    for k, v in best.params.items():
        if isinstance(v, bool):
            cmd += f" \\\n    --{'use_huber' if v else 'no-use_huber'}" if k == "use_huber" else ""
        else:
            cmd += f" \\\n    --{k} {v}"
    cmd += f" \\\n    --n_estimators {n_estimators}"
    cmd += f" \\\n    --early_stop {early_stop}"
    log.info(cmd)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Optuna tuning for LightGBM v2 (37 features).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_dir",   type=Path,
                   default=_ROOT / "data" / "dataset" / "processed_T4_improved")
    p.add_argument("--model_dir",  type=Path,
                   default=_ROOT / "models" / "qpe_cnn_kan")
    p.add_argument("--output_dir", type=Path,
                   default=_ROOT / "models" / "qpe_cnn_kan" / "optuna_v2")
    p.add_argument("--n_trials",   type=int,   default=60)

    # Objective weights
    p.add_argument("--rmse_weight",     type=float, default=0.20)
    p.add_argument("--mae_weight",      type=float, default=0.10)
    p.add_argument("--heavy_weight_25", type=float, default=0.25)
    p.add_argument("--heavy_weight_50", type=float, default=0.25)
    p.add_argument("--heavy_weight_75", type=float, default=0.20)
    p.add_argument("--mid_bias_weight", type=float, default=3.0)

    p.add_argument("--n_estimators", type=int, default=3000)
    p.add_argument("--early_stop",   type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(args.output_dir)
    log.info("LightGBM Optuna Tuning v2 (37 features) started.")
    log.info("Config: %s", vars(args))
    tune(
        data_dir        = args.data_dir,
        model_dir       = args.model_dir,
        output_dir      = args.output_dir,
        n_trials        = args.n_trials,
        rmse_weight     = args.rmse_weight,
        mae_weight      = args.mae_weight,
        heavy_weight_25 = args.heavy_weight_25,
        heavy_weight_50 = args.heavy_weight_50,
        heavy_weight_75 = args.heavy_weight_75,
        mid_bias_weight = args.mid_bias_weight,
        n_estimators    = args.n_estimators,
        early_stop      = args.early_stop,
        log             = log,
    )


if __name__ == "__main__":
    main()