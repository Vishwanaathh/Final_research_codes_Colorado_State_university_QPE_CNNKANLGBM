


"""
train_lightgbm_v2.py
=====================
Self-contained LightGBM residual corrector for the CNN+KAN QPE pipeline.
Version 2 — 37 features, smooth exponential sample weights, log1p residuals.

Changes vs train_lightgbm_corrector.py
---------------------------------------
1. Features expanded from 23 → 37 (see FEATURE_NAMES below).
   New features: Z_low_min, Z_low_skew, Z_low_kurt, ZDR_low_max,
   ZDR_low_skew, Z_high_conv_frac, Z_max_ratio, Z_mean_ratio, Z_high_min,
   Z_ZDR_consistency, Z_low_center_vs_mean, Z_high_center_vs_mean,
   Zratio_x_kan, ZDRmax_x_kan.

2. Sample weights use a smooth exponential curve instead of discrete step boosts:
       w(y) = w_min + (w_max - w_min) * expm1(gamma * y) / expm1(gamma * 150)
   Three searchable params (w_min, w_max, gamma) replace the old 4-threshold system.

3. Saves to lgb_residual_v2.pkl and lgb_metrics_v2.json to avoid overwriting v1.

Concept
-------
CNN+KAN predicts rain rate R_hat. LightGBM learns to predict the residual
in LOG1P space:
    epsilon_log = log1p(R_true) - log1p(R_hat)

Final corrected prediction:
    R_final = clamp(expm1(log1p(R_hat) + epsilon_hat), min=0)

Inference
---------
    epsilon_hat     = lgb_model.predict(features)
    log1p_corrected = log1p(R_hat) + epsilon_hat
    R_final         = clamp(expm1(log1p_corrected), min=0)

IMPORTANT: evaluate_all_v2.py must use apply_lgb_correction() — not mm/h addition.

Features (37 total)
--------------------
Center pixel (normalised):
    Z_low_norm, ZDR_low_norm, Z_high_norm, ZDR_high_norm

CNN+KAN prediction:
    kan_pred_mmh, kan_pred_log1p, kan_pred_sq

Vertical gradients:
    Z_diff, ZDR_diff

Raw center pixel:
    Z_low_raw, ZDR_low_raw

Low-Z neighbourhood:
    Z_low_std, Z_low_max, Z_low_min, Z_low_skew, Z_low_kurt, Z_conv_frac

Low-ZDR neighbourhood:
    ZDR_low_std, ZDR_low_mean, ZDR_low_max

High-Z neighbourhood:
    Z_high_std, Z_high_max, Z_high_conv_frac

High-ZDR neighbourhood:
    ZDR_high_std

Vertical profile shape:
    Z_max_ratio, Z_mean_ratio, Z_high_min

Z-ZDR consistency:
    ZDR_low_skew, Z_ZDR_consistency, Z_low_center_vs_mean

High-elevation center prominence:
    Z_high_center_vs_mean

Feature interactions:
    Z_x_kan, Zdiff_x_kan, Zhigh_x_kan, ZDR_x_kan, Zratio_x_kan, ZDRmax_x_kan
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
import time

import numpy as np
from scipy.stats import skew, kurtosis
import torch
import torch.nn as nn
import lightgbm as lgb

try:
    from cnn_feature_extractor import RadarCNN
    from kan_rainfall_predictor import RainfallKAN
except ImportError as e:
    sys.exit(f"[FATAL] Could not import from {_AI_SCRIPT_DIR}\n{e}")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(model_dir: Path) -> logging.Logger:
    log = logging.getLogger("lgb_v2")
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    log.addHandler(ch)
    fh = logging.FileHandler(model_dir / "lgb_v2_training_log.txt", mode="w")
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
        self.kan = RainfallKAN(
            input_dim=32, hidden1=64, hidden2=32,
            grid_size=5, spline_order=3,
        )

    def forward(self, x):
        return self.kan(self.cnn(x))


def load_cnn_kan(model_path: Path, device: torch.device, log: logging.Logger):
    ckpt       = torch.load(model_path, map_location=device)
    chan_means  = np.array(ckpt["chan_means"], dtype=np.float32)
    chan_stds   = np.array(ckpt["chan_stds"],  dtype=np.float32)
    model       = QPEModel().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    log.info("Loaded CNN+KAN  epoch=%d  val_loss=%.5f",
             ckpt.get("epoch", -1), ckpt.get("val_loss", -1))
    m = ckpt.get("metrics", {})
    if m:
        log.info("  RMSE=%.3f  r=%.3f", m.get("RMSE", -1), m.get("Correlation", -1))
    return model, chan_means, chan_stds


def normalise(X: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32).copy()
    for c in range(4):
        X[:, c] = (X[:, c] - means[c]) / (stds[c] + 1e-8)
    return X


@torch.no_grad()
def get_kan_predictions(model: nn.Module, X_norm: np.ndarray,
                         device: torch.device, batch: int = 512):
    preds = []
    for s in range(0, len(X_norm), batch):
        x_t = torch.from_numpy(X_norm[s:s+batch]).to(device)
        preds.append(model(x_t).squeeze(1).cpu().numpy())
    pred_log1p = np.concatenate(preds)
    pred_mmh   = np.clip(np.expm1(pred_log1p), 0.0, None)
    return pred_mmh, pred_log1p


# ---------------------------------------------------------------------------
# Inference helper — SINGLE SOURCE OF TRUTH
# ---------------------------------------------------------------------------

def apply_lgb_correction(pred_mmh: np.ndarray,
                          epsilon_log: np.ndarray) -> np.ndarray:
    """
    R_final = clamp(expm1(log1p(R_hat) + epsilon_log), min=0)
    Must be used identically in training eval and inference scripts.
    """
    log1p_corrected = np.log1p(np.clip(pred_mmh, 0.0, None)) + epsilon_log
    return np.clip(np.expm1(log1p_corrected), 0.0, None)


# ---------------------------------------------------------------------------
# Features (37 total) — self-contained, no external import needed
# ---------------------------------------------------------------------------

CENTER = 4   # row=4, col=4 of the 9x9 patch — actual center pixel

FEATURE_NAMES = [
    # [0-3]  normalised center pixel
    "Z_low_norm", "ZDR_low_norm", "Z_high_norm", "ZDR_high_norm",
    # [4-6]  KAN prediction
    "kan_pred_mmh", "kan_pred_log1p", "kan_pred_sq",
    # [7-8]  vertical gradients
    "Z_diff", "ZDR_diff",
    # [9-10] raw center pixel
    "Z_low_raw", "ZDR_low_raw",
    # [11-16] low-Z neighbourhood
    "Z_low_std", "Z_low_max", "Z_low_min", "Z_low_skew", "Z_low_kurt",
    "Z_conv_frac",
    # [17-19] low-ZDR neighbourhood
    "ZDR_low_std", "ZDR_low_mean", "ZDR_low_max",
    # [20-22] high-Z neighbourhood
    "Z_high_std", "Z_high_max", "Z_high_conv_frac",
    # [23]   high-ZDR neighbourhood
    "ZDR_high_std",
    # [24-26] vertical profile shape
    "Z_max_ratio", "Z_mean_ratio", "Z_high_min",
    # [27-29] Z-ZDR consistency
    "ZDR_low_skew", "Z_ZDR_consistency", "Z_low_center_vs_mean",
    # [30]   high-elevation center prominence
    "Z_high_center_vs_mean",
    # [31-34] original interactions
    "Z_x_kan", "Zdiff_x_kan", "Zhigh_x_kan", "ZDR_x_kan",
    # [35-36] new interactions
    "Zratio_x_kan", "ZDRmax_x_kan",
]

# Only KAN prediction features are monotone-constrained.
# All interaction and new features are unconstrained.
MONOTONE_CONSTRAINTS = [
    0, 0, 0, 0,   # normalised center pixel
    1, 1, 1,      # KAN prediction — MONOTONE
    0, 0,         # vertical gradients
    0, 0,         # raw center pixel
    0, 0, 0, 0, 0, 0,  # low-Z neighbourhood
    0, 0, 0,      # low-ZDR neighbourhood
    0, 0, 0,      # high-Z neighbourhood
    0,            # high-ZDR neighbourhood
    0, 0, 0,      # vertical profile shape
    0, 0, 0,      # Z-ZDR consistency
    0,            # high-elevation center prominence
    0, 0, 0, 0,   # original interactions
    0, 0,         # new interactions
]

assert len(MONOTONE_CONSTRAINTS) == len(FEATURE_NAMES), (
    f"Mismatch: {len(MONOTONE_CONSTRAINTS)} constraints vs {len(FEATURE_NAMES)} features"
)


def build_features(X_raw: np.ndarray,
                   X_norm: np.ndarray,
                   pred_mmh: np.ndarray,
                   pred_log1p=None) -> np.ndarray:
    """
    Build (N, 37) feature matrix.
    pred_log1p is accepted for API compatibility but not used.
    """
    N = len(X_raw)

    # Center pixel
    Z_l   = X_raw[:, 0, CENTER, CENTER]
    ZDR_l = X_raw[:, 1, CENTER, CENTER]
    Z_h   = X_raw[:, 2, CENTER, CENTER]
    ZDR_h = X_raw[:, 3, CENTER, CENTER]

    Z_diff   = (Z_l   - Z_h).astype(np.float32)
    ZDR_diff = (ZDR_l - ZDR_h).astype(np.float32)

    # Flatten windows
    Z_win        = X_raw[:, 0].reshape(N, -1)
    ZDR_win      = X_raw[:, 1].reshape(N, -1)
    Z_high_win   = X_raw[:, 2].reshape(N, -1)
    ZDR_high_win = X_raw[:, 3].reshape(N, -1)

    # Low-Z neighbourhood
    Z_low_std    = Z_win.std(axis=1).astype(np.float32)
    Z_low_max    = Z_win.max(axis=1).astype(np.float32)
    Z_low_min    = Z_win.min(axis=1).astype(np.float32)
    Z_low_mean   = Z_win.mean(axis=1).astype(np.float32)
    Z_conv_frac  = (Z_win > 40.0).mean(axis=1).astype(np.float32)
    Z_low_skew   = skew(Z_win,     axis=1).astype(np.float32)
    Z_low_kurt   = kurtosis(Z_win, axis=1).astype(np.float32)

    # Low-ZDR neighbourhood
    ZDR_low_std  = ZDR_win.std(axis=1).astype(np.float32)
    ZDR_low_mean = ZDR_win.mean(axis=1).astype(np.float32)
    ZDR_low_max  = ZDR_win.max(axis=1).astype(np.float32)
    ZDR_low_skew = skew(ZDR_win, axis=1).astype(np.float32)

    # High-Z neighbourhood
    Z_high_std       = Z_high_win.std(axis=1).astype(np.float32)
    Z_high_max       = Z_high_win.max(axis=1).astype(np.float32)
    Z_high_min       = Z_high_win.min(axis=1).astype(np.float32)
    Z_high_mean      = Z_high_win.mean(axis=1).astype(np.float32)
    Z_high_conv_frac = (Z_high_win > 35.0).mean(axis=1).astype(np.float32)

    # High-ZDR neighbourhood
    ZDR_high_std = ZDR_high_win.std(axis=1).astype(np.float32)

    # Vertical profile shape
    Z_max_ratio  = (Z_high_max  / (Z_low_max  + 1e-8)).astype(np.float32)
    Z_mean_ratio = (Z_high_mean / (Z_low_mean + 1e-8)).astype(np.float32)

    # Z-ZDR consistency
    ZDR_l_pos             = np.clip(ZDR_l, 0.0, None)
    Z_ZDR_consistency     = (Z_l - 20.0 * np.log10(ZDR_l_pos + 1.0)).astype(np.float32)
    Z_low_center_vs_mean  = (Z_l - Z_low_mean).astype(np.float32)
    Z_high_center_vs_mean = (Z_h - Z_high_mean).astype(np.float32)

    # KAN prediction features
    pred_mmh_pos   = np.clip(pred_mmh, 0.0, None).astype(np.float32)
    pred_log1p_pos = np.log1p(pred_mmh_pos)
    kan_pred_sq    = (pred_mmh_pos ** 2).astype(np.float32)

    # Interactions
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
        Z_diff, ZDR_diff,
        Z_l, ZDR_l,
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
# Sample weights — smooth exponential curve
# ---------------------------------------------------------------------------

def compute_sample_weights(y_true: np.ndarray,
                            w_min:  float = 1.0,
                            w_max:  float = 30.0,
                            gamma:  float = 0.05) -> np.ndarray:
    """
    w(y) = w_min + (w_max - w_min) * expm1(gamma * y) / expm1(gamma * 150)

    Every rain rate gets its own continuously-varying weight.
    Optuna searches w_min, w_max, gamma to find the optimal curve shape.
    """
    y      = np.clip(y_true, 0.0, None).astype(np.float64)
    y_max  = 150.0
    denom  = np.expm1(gamma * y_max)
    if denom < 1e-8:
        w = w_min + (w_max - w_min) * (y / y_max)
    else:
        w = w_min + (w_max - w_min) * np.expm1(gamma * y) / denom
    return w.astype(np.float32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    thresholds=(1.0, 5.0, 10.0, 25.0, 50.0)) -> dict:
    res    = y_pred - y_true
    ss_res = np.sum(res ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    metrics = {
        "RMSE":        float(np.sqrt(np.mean(res ** 2))),
        "MAE":         float(np.mean(np.abs(res))),
        "Bias":        float(np.mean(res)),
        "Correlation": float(np.corrcoef(y_true, y_pred)[0, 1]),
        "R2":          float(1 - ss_res / (ss_tot + 1e-8)),
    }
    for thr in thresholds:
        obs_p  = y_true >= thr
        pred_p = y_pred >= thr
        tp = float(np.sum(obs_p & pred_p))
        fp = float(np.sum(~obs_p & pred_p))
        fn = float(np.sum(obs_p & ~pred_p))
        metrics[f"CSI@{thr:.0f}"] = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return metrics


def log_extreme_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                         name: str, log: logging.Logger,
                         thresholds=(25.0, 50.0, 75.0, 100.0)) -> None:
    for thr in thresholds:
        mask = y_true >= thr
        n    = mask.sum()
        if n < 5:
            log.info("  %s @ >%3dmm/h : N=%d (too few)", name, thr, n)
            continue
        rmse = float(np.sqrt(np.mean((y_pred[mask] - y_true[mask]) ** 2)))
        bias = float(np.mean(y_pred[mask] - y_true[mask]))
        log.info("  %s @ >%3dmm/h : N=%4d  RMSE=%.2f  Bias=%+.2f",
                 name, thr, n, rmse, bias)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def compute_constant_offset_baseline(y_val, val_pred_mmh, train_resid_log_mean):
    val_const = apply_lgb_correction(
        val_pred_mmh,
        np.full(len(val_pred_mmh), train_resid_log_mean, dtype=np.float32))
    return val_const, compute_metrics(y_val, val_const)


def bootstrap_metric_deltas(y_val, val_pred_mmh, val_lgb, val_const,
                              n_bootstrap, seed, log):
    rng = np.random.default_rng(seed)
    n   = len(y_val)

    d_rmse_lgb   = np.empty(n_bootstrap)
    d_mae_lgb    = np.empty(n_bootstrap)
    d_rmse_const = np.empty(n_bootstrap)
    d_mae_const  = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        idx             = rng.integers(0, n, size=n)
        yt, yk, yl, yc  = y_val[idx], val_pred_mmh[idx], val_lgb[idx], val_const[idx]
        rmse_kan        = np.sqrt(np.mean((yk - yt) ** 2))
        mae_kan         = np.mean(np.abs(yk - yt))
        d_rmse_lgb[i]   = np.sqrt(np.mean((yl - yt) ** 2)) - rmse_kan
        d_mae_lgb[i]    = np.mean(np.abs(yl - yt))          - mae_kan
        d_rmse_const[i] = np.sqrt(np.mean((yc - yt) ** 2)) - rmse_kan
        d_mae_const[i]  = np.mean(np.abs(yc - yt))          - mae_kan

    def pcts(arr):
        p2_5, p50, p97_5 = np.percentile(arr, [2.5, 50, 97.5])
        return float(p2_5), float(p50), float(p97_5)

    results = {
        "rmse_lgb_minus_kan":   pcts(d_rmse_lgb),
        "mae_lgb_minus_kan":    pcts(d_mae_lgb),
        "rmse_const_minus_kan": pcts(d_rmse_const),
        "mae_const_minus_kan":  pcts(d_mae_const),
    }

    log.info("=" * 65)
    log.info("Bootstrap CIs (N=%d, negative = improvement over CNN+KAN):",
             n_bootstrap)
    for name, (lo, mid, hi) in results.items():
        sig = ("SIGNIFICANT" if hi < 0
               else "REGRESSION" if lo > 0
               else "not significant (CI spans 0)")
        log.info("  %-22s : [%+.4f, %+.4f, %+.4f]  -- %s",
                 name, lo, mid, hi, sig)
    return results


def compute_residual_correlation(val_resid_log, val_resid_log_pred, log):
    corr = float(np.corrcoef(val_resid_log, val_resid_log_pred)[0, 1])
    log.info("=" * 65)
    log.info("corr(val_resid_log_pred, val_resid_log) = %.4f", corr)
    if corr < 0.15:
        log.info("  -> LOW")
    elif corr < 0.30:
        log.info("  -> MODERATE")
    else:
        log.info("  -> STRONG")
    return corr


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_corrector(data_dir, model_dir, n_estimators, lr, max_depth,
                    num_leaves, subsample, colsample, reg_alpha, reg_lambda,
                    min_child_samples, early_stop,
                    w_min, w_max, gamma,
                    use_dart, use_huber, huber_delta,
                    n_bootstrap, bootstrap_seed, log):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Load data ──────────────────────────────────────────────────────────
    log.info("Loading dataset from %s", data_dir)
    X_train_raw = np.load(data_dir / "X_train.npy")
    y_train     = np.load(data_dir / "y_train.npy")
    X_val_raw   = np.load(data_dir / "X_val.npy")
    y_val       = np.load(data_dir / "y_val.npy")
    log.info("Train: %d  Val: %d", len(y_train), len(y_val))
    log.info("Train y: min=%.2f  max=%.2f  mean=%.2f mm/h",
             y_train.min(), y_train.max(), y_train.mean())
    for thr in [25, 50, 75]:
        log.info("  Train >%dmm/h: %d (%.1f%%)",
                 thr, (y_train > thr).sum(), 100*(y_train > thr).mean())

    # ── Load CNN+KAN ───────────────────────────────────────────────────────
    model, chan_means, chan_stds = load_cnn_kan(
        model_dir / "best_model.pt", device, log)

    X_train_norm = normalise(X_train_raw, chan_means, chan_stds)
    X_val_norm   = normalise(X_val_raw,   chan_means, chan_stds)

    log.info("Running CNN+KAN inference...")
    train_pred_mmh, train_pred_log1p = get_kan_predictions(model, X_train_norm, device)
    val_pred_mmh,   val_pred_log1p   = get_kan_predictions(model, X_val_norm,   device)

    # ── CNN+KAN baseline ───────────────────────────────────────────────────
    log.info("=" * 65)
    log.info("CNN+KAN baseline:")
    kan_metrics = compute_metrics(y_val, val_pred_mmh)
    for k, v in kan_metrics.items():
        log.info("  %-15s : %.4f", k, v)
    log.info("  Extreme performance:")
    log_extreme_metrics(y_val, val_pred_mmh, "CNN+KAN", log)

    # ── Log1p residuals ────────────────────────────────────────────────────
    y_train_log1p        = np.log1p(np.clip(y_train, 0.0, None)).astype(np.float32)
    y_val_log1p          = np.log1p(np.clip(y_val,   0.0, None)).astype(np.float32)
    train_pred_log1p_pos = np.log1p(np.clip(train_pred_mmh, 0.0, None)).astype(np.float32)
    val_pred_log1p_pos   = np.log1p(np.clip(val_pred_mmh,   0.0, None)).astype(np.float32)
    train_resid_log      = (y_train_log1p - train_pred_log1p_pos).astype(np.float32)
    val_resid_log        = (y_val_log1p   - val_pred_log1p_pos  ).astype(np.float32)

    log.info("=" * 65)
    log.info("Log1p residual stats (train):")
    log.info("  mean=%+.4f  std=%.4f  min=%+.4f  max=%+.4f",
             train_resid_log.mean(), train_resid_log.std(),
             train_resid_log.min(),  train_resid_log.max())
    for thr in [0, 25, 50, 75]:
        mask = y_train >= thr
        if mask.sum() > 5:
            log.info("  y_train >= %3d: mean=%+.4f  std=%.4f  N=%d",
                     thr, train_resid_log[mask].mean(),
                     train_resid_log[mask].std(), mask.sum())

    # ── Features ───────────────────────────────────────────────────────────
    log.info("Building feature matrices (%d features)...", len(FEATURE_NAMES))
    X_lgb_train = build_features(X_train_raw, X_train_norm, train_pred_mmh)
    X_lgb_val   = build_features(X_val_raw,   X_val_norm,   val_pred_mmh)
    log.info("  Train: %s  Val: %s", X_lgb_train.shape, X_lgb_val.shape)

    # Verify no NaNs
    assert not np.any(np.isnan(X_lgb_train)), "NaNs in train features!"
    assert not np.any(np.isnan(X_lgb_val)),   "NaNs in val features!"

    # ── Sample weights ─────────────────────────────────────────────────────
    weights = compute_sample_weights(y_train, w_min, w_max, gamma)
    log.info("Sample weights (smooth exp): w_min=%.2f  w_max=%.2f  gamma=%.4f",
             w_min, w_max, gamma)
    log.info("  range=[%.2f, %.2f]  median=%.2f",
             weights.min(), weights.max(), np.median(weights))
    for thr in [25, 50, 75]:
        mask = y_train > thr
        if mask.any():
            log.info("  Mean weight >%dmm/h: %.2f  (N=%d)",
                     thr, weights[mask].mean(), mask.sum())

    # ── Objective ──────────────────────────────────────────────────────────
    if use_huber:
        objective = "huber"
        alpha     = huber_delta
        metric    = "huber"
        obj_desc  = f"Huber(delta={huber_delta:.4f})"
    else:
        objective = "regression_l2"
        alpha     = 0.9
        metric    = "mse"
        obj_desc  = "MSE (L2)"

    log.info("=" * 65)
    log.info("Objective    : %s", obj_desc)
    log.info("Target space : LOG1P  epsilon = log1p(R_true) - log1p(R_hat)")
    log.info("Inference    : R_final = expm1(log1p(R_hat) + epsilon_hat)")
    log.info("Features     : %d  (v2 — expanded from 23)", len(FEATURE_NAMES))
    log.info("LightGBM     : n_est=%d  lr=%.5f  leaves=%d  depth=%d  "
             "min_child=%d  early_stop=%d",
             n_estimators, lr, num_leaves, max_depth, min_child_samples, early_stop)
    log.info("Regularise   : alpha=%.4f  lambda=%.4f  sub=%.3f  col=%.3f",
             reg_alpha, reg_lambda, subsample, colsample)
    log.info("=" * 65)

    lgb_train  = lgb.Dataset(X_lgb_train, label=train_resid_log,
                              weight=weights, feature_name=FEATURE_NAMES)
    lgb_val_ds = lgb.Dataset(X_lgb_val,   label=val_resid_log,
                              reference=lgb_train, feature_name=FEATURE_NAMES)

    params = {
        "boosting_type":        "dart" if use_dart else "gbdt",
        "objective":            objective,
        "alpha":                alpha,
        "metric":               metric,
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

    t0 = time.time()
    if use_dart:
        log.info("Training DART (no early stopping, n_estimators=%d)...", n_estimators)
        lgb_model = lgb.train(
            params, lgb_train,
            num_boost_round=n_estimators,
            callbacks=[lgb.log_evaluation(period=50)])
    else:
        log.info("Training GBDT + early stopping (patience=%d)...", early_stop)
        lgb_model = lgb.train(
            params, lgb_train,
            num_boost_round=n_estimators,
            valid_sets=[lgb_val_ds],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stop, verbose=True),
                lgb.log_evaluation(period=25),
            ])
    log.info("Done in %.1fs  best_iter=%d",
             time.time() - t0,
             lgb_model.best_iteration if not use_dart else n_estimators)

    # ── Evaluate ───────────────────────────────────────────────────────────
    num_iter = (lgb_model.best_iteration
                if hasattr(lgb_model, "best_iteration") and lgb_model.best_iteration > 0
                else lgb_model.num_trees())

    val_epsilon_log = lgb_model.predict(X_lgb_val, num_iteration=num_iter)
    val_final       = apply_lgb_correction(val_pred_mmh, val_epsilon_log)

    log.info("=" * 65)
    log.info("Final pipeline (CNN+KAN + LightGBM v2):")
    final_metrics = compute_metrics(y_val, val_final)
    for k, v in final_metrics.items():
        log.info("  %-15s : %.4f", k, v)
    log.info("  Extreme performance:")
    log_extreme_metrics(y_val, val_final, "Pipeline v2", log)

    log.info("=" * 65)
    log.info("Improvement over CNN+KAN:")
    for m in ["RMSE", "MAE", "Bias", "Correlation", "R2"]:
        b, a = kan_metrics[m], final_metrics[m]
        log.info("  %-13s : %.4f -> %.4f  (%+.4f)", m, b, a, a - b)

    # ── Diagnostics ────────────────────────────────────────────────────────
    train_resid_log_mean = float(train_resid_log.mean())
    val_const, const_metrics = compute_constant_offset_baseline(
        y_val, val_pred_mmh, train_resid_log_mean)

    log.info("=" * 65)
    log.info("Constant-offset baseline (mean resid = %+.4f):", train_resid_log_mean)
    for k, v in const_metrics.items():
        log.info("  %-15s : %.4f", k, v)
    log.info("LightGBM vs constant-offset:")
    for m in ["RMSE", "MAE", "Bias"]:
        c, a = const_metrics[m], final_metrics[m]
        log.info("  %-13s : const=%.4f  lgb=%.4f  (%+.4f)", m, c, a, a - c)

    bootstrap_results = bootstrap_metric_deltas(
        y_val, val_pred_mmh, val_final, val_const,
        n_bootstrap, bootstrap_seed, log)

    resid_corr = compute_residual_correlation(val_resid_log, val_epsilon_log, log)

    # ── Verdict ────────────────────────────────────────────────────────────
    rmse_ci       = bootstrap_results["rmse_lgb_minus_kan"]
    mae_ci        = bootstrap_results["mae_lgb_minus_kan"]
    rmse_sig      = rmse_ci[2] < 0
    mae_sig       = mae_ci[2]  < 0
    beats_const   = (final_metrics["RMSE"] < const_metrics["RMSE"] and
                     final_metrics["MAE"]  < const_metrics["MAE"])

    log.info("=" * 65)
    log.info("DIAGNOSTIC VERDICT:")
    if rmse_sig and mae_sig and beats_const and resid_corr >= 0.15:
        verdict = (f"STRONG: both RMSE and MAE CIs entirely below zero, "
                   f"beats constant-offset baseline, resid_corr={resid_corr:.3f}.")
    elif beats_const and (rmse_sig or mae_sig):
        verdict = (f"PARTIAL: beats constant-offset, one metric significant. "
                   f"resid_corr={resid_corr:.3f}.")
    else:
        verdict = (f"WEAK/NONE: CI spans zero or doesn't beat constant-offset. "
                   f"resid_corr={resid_corr:.3f}.")
    log.info("  %s", verdict)

    # ── Feature importance ─────────────────────────────────────────────────
    importance = dict(sorted(
        {n: int(s) for n, s in zip(
            FEATURE_NAMES,
            lgb_model.feature_importance(importance_type="gain"))}.items(),
        key=lambda x: x[1], reverse=True))
    log.info("Feature importance (gain):")
    for name, score in importance.items():
        log.info("  %-28s : %d", name, score)

    # ── Save ───────────────────────────────────────────────────────────────
    with open(model_dir / "lgb_residual_v2.pkl", "wb") as f:
        pickle.dump(lgb_model, f)

    metrics_out = {
        "version":          "v2",
        "n_features":       len(FEATURE_NAMES),
        "feature_names":    FEATURE_NAMES,
        "residual_space":   "log1p",
        "kan_baseline":     {k: round(v, 5) for k, v in kan_metrics.items()},
        "final_pipeline":   {k: round(v, 5) for k, v in final_metrics.items()},
        "constant_offset_baseline": {
            "train_resid_log_mean": round(train_resid_log_mean, 5),
            **{k: round(v, 5) for k, v in const_metrics.items()},
        },
        "improvement": {
            k: round(final_metrics[k] - kan_metrics[k], 5)
            for k in ["RMSE", "MAE", "Bias", "Correlation", "R2"]
        },
        "diagnostics": {
            "n_bootstrap":          n_bootstrap,
            "bootstrap_seed":       bootstrap_seed,
            "bootstrap_ci":         {
                k: {"p2.5": v[0], "p50": v[1], "p97.5": v[2]}
                for k, v in bootstrap_results.items()
            },
            "residual_correlation": round(resid_corr, 5),
            "verdict":              verdict,
        },
        "lgb_num_trees":        num_iter,
        "feature_importance":   importance,
        "monotone_constraints": MONOTONE_CONSTRAINTS,
        "boosting_type":        "dart" if use_dart else "gbdt",
        "objective":            obj_desc,
        "chan_means":           chan_means.tolist(),
        "chan_stds":            chan_stds.tolist(),
        "hyperparameters": {
            "n_estimators":      n_estimators,
            "lr":                lr,
            "max_depth":         max_depth,
            "num_leaves":        num_leaves,
            "subsample":         subsample,
            "colsample":         colsample,
            "reg_alpha":         reg_alpha,
            "reg_lambda":        reg_lambda,
            "min_child_samples": min_child_samples,
            "early_stop":        early_stop,
            "w_min":             w_min,
            "w_max":             w_max,
            "gamma":             gamma,
            "huber_delta":       huber_delta if use_huber else None,
        },
    }
    with open(model_dir / "lgb_metrics_v2.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    log.info("Saved: lgb_residual_v2.pkl  lgb_metrics_v2.json")
    log.info("Inference: epsilon = lgb.predict(features_v2)")
    log.info("           R_final = expm1(log1p(R_hat) + epsilon)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train LightGBM corrector v2 — 37 features, log1p residuals.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_dir",   type=Path,
                   default=_ROOT / "data" / "dataset" / "processed_T4_improved")
    p.add_argument("--model_dir",  type=Path,
                   default=_ROOT / "models" / "qpe_cnn_kan")

    # LightGBM hyperparameters — start from v10 Optuna best, re-tune after
    p.add_argument("--n_estimators",      type=int,   default=3000)
    p.add_argument("--lr",                type=float, default=0.0007869319255137371)
    p.add_argument("--max_depth",         type=int,   default=9)
    p.add_argument("--num_leaves",        type=int,   default=97)
    p.add_argument("--subsample",         type=float, default=0.8607309805384873)
    p.add_argument("--colsample",         type=float, default=0.8193162191494474)
    p.add_argument("--reg_alpha",         type=float, default=0.08734311987235173)
    p.add_argument("--reg_lambda",        type=float, default=9.891014204021735)
    p.add_argument("--min_child_samples", type=int,   default=25)
    p.add_argument("--early_stop",        type=int,   default=200)

    # Smooth exponential weight curve
    p.add_argument("--w_min",  type=float, default=0.8580321312112523,
                   help="Minimum sample weight (at y=0 mm/h).")
    p.add_argument("--w_max",  type=float, default=62.73994948705331,
                   help="Maximum sample weight (at y=150 mm/h).")
    p.add_argument("--gamma",  type=float, default=0.06544525565826403,
                   help="Exponential growth rate of weight curve.")

    p.add_argument("--use_huber", default=True,
                   action=argparse.BooleanOptionalAction,
                   help="Use Huber loss (recommended).")
    p.add_argument("--huber_delta", type=float, default=0.31857279994526255)
    p.add_argument("--use_dart",    default=False,
                   action=argparse.BooleanOptionalAction)
    p.add_argument("--n_bootstrap",    type=int, default=1000)
    p.add_argument("--bootstrap_seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(args.model_dir)
    log.info("LightGBM Corrector v2 (37 features) started.")
    log.info("Config: %s", vars(args))
    train_corrector(
        data_dir          = args.data_dir,
        model_dir         = args.model_dir,
        n_estimators      = args.n_estimators,
        lr                = args.lr,
        max_depth         = args.max_depth,
        num_leaves        = args.num_leaves,
        subsample         = args.subsample,
        colsample         = args.colsample,
        reg_alpha         = args.reg_alpha,
        reg_lambda        = args.reg_lambda,
        min_child_samples = args.min_child_samples,
        early_stop        = args.early_stop,
        w_min             = args.w_min,
        w_max             = args.w_max,
        gamma             = args.gamma,
        use_dart          = args.use_dart,
        use_huber         = args.use_huber,
        huber_delta       = args.huber_delta,
        n_bootstrap       = args.n_bootstrap,
        bootstrap_seed    = args.bootstrap_seed,
        log               = log,
    )


if __name__ == "__main__":
    main()