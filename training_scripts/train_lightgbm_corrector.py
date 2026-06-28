"""
train_lightgbm_corrector.py
============================
LightGBM residual corrector for the CNN+KAN QPE pipeline.

Concept
-------
The CNN+KAN predicts rain rate R_hat from radar spatial features.
LightGBM learns to predict the residual in LOG1P space:

    epsilon_log = log1p(R_true) - log1p(R_hat)

Final corrected prediction:
    R_final = clamp(expm1(log1p(R_hat) + epsilon_hat), min=0)

Why log1p-space residuals?
--------------------------
In mm/h space, residuals range from ~0.8 mm/h at 5 mm/h true
to ~45 mm/h at 120 mm/h true — a 56x range. LightGBM's gradient
signal is dominated by light-rain samples regardless of weighting,
because 79% of total weighted gradient still comes from y < 25 mm/h.
Sample weights scale the gradient but do NOT change leaf membership:
with min_child_samples=31 and ~390 samples above 50 mm/h, the tree
can make at most 12 splits on the extreme tail no matter how high
the weights are.

In log1p space, the same residuals are ~0.13 to ~0.47 — a 3.6x
range. LightGBM sees a near-uniform target distribution across the
full rain rate spectrum. No aggressive weighting is needed. The tree
can use its full split budget to learn the systematic compression
bias that the CNN+KAN introduces at high rain rates.

This also explains why earlier mm/h-space attempts with aggressive
weighting, constraint relaxation, and upsampling all failed — they
were fighting the symptom (gradient imbalance) rather than the cause
(wrong target space).

Monotonicity constraints (v4)
------------------------------
Only pure KAN output features remain monotone-constrained.
Interaction features are unconstrained to allow aggressive upward
corrections at high Z.

    +1: kan_pred_mmh, kan_pred_log1p, kan_pred_sq
    0 : everything else including Z_x_kan, Zdiff_x_kan,
        Zhigh_x_kan, ZDR_x_kan

Inference
---------
    epsilon_hat  = lgb_model.predict(features)           # log1p space
    log1p_corrected = log1p(R_hat) + epsilon_hat
    R_final      = clamp(expm1(log1p_corrected), min=0)

IMPORTANT: evaluate_all.py and plot scripts must use this same
inverse transform at inference, not the old mm/h addition.

Features (23 total)
-------------------
Center pixel:
    Z_low_norm, ZDR_low_norm, Z_high_norm, ZDR_high_norm

CNN+KAN prediction:
    kan_pred_mmh, kan_pred_log1p, kan_pred_sq

Vertical gradients:
    Z_diff (Z_low - Z_high), ZDR_diff

Raw values:
    Z_low_raw, ZDR_low_raw

Neighbourhood statistics:
    Z_low_std, Z_low_max, Z_conv_frac
    ZDR_low_std, ZDR_low_mean
    Z_high_std, Z_high_max
    ZDR_high_std

Feature interactions (unconstrained):
    Z_x_kan, Zdiff_x_kan, Zhigh_x_kan, ZDR_x_kan
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
from typing import Dict, List, Tuple

import numpy as np
import torch
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
    log = logging.getLogger("lgb_corrector")
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    log.addHandler(ch)
    fh = logging.FileHandler(model_dir / "lgb_training_log.txt", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log


# ---------------------------------------------------------------------------
# QPEModel
# ---------------------------------------------------------------------------

class QPEModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cnn = RadarCNN(in_channels=4, feature_dim=32)
        self.kan = RainfallKAN(
            input_dim=32, hidden1=64, hidden2=32,
            grid_size=5, spline_order=3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kan(self.cnn(x))


# ---------------------------------------------------------------------------
# Load CNN+KAN
# ---------------------------------------------------------------------------

def load_cnn_kan(model_path, device, log):
    ckpt      = torch.load(model_path, map_location=device)
    chan_means = np.array(ckpt["chan_means"], dtype=np.float32)
    chan_stds  = np.array(ckpt["chan_stds"],  dtype=np.float32)
    model      = QPEModel().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    log.info("Loaded CNN+KAN  epoch=%d  val_loss=%.5f",
             ckpt.get("epoch", -1), ckpt.get("val_loss", -1))
    m = ckpt.get("metrics", {})
    if m:
        log.info("  RMSE=%.3f  r=%.3f", m.get("RMSE", -1), m.get("Correlation", -1))
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
    pred_mmh   = np.clip(np.expm1(pred_log1p), 0.0, None)
    return pred_mmh, pred_log1p


# ---------------------------------------------------------------------------
# Log1p-space inference helper  ← SINGLE SOURCE OF TRUTH
# ---------------------------------------------------------------------------

def apply_lgb_correction(pred_mmh: np.ndarray,
                          epsilon_log: np.ndarray) -> np.ndarray:
    """
    Apply log1p-space residual correction.

    epsilon_log = LightGBM predicted residual in log1p space
    R_final = clamp(expm1(log1p(R_hat) + epsilon_log), min=0)

    This function is the SINGLE SOURCE OF TRUTH for the inverse
    transform. Both training evaluation and inference scripts must
    use this function (not mm/h addition) to stay consistent.
    """
    log1p_corrected = np.log1p(np.clip(pred_mmh, 0.0, None)) + epsilon_log
    return np.clip(np.expm1(log1p_corrected), 0.0, None)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

CENTER = 4

FEATURE_NAMES = [
    "Z_low_norm", "ZDR_low_norm", "Z_high_norm", "ZDR_high_norm",
    "kan_pred_mmh", "kan_pred_log1p", "kan_pred_sq",
    "Z_diff", "ZDR_diff",
    "Z_low_raw", "ZDR_low_raw",
    "Z_low_std", "Z_low_max", "Z_conv_frac",
    "ZDR_low_std", "ZDR_low_mean",
    "Z_high_std", "Z_high_max",
    "ZDR_high_std",
    "Z_x_kan",
    "Zdiff_x_kan",
    "Zhigh_x_kan",
    "ZDR_x_kan",
]

# v4: only KAN output features monotone-constrained.
# Interaction features unconstrained for aggressive heavy-rain correction.
MONOTONE_CONSTRAINTS = [
    0,  # Z_low_norm
    0,  # ZDR_low_norm
    0,  # Z_high_norm
    0,  # ZDR_high_norm
    1,  # kan_pred_mmh    — MONOTONE
    1,  # kan_pred_log1p  — MONOTONE
    1,  # kan_pred_sq     — MONOTONE
    0,  # Z_diff
    0,  # ZDR_diff
    0,  # Z_low_raw
    0,  # ZDR_low_raw
    0,  # Z_low_std
    0,  # Z_low_max
    0,  # Z_conv_frac
    0,  # ZDR_low_std
    0,  # ZDR_low_mean
    0,  # Z_high_std
    0,  # Z_high_max
    0,  # ZDR_high_std
    0,  # Z_x_kan      — UNCONSTRAINED
    0,  # Zdiff_x_kan  — UNCONSTRAINED
    0,  # Zhigh_x_kan  — UNCONSTRAINED
    0,  # ZDR_x_kan    — UNCONSTRAINED
]

assert len(MONOTONE_CONSTRAINTS) == len(FEATURE_NAMES), \
    f"Constraint length mismatch: {len(MONOTONE_CONSTRAINTS)} vs {len(FEATURE_NAMES)}"


def build_features(X_raw, X_norm, pred_mmh, pred_log1p):
    """
    Build (N, 23) LightGBM feature matrix.
    col 5 (kan_pred_log1p) = log1p(clip(pred_mmh, 0)).
    pred_log1p arg is accepted for API compatibility but col 5
    is always recomputed from pred_mmh for consistency.
    """
    Z_l   = X_raw[:, 0, CENTER, CENTER]
    ZDR_l = X_raw[:, 1, CENTER, CENTER]
    Z_h   = X_raw[:, 2, CENTER, CENTER]
    ZDR_h = X_raw[:, 3, CENTER, CENTER]

    Z_diff   = Z_l   - Z_h
    ZDR_diff = ZDR_l - ZDR_h

    Z_win        = X_raw[:, 0].reshape(len(X_raw), -1)
    Z_low_std    = Z_win.std(axis=1).astype(np.float32)
    Z_low_max    = Z_win.max(axis=1).astype(np.float32)
    Z_conv_frac  = (Z_win > 40.0).mean(axis=1).astype(np.float32)

    ZDR_win      = X_raw[:, 1].reshape(len(X_raw), -1)
    ZDR_low_std  = ZDR_win.std(axis=1).astype(np.float32)
    ZDR_low_mean = ZDR_win.mean(axis=1).astype(np.float32)

    Z_high_win   = X_raw[:, 2].reshape(len(X_raw), -1)
    Z_high_std   = Z_high_win.std(axis=1).astype(np.float32)
    Z_high_max   = Z_high_win.max(axis=1).astype(np.float32)

    ZDR_high_win = X_raw[:, 3].reshape(len(X_raw), -1)
    ZDR_high_std = ZDR_high_win.std(axis=1).astype(np.float32)

    pred_mmh_pos   = np.clip(pred_mmh, 0.0, None).astype(np.float32)
    pred_log1p_pos = np.log1p(pred_mmh_pos)
    kan_pred_sq    = (pred_mmh_pos ** 2).astype(np.float32)

    Z_x_kan     = (Z_l    * pred_log1p_pos).astype(np.float32)
    Zdiff_x_kan = (Z_diff * pred_log1p_pos).astype(np.float32)
    Zhigh_x_kan = (Z_h    * pred_log1p_pos).astype(np.float32)
    ZDR_x_kan   = (ZDR_l  * pred_log1p_pos).astype(np.float32)

    return np.column_stack([
        X_norm[:, 0, CENTER, CENTER], X_norm[:, 1, CENTER, CENTER],
        X_norm[:, 2, CENTER, CENTER], X_norm[:, 3, CENTER, CENTER],
        pred_mmh_pos, pred_log1p_pos, kan_pred_sq,
        Z_diff, ZDR_diff,
        Z_l, ZDR_l,
        Z_low_std, Z_low_max, Z_conv_frac,
        ZDR_low_std, ZDR_low_mean,
        Z_high_std, Z_high_max,
        ZDR_high_std,
        Z_x_kan, Zdiff_x_kan, Zhigh_x_kan, ZDR_x_kan,
    ]).astype(np.float32)


# ---------------------------------------------------------------------------
# Sample weights
# ---------------------------------------------------------------------------

def compute_sample_weights(y_true: np.ndarray,
                            w_min:  float = 1.0,
                            w_max:  float = 30.0,
                            gamma:  float = 0.05) -> np.ndarray:
    """
    Smooth exponential weight curve — no discrete steps.

    w(y) = w_min + (w_max - w_min) * (exp(gamma * y) - 1) / (exp(gamma * y_max) - 1)

    This gives every rain rate its own continuously-varying weight,
    avoiding the step-function discontinuities that caused the tuner
    to over-correct in some bins while under-correcting in others.

    Optuna searches w_min, w_max, gamma independently, so the shape
    of the curve (how steeply it rises and where) is fully data-driven.

    Example profiles (w_min=1, w_max=30, gamma=0.05, y_max=150):
         0 mm/h ->  1.0x
        10 mm/h ->  2.6x
        25 mm/h ->  5.8x
        50 mm/h -> 12.2x
        75 mm/h -> 19.8x
       100 mm/h -> 25.4x
       150 mm/h -> 30.0x
    """
    y = np.clip(y_true, 0.0, None).astype(np.float64)
    y_max = 150.0
    denom = np.expm1(gamma * y_max)
    if denom < 1e-8:
        # gamma near zero → linear fallback
        w = w_min + (w_max - w_min) * (y / y_max)
    else:
        w = w_min + (w_max - w_min) * np.expm1(gamma * y) / denom
    return w.astype(np.float32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, thresholds=[1.0, 5.0, 10.0, 25.0, 50.0]):
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


def log_extreme_metrics(y_true, y_pred, name, log,
                        thresholds=[25.0, 50.0, 75.0, 100.0]):
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
    """
    Constant offset in log1p space: shift every prediction by the
    mean log1p training residual. LightGBM must beat this.
    """
    val_const = apply_lgb_correction(val_pred_mmh,
                                      np.full(len(val_pred_mmh),
                                              train_resid_log_mean,
                                              dtype=np.float32))
    return val_const, compute_metrics(y_val, val_const)


def bootstrap_metric_deltas(y_val, val_pred_mmh, val_lgb, val_const,
                             n_bootstrap, seed, log):
    rng = np.random.default_rng(seed)
    n   = len(y_val)

    d_rmse_lgb   = np.empty(n_bootstrap, dtype=np.float64)
    d_mae_lgb    = np.empty(n_bootstrap, dtype=np.float64)
    d_rmse_const = np.empty(n_bootstrap, dtype=np.float64)
    d_mae_const  = np.empty(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt  = y_val[idx]
        yk  = val_pred_mmh[idx]
        yl  = val_lgb[idx]
        yc  = val_const[idx]

        rmse_kan        = np.sqrt(np.mean((yk - yt) ** 2))
        mae_kan         = np.mean(np.abs(yk - yt))
        d_rmse_lgb[i]   = np.sqrt(np.mean((yl - yt) ** 2)) - rmse_kan
        d_mae_lgb[i]    = np.mean(np.abs(yl - yt))          - mae_kan
        d_rmse_const[i] = np.sqrt(np.mean((yc - yt) ** 2)) - rmse_kan
        d_mae_const[i]  = np.mean(np.abs(yc - yt))          - mae_kan

    def pcts(arr):
        p2_5, p50, p97_5 = np.percentile(arr, [2.5, 50, 97.5])
        return (float(p2_5), float(p50), float(p97_5))

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
        log.info("  -> LOW: little generalizable residual structure found.")
    elif corr < 0.30:
        log.info("  -> MODERATE: some genuine residual signal captured.")
    else:
        log.info("  -> STRONG: meaningful residual structure captured.")
    return corr


# ---------------------------------------------------------------------------
# Main
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
    log.info("Train y: min=%.2f  max=%.2f  mean=%.2f  mm/h",
             y_train.min(), y_train.max(), y_train.mean())
    log.info("Train extreme (>25mm/h): %d (%.1f%%)",
             (y_train > 25).sum(), 100 * (y_train > 25).mean())
    log.info("Train extreme (>50mm/h): %d (%.1f%%)",
             (y_train > 50).sum(), 100 * (y_train > 50).mean())
    log.info("Train extreme (>75mm/h): %d (%.1f%%)",
             (y_train > 75).sum(), 100 * (y_train > 75).mean())

    # ── Load CNN+KAN ───────────────────────────────────────────────────────
    model, chan_means, chan_stds = load_cnn_kan(
        model_dir / "best_model.pt", device, log)

    X_train_norm = normalise(X_train_raw, chan_means, chan_stds)
    X_val_norm   = normalise(X_val_raw,   chan_means, chan_stds)

    log.info("Running CNN+KAN inference...")
    train_pred_mmh, train_pred_log1p = get_kan_predictions(
        model, X_train_norm, device)
    val_pred_mmh, val_pred_log1p = get_kan_predictions(
        model, X_val_norm, device)

    # ── CNN+KAN baseline ───────────────────────────────────────────────────
    log.info("=" * 65)
    log.info("CNN+KAN baseline:")
    kan_metrics = compute_metrics(y_val, val_pred_mmh)
    for k, v in kan_metrics.items():
        log.info("  %-15s : %.4f", k, v)
    log.info("  Extreme performance:")
    log_extreme_metrics(y_val, val_pred_mmh, "CNN+KAN", log)

    # ── LOG1P-SPACE RESIDUALS ──────────────────────────────────────────────
    # Target: log1p(R_true) - log1p(R_hat)
    # Range: ~0.1 to ~0.5 across all rain rates (vs 0.8 to 45 mm/h)
    # This gives LightGBM equal gradient signal at all rain rates.
    # The heavy-rain sample weights then amplify the tail further.
    y_train_log1p = np.log1p(np.clip(y_train, 0.0, None)).astype(np.float32)
    y_val_log1p   = np.log1p(np.clip(y_val,   0.0, None)).astype(np.float32)

    train_pred_log1p_pos = np.log1p(
        np.clip(train_pred_mmh, 0.0, None)).astype(np.float32)
    val_pred_log1p_pos   = np.log1p(
        np.clip(val_pred_mmh,   0.0, None)).astype(np.float32)

    train_resid_log = (y_train_log1p - train_pred_log1p_pos).astype(np.float32)
    val_resid_log   = (y_val_log1p   - val_pred_log1p_pos  ).astype(np.float32)

    log.info("=" * 65)
    log.info("Log1p-space residual stats (train):")
    log.info("  mean=%+.4f  std=%.4f  min=%+.4f  max=%+.4f",
             train_resid_log.mean(), train_resid_log.std(),
             train_resid_log.min(),  train_resid_log.max())

    for thr in [0, 25, 50, 75]:
        mask = y_train >= thr
        if mask.sum() > 5:
            log.info("  y_train >= %3d: resid_log mean=%+.4f  std=%.4f  N=%d",
                     thr, train_resid_log[mask].mean(),
                     train_resid_log[mask].std(), mask.sum())

    # ── Features ───────────────────────────────────────────────────────────
    log.info("Building feature matrices (%d features)...", len(FEATURE_NAMES))
    X_lgb_train = build_features(X_train_raw, X_train_norm,
                                  train_pred_mmh, train_pred_log1p)
    X_lgb_val   = build_features(X_val_raw,   X_val_norm,
                                  val_pred_mmh,   val_pred_log1p)

    # ── Sample weights ─────────────────────────────────────────────────────
    # Two-stage: sqrt baseline + additive step boosts at 25/50/75 mm/h.
    # This directly amplifies the tail that the reliability diagram shows
    # is severely under-corrected, without relying on the tuner to
    # compensate with high weight_scale alone.
    weights = compute_sample_weights(y_train, w_min, w_max, gamma)
    log.info("Sample weights (smooth exp curve): w_min=%.2f  w_max=%.2f  gamma=%.4f",
             w_min, w_max, gamma)
    log.info("  Weight range=[%.2f, %.2f]  median=%.2f",
             weights.min(), weights.max(), np.median(weights))
    for thr in [25, 50, 75]:
        mask = y_train > thr
        if mask.any():
            log.info("  Mean weight >%dmm/h: %.2f  (N=%d)",
                     thr, weights[mask].mean(), mask.sum())

    # ── Objective ──────────────────────────────────────────────────────────
    # MAE (L1) is preferred for log1p-space residuals:
    # - The target distribution is now compact (~0.1 to 0.5)
    # - L1 is median-seeking per leaf, robust to the remaining
    #   asymmetry in the log1p residual distribution.
    # - Huber is a reasonable alternative for fine-tuning but MAE is the
    #   recommended default after the space change.
    if use_huber:
        objective = "huber"
        alpha     = huber_delta
        metric    = "huber"
        obj_desc  = f"Huber(delta={huber_delta:.4f})"
    else:
        objective = "regression_l2"
        alpha     = 0.9
        metric    = "mae"
        obj_desc  = "MAE (L1)"

    log.info("Objective      : %s", obj_desc)
    log.info("Target space   : LOG1P (epsilon = log1p(R_true) - log1p(R_hat))")
    log.info("Inference      : R_final = expm1(log1p(R_hat) + epsilon_hat)")
    log.info("Constraints    : v4 — KAN features monotone, interactions unconstrained")
    log.info("Sample weights : smooth exp curve  w_min=%.2f  w_max=%.2f  gamma=%.4f",
             w_min, w_max, gamma)
    log.info("LightGBM config: n_estimators=%d  lr=%.6f  num_leaves=%d  "
             "max_depth=%d  min_child=%d  early_stop=%d",
             n_estimators, lr, num_leaves, max_depth,
             min_child_samples, early_stop)
    log.info("Regularisation : reg_alpha=%.4f  reg_lambda=%.4f  "
             "subsample=%.4f  colsample=%.4f",
             reg_alpha, reg_lambda, subsample, colsample)
    log.info("=" * 65)

    lgb_train  = lgb.Dataset(X_lgb_train, label=train_resid_log,
                              weight=weights, feature_name=FEATURE_NAMES)
    lgb_val_ds = lgb.Dataset(X_lgb_val,   label=val_resid_log,
                              reference=lgb_train, feature_name=FEATURE_NAMES)

    if use_dart:
        log.info("Training with DART boosting (no early stopping, "
                 "fixed n_estimators=%d)", n_estimators)
        params = {
            "boosting_type":        "dart",
            "drop_rate":            0.1,
            "skip_drop":            0.5,
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
        lgb_model = lgb.train(params, lgb_train,
                               num_boost_round=n_estimators,
                               callbacks=[lgb.log_evaluation(period=50)])
        log.info("DART done in %.1fs", time.time() - t0)

    else:
        log.info("Training with GBDT + early stopping (patience=%d)", early_stop)
        params = {
            "boosting_type":        "gbdt",
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
        lgb_model = lgb.train(
            params, lgb_train,
            num_boost_round=n_estimators,
            valid_sets=[lgb_val_ds],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stop, verbose=True),
                lgb.log_evaluation(period=25),
            ])
        log.info("GBDT done in %.1fs  best_iter=%d",
                 time.time() - t0, lgb_model.best_iteration)

    # ── Evaluate ───────────────────────────────────────────────────────────
    num_iter = (lgb_model.best_iteration
                if hasattr(lgb_model, "best_iteration")
                and lgb_model.best_iteration > 0
                else lgb_model.num_trees())

    # Predict log1p-space residuals, apply inverse transform
    # CRITICAL: use apply_lgb_correction (expm1 inverse), NOT mm/h addition
    val_epsilon_log = lgb_model.predict(X_lgb_val, num_iteration=num_iter)
    val_final       = apply_lgb_correction(val_pred_mmh, val_epsilon_log)

    log.info("=" * 65)
    log.info("Final pipeline metrics (CNN+KAN + LightGBM):")
    final_metrics = compute_metrics(y_val, val_final)
    for k, v in final_metrics.items():
        log.info("  %-15s : %.4f", k, v)
    log.info("  Extreme performance:")
    log_extreme_metrics(y_val, val_final, "Full pipeline", log)

    log.info("=" * 65)
    log.info("Improvement over CNN+KAN baseline:")
    for m in ["RMSE", "MAE", "Bias", "Correlation", "R2"]:
        b = kan_metrics.get(m, 0)
        a = final_metrics.get(m, 0)
        log.info("  %-13s : %.4f -> %.4f  (%+.4f)", m, b, a, a - b)

    # ── Diagnostics ────────────────────────────────────────────────────────
    train_resid_log_mean = float(train_resid_log.mean())
    val_const, const_metrics = compute_constant_offset_baseline(
        y_val, val_pred_mmh, train_resid_log_mean)

    log.info("=" * 65)
    log.info("Constant-offset baseline (mean log1p train resid = %+.4f):",
             train_resid_log_mean)
    for k, v in const_metrics.items():
        log.info("  %-15s : %.4f", k, v)
    log.info("LightGBM vs constant-offset:")
    for m in ["RMSE", "MAE", "Bias"]:
        c = const_metrics.get(m, 0)
        a = final_metrics.get(m, 0)
        log.info("  %-13s : const=%.4f  lgb=%.4f  (lgb-const=%+.4f)",
                 m, c, a, a - c)

    bootstrap_results = bootstrap_metric_deltas(
        y_val, val_pred_mmh, val_final, val_const,
        n_bootstrap, bootstrap_seed, log)

    resid_corr = compute_residual_correlation(
        val_resid_log, val_epsilon_log, log)

    # ── Verdict ────────────────────────────────────────────────────────────
    rmse_ci = bootstrap_results["rmse_lgb_minus_kan"]
    mae_ci  = bootstrap_results["mae_lgb_minus_kan"]
    rmse_vs_const = final_metrics["RMSE"] - const_metrics["RMSE"]
    mae_vs_const  = final_metrics["MAE"]  - const_metrics["MAE"]

    rmse_sig    = rmse_ci[2] < 0
    mae_sig     = mae_ci[2]  < 0
    beats_const = (rmse_vs_const < 0) and (mae_vs_const < 0)

    log.info("=" * 65)
    log.info("DIAGNOSTIC VERDICT:")
    if rmse_sig and mae_sig and beats_const and resid_corr >= 0.15:
        verdict = ("STRONG: RMSE and MAE improvements are significant "
                   "(95% CI entirely below zero), LightGBM beats the "
                   "constant-offset baseline on both metrics, and residual "
                   f"correlation ({resid_corr:.3f}) confirms genuine "
                   "structure was captured.")
    elif beats_const and (rmse_sig or mae_sig):
        verdict = ("PARTIAL: LightGBM beats the constant-offset baseline "
                   "and at least one metric is significantly improved, but "
                   "not both, or residual correlation "
                   f"({resid_corr:.3f}) is low. Treat gain as marginal.")
    else:
        verdict = ("WEAK/NONE: improvements are not distinguishable from "
                   "noise (CI spans zero) and/or LightGBM does not beat "
                   "the trivial constant-offset baseline. Residual "
                   f"correlation = {resid_corr:.3f}.")
    log.info("  %s", verdict)

    # ── Feature importance ─────────────────────────────────────────────────
    importance = dict(sorted(
        {name: int(score) for name, score in zip(
            FEATURE_NAMES,
            lgb_model.feature_importance(importance_type="gain"))}.items(),
        key=lambda x: x[1], reverse=True))
    log.info("Feature importance (gain):")
    for name, score in importance.items():
        log.info("  %-20s : %d", name, score)

    # ── Save ───────────────────────────────────────────────────────────────
    with open(model_dir / "lgb_residual.pkl", "wb") as f:
        pickle.dump(lgb_model, f)

    metrics_out = {
        "kan_baseline":   {k: round(v, 5) for k, v in kan_metrics.items()},
        "final_pipeline": {k: round(v, 5) for k, v in final_metrics.items()},
        "constant_offset_baseline": {
            "train_resid_log_mean": round(train_resid_log_mean, 5),
            **{k: round(v, 5) for k, v in const_metrics.items()},
        },
        "improvement": {
            k: round(final_metrics.get(k, 0) - kan_metrics.get(k, 0), 5)
            for k in ["RMSE", "MAE", "Bias", "Correlation", "R2"]
        },
        "improvement_vs_constant_offset": {
            k: round(final_metrics.get(k, 0) - const_metrics.get(k, 0), 5)
            for k in ["RMSE", "MAE", "Bias", "Correlation", "R2"]
        },
        "diagnostics": {
            "n_bootstrap":    n_bootstrap,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_ci": {
                k: {"p2.5": v[0], "p50": v[1], "p97.5": v[2]}
                for k, v in bootstrap_results.items()
            },
            "residual_correlation": round(resid_corr, 5),
            "verdict": verdict,
        },
        "lgb_num_trees":        num_iter,
        "feature_importance":   importance,
        "feature_names":        FEATURE_NAMES,
        "residual_space":       "log1p",
        "monotone_constraints": MONOTONE_CONSTRAINTS,
        "constraint_version":   "v4_interactions_unconstrained",
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
    with open(model_dir / "lgb_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    log.info("Saved: lgb_residual.pkl  lgb_metrics.json")
    log.info("LightGBM training complete.")
    log.info("IMPORTANT: inference must use apply_lgb_correction():")
    log.info("  epsilon = lgb.predict(features)          # log1p space residual")
    log.info("  R_final = expm1(log1p(R_hat) + epsilon)  # NOT mm/h addition")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train LightGBM corrector — log1p residual space.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_dir",   type=Path,
                   default=_ROOT / "data" / "dataset" / "processed_T4_improved")
    p.add_argument("--model_dir",  type=Path,
                   default=_ROOT / "models" / "qpe_cnn_kan")

    # Hyperparameters — v5 Optuna best (trial #55)
    p.add_argument("--n_estimators",      type=int,   default=3000)
    p.add_argument("--lr",                type=float, default=0.002668975812939989)
    p.add_argument("--max_depth",         type=int,   default=8)
    p.add_argument("--num_leaves",        type=int,   default=52)
    p.add_argument("--subsample",         type=float, default=0.6671197214517827)
    p.add_argument("--colsample",         type=float, default=0.7816647455711815)
    p.add_argument("--reg_alpha",         type=float, default=0.06403799953123597)
    p.add_argument("--reg_lambda",        type=float, default=24.81351678103886)
    p.add_argument("--min_child_samples", type=int,   default=17)
    p.add_argument("--early_stop",        type=int,   default=200)

    # Smooth exponential weight curve — replaces step boosts
    p.add_argument("--w_min",  type=float, default=0.8536416401563117,
                   help="Minimum sample weight (at y=0 mm/h).")
    p.add_argument("--w_max",  type=float, default=37.9585681574043,
                   help="Maximum sample weight (at y=150 mm/h).")
    p.add_argument("--gamma",  type=float, default=0.04383423058756062,
                   help="Exponential growth rate of weight curve.")

    # Huber — v5 Optuna best
    p.add_argument("--use_huber", default=True,
                   action=argparse.BooleanOptionalAction,
                   help="Use Huber loss instead of MAE.")
    p.add_argument("--huber_delta", type=float, default=0.38797174962427217,
                   help="Huber delta (only used if --use_huber).")
    p.add_argument("--use_dart",  default=False,
                   action=argparse.BooleanOptionalAction,
                   help="Use DART boosting (slower, no early stopping).")

    p.add_argument("--n_bootstrap",    type=int, default=1000)
    p.add_argument("--bootstrap_seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(args.model_dir)
    log.info("LightGBM Corrector (log1p residual space) started.")
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