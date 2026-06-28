"""
evaluate_all_v2.py
==================
Unified evaluation comparing ALL QPE models on the validation set.
Version 2 — uses 37-feature LightGBM corrector (lgb_residual_v2.pkl).

Models evaluated
----------------
    Z-R Baselines:
        1. Z-R Marshall-Palmer        Z = 200 * R^1.6
        2. Z-R Tropical (NWS Miami)   Z = 300 * R^1.35
        3. Z-R WSR-88D Default        Z = 300 * R^1.4

    Deep Learning:
        4. CNN+KAN
        5. CNN+KAN + LightGBM v2   (37 features)
        6. RQPENetD1
        7. ConvLSTMQPE

Outputs
-------
    eval/evaluation_log_all_v2.txt
    eval/evaluation_report_all_v2.json
    eval/evaluation_summary_all_v2.txt

Per-category metrics
--------------------
    light    :  1 – 10  mm/h
    moderate : 10 – 25  mm/h
    heavy    : 25 – 50  mm/h
    extreme  :     > 50 mm/h
    all      :     >= 1 mm/h
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
from typing import Dict, List

import numpy as np
from scipy.stats import skew, kurtosis
import torch
import torch.nn as nn

try:
    from cnn_feature_extractor import RadarCNN
    from kan_rainfall_predictor import RainfallKAN
    _HAS_CNNKAN = True
except ImportError:
    _HAS_CNNKAN = False

try:
    from rqpenetd1 import RQPENetD1
    _HAS_RQPENET = True
except ImportError:
    _HAS_RQPENET = False

try:
    from convlstm_qpe import ConvLSTMQPE
    _HAS_CONVLSTM = True
except ImportError:
    _HAS_CONVLSTM = False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(eval_dir: Path) -> logging.Logger:
    log = logging.getLogger("qpe_eval_v2")
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    log.addHandler(ch)
    fh = logging.FileHandler(eval_dir / "evaluation_log_all_v2.txt", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------

class CNNKANModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = RadarCNN(in_channels=4, feature_dim=32)
        self.kan = RainfallKAN(input_dim=32, hidden1=64, hidden2=32,
                               grid_size=5, spline_order=3)
    def forward(self, x):
        return self.kan(self.cnn(x))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CENTER = 4

def normalise(X: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32).copy()
    for c in range(4):
        X[:, c] = (X[:, c] - means[c]) / (stds[c] + 1e-8)
    return X


ZR_RELATIONS = [
    ("Z-R Marshall-Palmer",      200.0, 1.6),
    ("Z-R Tropical (NWS Miami)", 300.0, 1.35),
    ("Z-R WSR-88D Default",      300.0, 1.4),
]

def zr_predict(Z_dbz: np.ndarray, a: float, b: float) -> np.ndarray:
    return np.clip((10.0 ** (Z_dbz / 10.0) / a) ** (1.0 / b), 0.0, None)


@torch.no_grad()
def dl_predict(model: nn.Module, X_norm: np.ndarray,
               device: torch.device, batch: int = 512) -> np.ndarray:
    model.eval()
    preds = []
    for s in range(0, len(X_norm), batch):
        x_t = torch.from_numpy(X_norm[s:s+batch]).to(device)
        preds.append(model(x_t).squeeze(1).cpu().numpy())
    return np.clip(np.expm1(np.concatenate(preds)), 0.0, None)


# ---------------------------------------------------------------------------
# Feature builder — 37 features (must match train_lightgbm_v2.py exactly)
# ---------------------------------------------------------------------------

def build_lgb_features(X_raw: np.ndarray,
                        X_norm: np.ndarray,
                        pred_mmh: np.ndarray) -> np.ndarray:
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


def apply_lgb_correction(pred_mmh: np.ndarray,
                          epsilon_log: np.ndarray) -> np.ndarray:
    """R_final = clamp(expm1(log1p(R_hat) + epsilon_log), min=0)"""
    return np.clip(np.expm1(np.log1p(np.clip(pred_mmh, 0.0, None)) + epsilon_log), 0.0, None)


# ---------------------------------------------------------------------------
# Rainfall categories
# ---------------------------------------------------------------------------

RAIN_CATEGORIES = {
    "light (1-10)":     (1.0,    10.0),
    "moderate (10-25)": (10.0,   25.0),
    "heavy (25-50)":    (25.0,   50.0),
    "extreme (>50)":    (50.0,   9999.0),
    "all (>=1)":        (1.0,    9999.0),
}


def category_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {}
    for cat, (lo, hi) in RAIN_CATEGORIES.items():
        mask = (y_true >= lo) & (y_true < hi)
        n = int(mask.sum())
        if n < 5:
            out[cat] = {"RMSE": None, "MAE": None, "Bias": None, "N": n}
            continue
        res = y_pred[mask] - y_true[mask]
        out[cat] = {
            "RMSE": round(float(np.sqrt(np.mean(res**2))), 4),
            "MAE":  round(float(np.mean(np.abs(res))),     4),
            "Bias": round(float(np.mean(res)),              4),
            "N":    n,
        }
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def continuous_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    res  = y_pred - y_true
    ss_r = np.sum(res ** 2)
    ss_t = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "RMSE":        round(float(np.sqrt(np.mean(res**2))),                    5),
        "MAE":         round(float(np.mean(np.abs(res))),                         5),
        "MRE":         round(float(np.mean(np.abs(res) / (np.abs(y_true)+1e-8))),5),
        "R2":          round(float(1 - ss_r / (ss_t + 1e-8)),                     5),
        "Bias":        round(float(np.mean(res)),                                  5),
        "Correlation": round(float(np.corrcoef(y_true, y_pred)[0, 1]),            5),
    }


def categorical_metrics(y_true: np.ndarray, y_pred: np.ndarray, thr: float) -> dict:
    obs_p = y_true >= thr;  pred_p = y_pred >= thr
    obs_n = ~obs_p;          pred_n = ~pred_p
    TP = float(np.sum(obs_p  & pred_p))
    FP = float(np.sum(obs_n  & pred_p))
    FN = float(np.sum(obs_p  & pred_n))
    TN = float(np.sum(obs_n  & pred_n))
    N  = TP + FP + FN + TN
    pod = TP/(TP+FN)        if (TP+FN)    > 0 else 0.0
    far = FP/(TP+FP)        if (TP+FP)    > 0 else 0.0
    csi = TP/(TP+FN+FP)     if (TP+FN+FP) > 0 else 0.0
    dh  = (TP+FN)*(FN+TN) + (TP+FP)*(FP+TN)
    hss = 2*(TP*TN-FP*FN)/dh if dh > 0 else 0.0
    Q   = (TP+FN)*(TP+FP)/N  if N  > 0 else 0.0
    dg  = TP+FN+FP-Q
    gss = (TP-Q)/dg           if dg > 0 else 0.0
    return {
        "POD": round(pod,5), "FAR": round(far,5), "CSI": round(csi,5),
        "HSS": round(hss,5), "GSS": round(gss,5),
        "TP": int(TP), "FP": int(FP), "FN": int(FN), "TN": int(TN),
    }


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                   thresholds: List[float] = [1.0, 5.0, 10.0, 25.0]) -> dict:
    return {
        "continuous":             continuous_metrics(y_true, y_pred),
        "continuous_by_category": category_metrics(y_true, y_pred),
        "categorical": {
            f"{t:.0f}mm/h": categorical_metrics(y_true, y_pred, t)
            for t in thresholds
        },
        "n_samples": int(len(y_true)),
        "pred_mean": round(float(y_pred.mean()), 3),
        "pred_max":  round(float(y_pred.max()),  3),
    }


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def print_table(results: dict, y_true: np.ndarray, log: logging.Logger) -> None:
    sep   = "=" * 108
    names = list(results.keys())

    # Overall continuous
    log.info(sep)
    log.info("CONTINUOUS METRICS — OVERALL  (N=%d  mean=%.2f  max=%.2f mm/h)",
             len(y_true), y_true.mean(), y_true.max())
    log.info(sep)
    log.info(f"{'Model':<32} {'RMSE':>8} {'MAE':>8} {'MRE':>8} "
             f"{'R2':>8} {'Bias':>9} {'r':>8}")
    log.info("-" * 84)
    for name in names:
        c = results[name]["continuous"]
        log.info(f"{name:<32} {c['RMSE']:>8.3f} {c['MAE']:>8.3f} "
                 f"{c['MRE']:>8.4f} {c['R2']:>8.4f} "
                 f"{c['Bias']:>+9.3f} {c['Correlation']:>8.4f}")

    # Per-category
    for cat in RAIN_CATEGORIES:
        log.info("")
        log.info(sep)
        log.info("CONTINUOUS METRICS — %s", cat.upper())
        log.info(sep)
        log.info(f"{'Model':<32} {'RMSE':>8} {'MAE':>8} {'Bias':>9} {'N':>7}")
        log.info("-" * 65)
        for name in names:
            m = results[name]["continuous_by_category"].get(cat, {})
            if m.get("RMSE") is None:
                log.info(f"{name:<32} {'—':>8} {'—':>8} {'—':>9} {m.get('N',0):>7d}")
            else:
                log.info(f"{name:<32} {m['RMSE']:>8.3f} {m['MAE']:>8.3f} "
                         f"{m['Bias']:>+9.3f} {m['N']:>7d}")

    # Categorical
    for thr in ["1mm/h", "5mm/h", "10mm/h", "25mm/h"]:
        log.info("")
        log.info(sep)
        log.info("CATEGORICAL @ %s", thr)
        log.info(sep)
        log.info(f"{'Model':<32} {'POD':>8} {'FAR':>8} {'CSI':>8} "
                 f"{'HSS':>8} {'GSS':>8}   {'TP':>6} {'FP':>6} {'FN':>6}")
        log.info("-" * 92)
        for name in names:
            cat = results[name]["categorical"][thr]
            log.info(f"{name:<32} {cat['POD']:>8.4f} {cat['FAR']:>8.4f} "
                     f"{cat['CSI']:>8.4f} {cat['HSS']:>8.4f} {cat['GSS']:>8.4f} "
                     f"  {cat['TP']:>6d} {cat['FP']:>6d} {cat['FN']:>6d}")
    log.info(sep)


def best_per_metric(results: dict, log: logging.Logger) -> None:
    log.info("BEST MODEL PER METRIC:")
    for metric in ["RMSE", "MAE", "MRE", "R2", "Bias", "Correlation"]:
        vals = {m: results[m]["continuous"][metric] for m in results}
        if metric in {"R2", "Correlation"}:
            best = max(vals, key=vals.get)
        elif metric == "Bias":
            best = min(vals, key=lambda m: abs(vals[m]))
        else:
            best = min(vals, key=vals.get)
        log.info("  %-15s : %-32s (%.4f)", metric, best, vals[best])


def write_summary(results: dict, y_true: np.ndarray, path: Path) -> None:
    lines = [
        "QPE MODEL EVALUATION SUMMARY v2 — ALL MODELS", "=" * 108,
        f"N={len(y_true)}  mean={y_true.mean():.2f}  max={y_true.max():.2f} mm/h",
        f"Models: {len(results)}", "",
        "CONTINUOUS METRICS — OVERALL", "-" * 84,
        f"{'Model':<32} {'RMSE':>8} {'MAE':>8} {'MRE':>8} "
        f"{'R2':>8} {'Bias':>9} {'r':>8}", "-" * 84,
    ]
    for name, res in results.items():
        c = res["continuous"]
        lines.append(f"{name:<32} {c['RMSE']:>8.3f} {c['MAE']:>8.3f} "
                     f"{c['MRE']:>8.4f} {c['R2']:>8.4f} "
                     f"{c['Bias']:>+9.3f} {c['Correlation']:>8.4f}")

    for cat in RAIN_CATEGORIES:
        lines += ["", f"CONTINUOUS — {cat.upper()}", "-" * 65,
                  f"{'Model':<32} {'RMSE':>8} {'MAE':>8} {'Bias':>9} {'N':>7}", "-" * 65]
        for name, res in results.items():
            m = res["continuous_by_category"].get(cat, {})
            if m.get("RMSE") is None:
                lines.append(f"{name:<32} {'—':>8} {'—':>8} {'—':>9} {m.get('N',0):>7d}")
            else:
                lines.append(f"{name:<32} {m['RMSE']:>8.3f} {m['MAE']:>8.3f} "
                             f"{m['Bias']:>+9.3f} {m['N']:>7d}")

    for thr in ["1mm/h", "5mm/h", "10mm/h", "25mm/h"]:
        lines += ["", f"CATEGORICAL @ {thr}", "-" * 92,
                  f"{'Model':<32} {'POD':>8} {'FAR':>8} {'CSI':>8} "
                  f"{'HSS':>8} {'GSS':>8}   {'TP':>6} {'FP':>6} {'FN':>6}", "-" * 92]
        for name, res in results.items():
            cat = res["categorical"][thr]
            lines.append(f"{name:<32} {cat['POD']:>8.4f} {cat['FAR']:>8.4f} "
                         f"{cat['CSI']:>8.4f} {cat['HSS']:>8.4f} {cat['GSS']:>8.4f} "
                         f"  {cat['TP']:>6d} {cat['FP']:>6d} {cat['FN']:>6d}")
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(data_dir: Path, models_dir: Path,
             eval_dir: Path, log: logging.Logger) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    log.info("Loading validation set from %s", data_dir)
    X_val_raw = np.load(data_dir / "X_val.npy")
    y_val     = np.load(data_dir / "y_val.npy")
    log.info("Val: N=%d  y=[%.2f, %.2f] mm/h  mean=%.2f",
             len(y_val), y_val.min(), y_val.max(), y_val.mean())

    ckpt_path = models_dir / "qpe_cnn_kan" / "best_model.pt"
    if not ckpt_path.exists():
        with open(data_dir / "dataset_stats.json") as f:
            stats = json.load(f)
        channel_names = ["Z_low", "ZDR_low", "Z_high", "ZDR_high"]
        chan_means = np.array([stats[n]["mean"] for n in channel_names], np.float32)
        chan_stds  = np.array([stats[n]["std"]  for n in channel_names], np.float32)
        ckpt_cnn   = None
    else:
        ckpt_cnn  = torch.load(ckpt_path, map_location="cpu")
        chan_means = np.array(ckpt_cnn["chan_means"], dtype=np.float32)
        chan_stds  = np.array(ckpt_cnn["chan_stds"],  dtype=np.float32)

    X_val_norm    = normalise(X_val_raw, chan_means, chan_stds)
    Z_low_raw_val = X_val_raw[:, 0, CENTER, CENTER]

    results: Dict = {}

    # ── Z-R baselines ──────────────────────────────────────────────────────
    log.info("\nEvaluating Z-R baselines...")
    for name, a, b in ZR_RELATIONS:
        pred = zr_predict(Z_low_raw_val, a, b)
        results[name] = evaluate_model(y_val, pred)
        log.info("  %-32s  RMSE=%.3f  r=%.4f  Bias=%+.3f",
                 name, results[name]["continuous"]["RMSE"],
                 results[name]["continuous"]["Correlation"],
                 results[name]["continuous"]["Bias"])

    # ── CNN+KAN ────────────────────────────────────────────────────────────
    log.info("\nEvaluating CNN+KAN...")
    if not _HAS_CNNKAN or ckpt_cnn is None:
        log.warning("  SKIPPED — CNN+KAN checkpoint not found")
    else:
        model_ck = CNNKANModel().to(device)
        model_ck.load_state_dict(ckpt_cnn["model_state"])
        kan_pred_mmh = dl_predict(model_ck, X_val_norm, device)
        results["CNN+KAN"] = evaluate_model(y_val, kan_pred_mmh)
        log.info("  %-32s  RMSE=%.3f  r=%.4f  Bias=%+.3f",
                 "CNN+KAN", results["CNN+KAN"]["continuous"]["RMSE"],
                 results["CNN+KAN"]["continuous"]["Correlation"],
                 results["CNN+KAN"]["continuous"]["Bias"])

        # ── CNN+KAN + LightGBM v2 ──────────────────────────────────────────
        log.info("\nEvaluating CNN+KAN + LightGBM v2...")
        lgb_path = models_dir / "qpe_cnn_kan" / "lgb_residual_v2.pkl"
        meta_path = models_dir / "qpe_cnn_kan" / "lgb_metrics_v2.json"

        if not lgb_path.exists():
            log.warning("  SKIPPED — lgb_residual_v2.pkl not found. "
                        "Run train_lightgbm_v2.py first.")
        else:
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                log.info("  Loaded v2 model: %d features  %d trees",
                         meta.get("n_features", "?"),
                         meta.get("lgb_num_trees", "?"))
                if meta.get("residual_space") != "log1p":
                    log.warning("  WARNING: residual_space is not log1p — "
                                "wrong model loaded?")

            with open(lgb_path, "rb") as f:
                lgb_model = pickle.load(f)

            X_lgb    = build_lgb_features(X_val_raw, X_val_norm, kan_pred_mmh)
            num_iter = (lgb_model.best_iteration
                        if hasattr(lgb_model, "best_iteration")
                        and lgb_model.best_iteration > 0
                        else lgb_model.num_trees())

            epsilon_log = lgb_model.predict(X_lgb, num_iteration=num_iter)
            lgb_pred    = apply_lgb_correction(kan_pred_mmh, epsilon_log)

            results["CNN+KAN+LightGBM v2"] = evaluate_model(y_val, lgb_pred)
            log.info("  %-32s  RMSE=%.3f  r=%.4f  Bias=%+.3f",
                     "CNN+KAN+LightGBM v2",
                     results["CNN+KAN+LightGBM v2"]["continuous"]["RMSE"],
                     results["CNN+KAN+LightGBM v2"]["continuous"]["Correlation"],
                     results["CNN+KAN+LightGBM v2"]["continuous"]["Bias"])

    # ── RQPENetD1 ──────────────────────────────────────────────────────────
    log.info("\nEvaluating RQPENetD1...")
    rqpe_ckpt = models_dir / "rqpenetd1" / "best_model.pt"
    if not _HAS_RQPENET or not rqpe_ckpt.exists():
        log.warning("  SKIPPED")
    else:
        ckpt_rq  = torch.load(rqpe_ckpt, map_location="cpu")
        cfg      = ckpt_rq.get("model_config", {})
        model_rq = RQPENetD1(
            growth_rate   = cfg.get("growth_rate",   12),
            block_config  = tuple(cfg.get("block_config", [6, 12, 12, 8])),
            stem_channels = cfg.get("stem_channels", 32),
        ).to(device)
        model_rq.load_state_dict(ckpt_rq["model_state"])
        rq_means = np.array(ckpt_rq.get("chan_means", chan_means.tolist()), np.float32)
        rq_stds  = np.array(ckpt_rq.get("chan_stds",  chan_stds.tolist()),  np.float32)
        rq_pred  = dl_predict(model_rq, normalise(X_val_raw, rq_means, rq_stds), device)
        results["RQPENetD1"] = evaluate_model(y_val, rq_pred)
        log.info("  %-32s  RMSE=%.3f  r=%.4f  Bias=%+.3f",
                 "RQPENetD1",
                 results["RQPENetD1"]["continuous"]["RMSE"],
                 results["RQPENetD1"]["continuous"]["Correlation"],
                 results["RQPENetD1"]["continuous"]["Bias"])

    # ── ConvLSTMQPE ────────────────────────────────────────────────────────
    log.info("\nEvaluating ConvLSTMQPE...")
    clstm_ckpt = models_dir / "convlstm_qpe" / "best_model.pt"
    if not _HAS_CONVLSTM or not clstm_ckpt.exists():
        log.warning("  SKIPPED")
    else:
        ckpt_cl  = torch.load(clstm_ckpt, map_location="cpu")
        cfg_cl   = ckpt_cl.get("model_config", {})
        model_cl = ConvLSTMQPE(
            in_channels = cfg_cl.get("in_channels", 1),
            hidden_dim  = cfg_cl.get("hidden_dim",  64),
            num_layers  = cfg_cl.get("num_layers",  2),
            kernel_size = cfg_cl.get("kernel_size", 3),
        ).to(device)
        model_cl.load_state_dict(ckpt_cl["model_state"])
        cl_means = np.array(ckpt_cl.get("chan_means", chan_means.tolist()), np.float32)
        cl_stds  = np.array(ckpt_cl.get("chan_stds",  chan_stds.tolist()),  np.float32)
        cl_pred  = dl_predict(model_cl, normalise(X_val_raw, cl_means, cl_stds), device)
        results["ConvLSTMQPE"] = evaluate_model(y_val, cl_pred)
        log.info("  %-32s  RMSE=%.3f  r=%.4f  Bias=%+.3f",
                 "ConvLSTMQPE",
                 results["ConvLSTMQPE"]["continuous"]["RMSE"],
                 results["ConvLSTMQPE"]["continuous"]["Correlation"],
                 results["ConvLSTMQPE"]["continuous"]["Bias"])

    # ── Print & save ───────────────────────────────────────────────────────
    log.info("")
    print_table(results, y_val, log)
    log.info("")
    best_per_metric(results, log)

    report = {
        "version": "v2",
        "dataset": {
            "n_val":           int(len(y_val)),
            "y_min":           round(float(y_val.min()), 3),
            "y_max":           round(float(y_val.max()), 3),
            "y_mean":          round(float(y_val.mean()), 3),
            "thresholds":      [1.0, 5.0, 10.0, 25.0],
            "rain_categories": {k: list(v) for k, v in RAIN_CATEGORIES.items()},
        },
        "models_evaluated": list(results.keys()),
        "models":           results,
    }
    report_path = eval_dir / "evaluation_report_all_v2.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Report  saved : %s", report_path)

    summary_path = eval_dir / "evaluation_summary_all_v2.txt"
    write_summary(results, y_val, summary_path)
    log.info("Summary saved : %s", summary_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate ALL QPE models — v2 (37-feature LightGBM)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_dir",   type=Path,
                   default=_ROOT / "data" / "dataset" / "processed_T4_improved")
    p.add_argument("--models_dir", type=Path,
                   default=_ROOT / "models")
    p.add_argument("--eval_dir",   type=Path,
                   default=_ROOT / "eval")
    return p.parse_args()


def main():
    args = parse_args()
    args.eval_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(args.eval_dir)
    log.info("QPE Full Evaluation v2 started.")
    log.info("Config: %s", vars(args))
    evaluate(data_dir   = args.data_dir,
             models_dir = args.models_dir,
             eval_dir   = args.eval_dir,
             log        = log)


if __name__ == "__main__":
    main()