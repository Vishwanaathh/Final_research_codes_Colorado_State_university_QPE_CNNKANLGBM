"""
evaluate.py
===========
Comprehensive evaluation comparing traditional Z-R baselines against
the CNN+KAN and CNN+KAN+LightGBM pipeline on the QPE validation set.

Models evaluated
----------------
    1. Z-R Marshall-Palmer      Z = 200 * R^1.6
    2. Z-R Tropical (NWS Miami) Z = 300 * R^1.35
    3. Z-R WSR-88D Default      Z = 300 * R^1.4
    4. CNN+KAN
    5. CNN+KAN + LightGBM

Changes
-------
  1. LightGBM inference updated to log1p-space inverse transform.
     LightGBM now predicts epsilon = log1p(R_true) - log1p(R_hat).
     The corrected prediction is:
         R_final = expm1(log1p(R_hat) + epsilon)
     Previously: R_final = clamp(R_hat + epsilon_mmh, 0)
     This matches the updated train_lightgbm_corrector.py.

  2. lgb_metrics.json residual_space field checked to warn if a
     model trained in mm/h space is loaded with the wrong transform.

  3. build_lgb_features() pred_log1p arg removed — col 5 is always
     log1p(clip(pred_mmh, 0)), consistent with train script.

  4. Dead code removed from build_lgb_features() (old 17-feature
     unreachable block).

Usage
-----
    python evaluate.py
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
from typing import Dict, List, Tuple

import numpy as np
import torch

try:
    from cnn_feature_extractor import RadarCNN
    from kan_rainfall_predictor import RainfallKAN
except ImportError as e:
    sys.exit(f"[FATAL] Cannot import from {_AI_SCRIPT_DIR}\n{e}")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(eval_dir: Path) -> logging.Logger:
    log = logging.getLogger("qpe_eval")
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    log.addHandler(ch)
    fh = logging.FileHandler(eval_dir / "evaluation_log.txt", mode="w")
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
# Z-R relationships
# ---------------------------------------------------------------------------

ZR_RELATIONS = [
    ("Z-R Marshall-Palmer",      200.0, 1.6),
    ("Z-R Tropical (NWS Miami)", 300.0, 1.35),
    ("Z-R WSR-88D Default",      300.0, 1.4),
]


def zr_predict(Z_dbz: np.ndarray, a: float, b: float) -> np.ndarray:
    Z_linear = 10.0 ** (Z_dbz / 10.0)
    R        = (Z_linear / a) ** (1.0 / b)
    return np.clip(R, 0.0, None)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def continuous_metrics(y_true: np.ndarray,
                        y_pred: np.ndarray) -> Dict[str, float]:
    residuals = y_pred - y_true
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae  = float(np.mean(np.abs(residuals)))
    bias = float(np.mean(residuals))
    mre  = float(np.mean(np.abs(residuals) / (np.abs(y_true) + 1e-8)))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2   = float(1.0 - ss_res / (ss_tot + 1e-8))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {
        "RMSE":        round(rmse, 5),
        "MAE":         round(mae,  5),
        "MRE":         round(mre,  5),
        "R2":          round(r2,   5),
        "Bias":        round(bias, 5),
        "Correlation": round(corr, 5),
    }


def categorical_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                         threshold: float) -> Dict[str, float]:
    obs_pos  = y_true >= threshold
    pred_pos = y_pred >= threshold
    obs_neg  = ~obs_pos
    pred_neg = ~pred_pos
    TP = float(np.sum( obs_pos &  pred_pos))
    FP = float(np.sum( obs_neg &  pred_pos))
    FN = float(np.sum( obs_pos &  pred_neg))
    TN = float(np.sum( obs_neg &  pred_neg))
    N  = TP + FP + FN + TN
    pod = TP / (TP + FN)        if (TP + FN) > 0 else 0.0
    far = FP / (TP + FP)        if (TP + FP) > 0 else 0.0
    csi = TP / (TP + FN + FP)   if (TP + FN + FP) > 0 else 0.0
    denom_hss = ((TP+FN)*(FN+TN) + (TP+FP)*(FP+TN))
    hss = (2.0*(TP*TN - FP*FN)) / denom_hss if denom_hss > 0 else 0.0
    Q   = (TP+FN)*(TP+FP)/N if N > 0 else 0.0
    denom_gss = TP + FN + FP - Q
    gss = (TP - Q) / denom_gss if denom_gss > 0 else 0.0
    return {
        "POD": round(pod, 5), "FAR": round(far, 5),
        "CSI": round(csi, 5), "HSS": round(hss, 5),
        "GSS": round(gss, 5),
        "TP": int(TP), "FP": int(FP), "FN": int(FN), "TN": int(TN),
    }


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                   thresholds: List[float] = [1.0, 5.0, 10.0, 25.0]) -> Dict:
    result = {
        "continuous":  continuous_metrics(y_true, y_pred),
        "categorical": {},
        "n_samples":   int(len(y_true)),
        "pred_range":  [round(float(y_pred.min()), 3),
                        round(float(y_pred.max()), 3)],
        "pred_mean":   round(float(y_pred.mean()), 3),
    }
    for thr in thresholds:
        result["categorical"][f"{thr:.0f}mm/h"] = categorical_metrics(
            y_true, y_pred, thr)
    return result


# ---------------------------------------------------------------------------
# CNN+KAN inference
# ---------------------------------------------------------------------------

def normalise(X: np.ndarray, means: np.ndarray,
              stds: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32).copy()
    for c in range(4):
        X[:, c, :, :] = (X[:, c, :, :] - means[c]) / (stds[c] + 1e-8)
    return X


@torch.no_grad()
def cnn_kan_predict(model: QPEModel, X_norm: np.ndarray,
                    device: torch.device,
                    batch: int = 512) -> np.ndarray:
    """Returns pred_mmh only. log1p output is not needed externally."""
    log1p_list = []
    for start in range(0, len(X_norm), batch):
        x_t = torch.from_numpy(X_norm[start:start+batch]).to(device)
        log1p_list.append(model(x_t).squeeze(1).cpu().numpy())
    pred_log1p = np.concatenate(log1p_list)
    return np.clip(np.expm1(pred_log1p), 0.0, None)


# ---------------------------------------------------------------------------
# LightGBM features + inference
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
    "Z_x_kan", "Zdiff_x_kan", "Zhigh_x_kan", "ZDR_x_kan",
]


def build_lgb_features(X_raw: np.ndarray, X_norm: np.ndarray,
                        pred_mmh: np.ndarray) -> np.ndarray:
    """
    Build (N, 23) feature matrix matching train_lightgbm_corrector.py.
    col 5 (kan_pred_log1p) = log1p(clip(pred_mmh, 0)).
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


def apply_lgb_correction(pred_mmh: np.ndarray,
                          epsilon_log: np.ndarray) -> np.ndarray:
    """
    Apply log1p-space LightGBM correction.
    R_final = clamp(expm1(log1p(R_hat) + epsilon_log), min=0)
    Must match apply_lgb_correction() in train_lightgbm_corrector.py.
    """
    log1p_corrected = np.log1p(np.clip(pred_mmh, 0.0, None)) + epsilon_log
    return np.clip(np.expm1(log1p_corrected), 0.0, None)


# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------

def print_table(results: Dict, y_true: np.ndarray,
                log: logging.Logger) -> None:
    models     = list(results.keys())
    thresholds = ["1mm/h", "5mm/h", "10mm/h", "25mm/h"]
    sep        = "=" * 100

    log.info(sep)
    log.info("QPE MODEL EVALUATION SUMMARY")
    log.info("Validation set: N=%d  y_mean=%.2f mm/h  y_max=%.2f mm/h",
             len(y_true), y_true.mean(), y_true.max())
    log.info("")
    log.info("CONTINUOUS METRICS")
    log.info("-" * 82)
    log.info(f"{'Model':<35} {'RMSE':>8} {'MAE':>8} {'MRE':>8} "
             f"{'R2':>8} {'Bias':>9}{'r':>9}")
    log.info("-" * 82)
    for name in models:
        c = results[name]["continuous"]
        log.info(f"{name:<35} {c['RMSE']:>8.3f} {c['MAE']:>8.3f} "
                 f"{c['MRE']:>8.4f} {c['R2']:>8.4f} "
                 f"{c['Bias']:>+9.3f}{c['Correlation']:>9.4f}")

    for thr in thresholds:
        log.info("")
        log.info("CATEGORICAL @ %s", thr)
        log.info("-" * 90)
        log.info(f"{'Model':<35} {'POD':>8} {'FAR':>8} {'CSI':>8} "
                 f"{'HSS':>8} {'GSS':>8}  {'TP':>6} {'FP':>6} {'FN':>6}")
        log.info("-" * 90)
        for name in models:
            cat = results[name]["categorical"][thr]
            log.info(f"{name:<35} {cat['POD']:>8.4f} {cat['FAR']:>8.4f} "
                     f"{cat['CSI']:>8.4f} {cat['HSS']:>8.4f} "
                     f"{cat['GSS']:>8.4f}  "
                     f"{cat['TP']:>6d} {cat['FP']:>6d} {cat['FN']:>6d}")
    log.info(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(data_dir: Path, model_dir: Path, eval_dir: Path,
             log: logging.Logger) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Load validation set ────────────────────────────────────────────────
    log.info("Loading validation set from %s", data_dir)
    X_val_raw = np.load(data_dir / "X_val.npy")
    y_val     = np.load(data_dir / "y_val.npy")
    log.info("Val samples : %d", len(y_val))
    log.info("y_val range : %.2f -- %.2f mm/h  (mean=%.2f)",
             y_val.min(), y_val.max(), y_val.mean())

    checkpoint = torch.load(model_dir / "best_model.pt", map_location="cpu")
    chan_means  = np.array(checkpoint["chan_means"], dtype=np.float32)
    chan_stds   = np.array(checkpoint["chan_stds"],  dtype=np.float32)
    X_val_norm  = normalise(X_val_raw, chan_means, chan_stds)
    Z_low_raw_val = X_val_raw[:, 0, CENTER, CENTER]

    results = {}

    # ── Z-R baselines ──────────────────────────────────────────────────────
    log.info("Evaluating Z-R baselines...")
    for name, a, b in ZR_RELATIONS:
        pred = zr_predict(Z_low_raw_val, a, b)
        results[name] = evaluate_model(y_val, pred)
        c = results[name]["continuous"]
        log.info("  %-35s  RMSE=%.3f  MAE=%.3f  r=%.4f",
                 name, c["RMSE"], c["MAE"], c["Correlation"])

    # ── CNN+KAN ────────────────────────────────────────────────────────────
    log.info("Evaluating CNN+KAN...")
    model = QPEModel().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    kan_pred_mmh = cnn_kan_predict(model, X_val_norm, device)
    results["CNN+KAN"] = evaluate_model(y_val, kan_pred_mmh)
    c = results["CNN+KAN"]["continuous"]
    log.info("  %-35s  RMSE=%.3f  MAE=%.3f  r=%.4f",
             "CNN+KAN", c["RMSE"], c["MAE"], c["Correlation"])

    # ── CNN+KAN + LightGBM ─────────────────────────────────────────────────
    log.info("Evaluating CNN+KAN + LightGBM...")
    lgb_path     = model_dir / "lgb_residual.pkl"
    metrics_path = model_dir / "lgb_metrics.json"

    if not lgb_path.exists():
        log.warning("lgb_residual.pkl not found at %s — skipping", lgb_path)
    else:
        # Warn if model was trained in mm/h space (wrong transform)
        if metrics_path.exists():
            with open(metrics_path) as f:
                lgb_meta = json.load(f)
            resid_space = lgb_meta.get("residual_space", "unknown")
            if resid_space != "log1p":
                log.warning(
                    "lgb_metrics.json shows residual_space='%s'. "
                    "This script expects log1p space. "
                    "Retrain with updated train_lightgbm_corrector.py.",
                    resid_space)

        with open(lgb_path, "rb") as f:
            lgb_model = pickle.load(f)

        X_lgb_val = build_lgb_features(X_val_raw, X_val_norm, kan_pred_mmh)

        assert X_lgb_val.shape[1] == len(FEATURE_NAMES), \
            f"Feature count mismatch: {X_lgb_val.shape[1]} vs {len(FEATURE_NAMES)}"

        num_iter = (lgb_model.best_iteration
                    if hasattr(lgb_model, "best_iteration")
                    and lgb_model.best_iteration > 0
                    else lgb_model.num_trees())

        # Predict log1p-space epsilon, apply inverse transform
        epsilon_log = lgb_model.predict(X_lgb_val, num_iteration=num_iter)
        final_pred  = apply_lgb_correction(kan_pred_mmh, epsilon_log)

        results["CNN+KAN+LightGBM"] = evaluate_model(y_val, final_pred)
        c = results["CNN+KAN+LightGBM"]["continuous"]
        log.info("  %-35s  RMSE=%.3f  MAE=%.3f  r=%.4f",
                 "CNN+KAN+LightGBM", c["RMSE"], c["MAE"], c["Correlation"])

    # ── Print table ────────────────────────────────────────────────────────
    log.info("")
    print_table(results, y_val, log)

    # ── Best model per metric ──────────────────────────────────────────────
    log.info("")
    log.info("BEST MODEL PER METRIC:")
    for metric in ["RMSE", "MAE", "MRE", "R2", "Bias", "Correlation"]:
        vals = {m: results[m]["continuous"][metric] for m in results}
        if metric in {"R2", "Correlation"}:
            best = max(vals, key=vals.get)
        elif metric == "Bias":
            best = min(vals, key=lambda m: abs(vals[m]))
        else:
            best = min(vals, key=vals.get)
        log.info("  %-15s : %-35s (%.4f)", metric, best, vals[best])

    # ── Save outputs ───────────────────────────────────────────────────────
    report = {
        "dataset": {
            "n_val":      int(len(y_val)),
            "y_min":      round(float(y_val.min()), 3),
            "y_max":      round(float(y_val.max()), 3),
            "y_mean":     round(float(y_val.mean()), 3),
            "thresholds": [1.0, 5.0, 10.0, 25.0],
        },
        "models": results,
    }
    report_path = eval_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Report saved: %s", report_path)

    summary_lines = ["QPE MODEL EVALUATION SUMMARY", "=" * 100,
                     f"Validation set: N={len(y_val)}  "
                     f"y_mean={y_val.mean():.2f} mm/h  "
                     f"y_max={y_val.max():.2f} mm/h", ""]
    summary_lines += [
        "CONTINUOUS METRICS", "-" * 82,
        f"{'Model':<35} {'RMSE':>8} {'MAE':>8} {'MRE':>8} "
        f"{'R2':>8} {'Bias':>9}{'r':>9}", "-" * 82,
    ]
    for name, res in results.items():
        c = res["continuous"]
        summary_lines.append(
            f"{name:<35} {c['RMSE']:>8.3f} {c['MAE']:>8.3f} "
            f"{c['MRE']:>8.4f} {c['R2']:>8.4f} "
            f"{c['Bias']:>+9.3f}{c['Correlation']:>9.4f}")
    for thr in ["1mm/h", "5mm/h", "10mm/h", "25mm/h"]:
        summary_lines += ["", f"CATEGORICAL @ {thr}", "-" * 90,
                          f"{'Model':<35} {'POD':>8} {'FAR':>8} {'CSI':>8} "
                          f"{'HSS':>8} {'GSS':>8}  {'TP':>6} {'FP':>6} {'FN':>6}",
                          "-" * 90]
        for name, res in results.items():
            cat = res["categorical"][thr]
            summary_lines.append(
                f"{name:<35} {cat['POD']:>8.4f} {cat['FAR']:>8.4f} "
                f"{cat['CSI']:>8.4f} {cat['HSS']:>8.4f} {cat['GSS']:>8.4f}  "
                f"{cat['TP']:>6d} {cat['FP']:>6d} {cat['FN']:>6d}")

    summary_path = eval_dir / "evaluation_summary.txt"
    summary_path.write_text("\n".join(summary_lines))
    log.info("Summary saved: %s", summary_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate QPE models vs Z-R baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_dir",  type=Path,
                   default=_ROOT / "data" / "dataset" / "processed_T4_improved")
    p.add_argument("--model_dir", type=Path,
                   default=_ROOT / "models" / "qpe_cnn_kan")
    p.add_argument("--eval_dir",  type=Path,
                   default=_ROOT / "eval")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.eval_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(args.eval_dir)
    log.info("QPE Evaluation started.")
    log.info("Config: %s", vars(args))
    evaluate(
        data_dir  = args.data_dir,
        model_dir = args.model_dir,
        eval_dir  = args.eval_dir,
        log       = log,
    )


if __name__ == "__main__":
    main()