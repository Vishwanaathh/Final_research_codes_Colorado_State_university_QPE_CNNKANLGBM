# -*- coding: utf-8 -*-
"""
plot_evaluation_all_v2.py
=========================
Visualization for ALL QPE models — v2 (37-feature LightGBM corrector).

Plots generated (saved to eval/all_plots_v2/):
    01_scatter_grouped.png
    02_continuous_metrics.png          — overall RMSE/MAE/Bias
    02b_continuous_light.png           — per-category: light (1-10 mm/h)
    02b_continuous_moderate.png        — per-category: moderate (10-25 mm/h)
    02b_continuous_heavy.png           — per-category: heavy (25-50 mm/h)
    02b_continuous_extreme.png         — per-category: extreme (>50 mm/h)
    02b_continuous_all.png             — per-category: all (>=1 mm/h)
    03_categorical_01mmh.png  ..  03_categorical_25mmh.png
    04_skill_01mmh.png        ..  04_skill_25mmh.png
    05_pdf.png
    06_reliability.png
"""

import sys
from pathlib import Path

_ROOT          = Path(__file__).resolve().parent.parent
_AI_SCRIPT_DIR = _ROOT / "AI_Script"
if str(_AI_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_AI_SCRIPT_DIR))

import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
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
# Style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       200,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})

MODEL_COLORS  = ["#e74c3c", "#e67e22", "#f1c40f", "#3498db",
                 "#2ecc71", "#9b59b6", "#1abc9c"]
MODEL_MARKERS = ["o", "s", "^", "D", "*", "P", "X"]

CENTER = 4

RAIN_CATEGORIES = {
    "light\n(1-10)":     (1.0,    10.0),
    "moderate\n(10-25)": (10.0,   25.0),
    "heavy\n(25-50)":    (25.0,   50.0),
    "extreme\n(>50)":    (50.0,   9999.0),
    "all\n(>=1)":        (1.0,    9999.0),
}

CAT_FILENAMES = {
    "light\n(1-10)":     "02b_continuous_light.png",
    "moderate\n(10-25)": "02b_continuous_moderate.png",
    "heavy\n(25-50)":    "02b_continuous_heavy.png",
    "extreme\n(>50)":    "02b_continuous_extreme.png",
    "all\n(>=1)":        "02b_continuous_all.png",
}


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

def normalise(X, means, stds):
    X = X.astype(np.float32).copy()
    for c in range(4):
        X[:, c] = (X[:, c] - means[c]) / (stds[c] + 1e-8)
    return X

def zr_predict(Z_dbz, a, b):
    return np.clip((10.0 ** (Z_dbz / 10.0) / a) ** (1.0 / b), 0.0, None)

@torch.no_grad()
def dl_predict(model, X_norm, device, batch=512):
    model.eval()
    out = []
    for s in range(0, len(X_norm), batch):
        x = torch.from_numpy(X_norm[s:s+batch]).to(device)
        out.append(model(x).squeeze(1).cpu().numpy())
    return np.clip(np.expm1(np.concatenate(out)), 0.0, None)


def build_lgb_features(X_raw, X_norm, pred_mmh):
    """37-feature matrix — must match train_lightgbm_v2.py exactly."""
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


def apply_lgb_correction(pred_mmh, epsilon_log):
    return np.clip(np.expm1(np.log1p(np.clip(pred_mmh, 0.0, None)) + epsilon_log), 0.0, None)


def categorical_scores(y_true, y_pred, thr):
    TP = float(np.sum((y_true >= thr) & (y_pred >= thr)))
    FP = float(np.sum((y_true <  thr) & (y_pred >= thr)))
    FN = float(np.sum((y_true >= thr) & (y_pred <  thr)))
    TN = float(np.sum((y_true <  thr) & (y_pred <  thr)))
    N  = TP + FP + FN + TN
    pod = TP/(TP+FN)      if (TP+FN)    > 0 else 0.
    far = FP/(TP+FP)      if (TP+FP)    > 0 else 0.
    csi = TP/(TP+FN+FP)   if (TP+FN+FP) > 0 else 0.
    dh  = (TP+FN)*(FN+TN) + (TP+FP)*(FP+TN)
    hss = 2*(TP*TN-FP*FN)/dh if dh > 0 else 0.
    Q   = (TP+FN)*(TP+FP)/N  if N  > 0 else 0.
    dg  = TP+FN+FP-Q
    gss = (TP-Q)/dg if dg > 0 else 0.
    return dict(POD=pod, FAR=far, CSI=csi, HSS=hss, GSS=gss)

def save(fig, path, name):
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")

def get_color(i):  return MODEL_COLORS[i % len(MODEL_COLORS)]
def get_marker(i): return MODEL_MARKERS[i % len(MODEL_MARKERS)]


# ---------------------------------------------------------------------------
# Load predictions
# ---------------------------------------------------------------------------

def load_predictions(data_dir, models_dir, device):
    X_val_raw = np.load(data_dir / "X_val.npy")
    y_val     = np.load(data_dir / "y_val.npy")

    ckpt_path = models_dir / "qpe_cnn_kan" / "best_model.pt"
    if not ckpt_path.exists():
        with open(data_dir / "dataset_stats.json") as f:
            stats = json.load(f)
        names = ["Z_low", "ZDR_low", "Z_high", "ZDR_high"]
        chan_means = np.array([stats[n]["mean"] for n in names], np.float32)
        chan_stds  = np.array([stats[n]["std"]  for n in names], np.float32)
        ckpt = None
    else:
        ckpt       = torch.load(ckpt_path, map_location="cpu")
        chan_means  = np.array(ckpt["chan_means"], dtype=np.float32)
        chan_stds   = np.array(ckpt["chan_stds"],  dtype=np.float32)

    X_val_norm = normalise(X_val_raw, chan_means, chan_stds)
    Z_raw      = X_val_raw[:, 0, CENTER, CENTER]
    preds      = {}

    # Z-R baselines
    for name, a, b in [
        ("Z-R Marshall-Palmer",      200.0, 1.6),
        ("Z-R Tropical (NWS Miami)", 300.0, 1.35),
        ("Z-R WSR-88D Default",      300.0, 1.4),
    ]:
        preds[name] = zr_predict(Z_raw, a, b)
        print(f"  \u2713 {name}")

    # CNN+KAN
    kan_mmh = None
    if _HAS_CNNKAN and ckpt is not None:
        model_ck = CNNKANModel().to(device)
        model_ck.load_state_dict(ckpt["model_state"])
        kan_mmh = dl_predict(model_ck, X_val_norm, device)
        preds["CNN+KAN"] = kan_mmh
        print("  \u2713 CNN+KAN")

        # LightGBM v2
        lgb_path = models_dir / "qpe_cnn_kan" / "lgb_residual_v2.pkl"
        if lgb_path.exists():
            with open(lgb_path, "rb") as f:
                lgb_model = pickle.load(f)
            feats       = build_lgb_features(X_val_raw, X_val_norm, kan_mmh)
            num_iter    = (lgb_model.best_iteration
                          if hasattr(lgb_model, "best_iteration")
                          and lgb_model.best_iteration > 0
                          else lgb_model.num_trees())
            epsilon_log = lgb_model.predict(feats, num_iteration=num_iter)
            preds["CNN+KAN+LightGBM v2"] = apply_lgb_correction(kan_mmh, epsilon_log)
            print("  \u2713 CNN+KAN+LightGBM v2")
        else:
            print("  \u2717 CNN+KAN+LightGBM v2 \u2014 lgb_residual_v2.pkl not found")
    else:
        print("  \u2717 CNN+KAN \u2014 checkpoint not found")

    # RQPENetD1
    rq_path = models_dir / "rqpenetd1" / "best_model.pt"
    if _HAS_RQPENET and rq_path.exists():
        ckpt_rq  = torch.load(rq_path, map_location="cpu")
        cfg      = ckpt_rq.get("model_config", {})
        model_rq = RQPENetD1(
            growth_rate   = cfg.get("growth_rate",   12),
            block_config  = tuple(cfg.get("block_config", [6,12,12,8])),
            stem_channels = cfg.get("stem_channels", 32),
        ).to(device)
        model_rq.load_state_dict(ckpt_rq["model_state"])
        rq_means = np.array(ckpt_rq.get("chan_means", chan_means.tolist()), np.float32)
        rq_stds  = np.array(ckpt_rq.get("chan_stds",  chan_stds.tolist()),  np.float32)
        preds["RQPENetD1"] = dl_predict(model_rq, normalise(X_val_raw, rq_means, rq_stds), device)
        print("  \u2713 RQPENetD1")
    else:
        print("  \u2717 RQPENetD1 \u2014 checkpoint not found")

    # ConvLSTMQPE
    cl_path = models_dir / "convlstm_qpe" / "best_model.pt"
    if _HAS_CONVLSTM and cl_path.exists():
        ckpt_cl  = torch.load(cl_path, map_location="cpu")
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
        preds["ConvLSTMQPE"] = dl_predict(model_cl, normalise(X_val_raw, cl_means, cl_stds), device)
        print("  \u2713 ConvLSTMQPE")
    else:
        print("  \u2717 ConvLSTMQPE \u2014 checkpoint not found")

    return X_val_raw, y_val, preds


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_scatter(preds, y_true, out):
    names = list(preds.keys())
    ncols = 3
    nrows = (len(names) + ncols - 1) // ncols
    vmax  = min(float(y_true.max()), 150.0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 6))
    axes = axes.ravel()
    for i, (name, pred) in enumerate(preds.items()):
        ax = axes[i]
        h  = ax.hist2d(y_true, pred, bins=60,
                       range=[[0, vmax], [0, vmax]],
                       norm=LogNorm(), cmap="plasma")
        plt.colorbar(h[3], ax=ax, label="Count")
        ax.plot([0, vmax], [0, vmax], "w--", lw=1.5)
        r2   = 1 - np.sum((pred-y_true)**2) / np.sum((y_true-y_true.mean())**2)
        rmse = np.sqrt(np.mean((pred-y_true)**2))
        bias = np.mean(pred-y_true)
        corr = np.corrcoef(y_true, pred)[0, 1]
        ax.set_title(
            f"{name}\nRMSE={rmse:.2f}  R\u00b2={r2:.3f}  "
            f"Bias={bias:+.2f}  r={corr:.3f}", fontsize=10)
        ax.set_xlabel("Observed (mm/h)")
        ax.set_ylabel("Predicted (mm/h)")
        ax.set_xlim(0, vmax); ax.set_ylim(0, vmax)
    for j in range(len(names), len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    save(fig, out, "01_scatter_grouped.png")


def plot_continuous_bars(preds, y_true, out):
    """Overall RMSE / MAE / Bias — one bar per model."""
    names  = list(preds.keys())
    x      = np.arange(len(names))
    w      = 0.55
    colors = [get_color(i) for i in range(len(names))]
    metrics = {n: {"RMSE": float(np.sqrt(np.mean((p-y_true)**2))),
                   "MAE":  float(np.mean(np.abs(p-y_true))),
                   "Bias": float(np.mean(p-y_true))}
               for n, p in preds.items()}

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle("Continuous Metrics \u2014 All Models (Overall)", fontsize=13, y=1.02)

    for ax, (metric, ylabel) in zip(axes, [
        ("RMSE", "RMSE (mm/h)"),
        ("MAE",  "MAE (mm/h)"),
        ("Bias", "Bias (mm/h)"),
    ]):
        vals = [metrics[n][metric] for n in names]
        bars = ax.bar(x, vals, width=w, color=colors, edgecolor="white", linewidth=0.5)
        if metric == "Bias":
            ax.axhline(0, color="black", lw=1.2, ls="--")
        best_i = int(np.argmin(np.abs(vals)))
        bars[best_i].set_edgecolor("black"); bars[best_i].set_linewidth(2.5)
        ax.set_title(metric, fontsize=12)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [n.replace(" ", "\n").replace("+", "+\n") for n in names], fontsize=8)
        ypad = max(abs(v) for v in vals) * 0.02 + 0.1
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (ypad if val >= 0 else -ypad*3),
                    f"{val:.2f}", ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=9, fontweight="bold")
    plt.tight_layout()
    save(fig, out, "02_continuous_metrics.png")


def plot_continuous_by_category(preds, y_true, out):
    """One figure per rainfall category — 1×3 (RMSE, MAE, Bias)."""
    names  = list(preds.keys())
    x      = np.arange(len(names))
    w      = 0.55
    colors = [get_color(i) for i in range(len(names))]

    # Pre-compute
    cat_metrics = {}
    for name, pred in preds.items():
        cat_metrics[name] = {}
        for cat_label, (lo, hi) in RAIN_CATEGORIES.items():
            mask = (y_true >= lo) & (y_true < hi)
            n    = int(mask.sum())
            if n < 5:
                cat_metrics[name][cat_label] = {"RMSE": None, "MAE": None, "Bias": None, "N": n}
                continue
            res = pred[mask] - y_true[mask]
            cat_metrics[name][cat_label] = {
                "RMSE": float(np.sqrt(np.mean(res**2))),
                "MAE":  float(np.mean(np.abs(res))),
                "Bias": float(np.mean(res)),
                "N":    n,
            }

    for cat_label, filename in CAT_FILENAMES.items():
        lo, hi = RAIN_CATEGORIES[cat_label]
        n_obs  = int(((y_true >= lo) & (y_true < hi)).sum())
        title  = cat_label.replace("\n", " ")

        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        fig.suptitle(f"Continuous Metrics \u2014 {title} mm/h  (N={n_obs:,})",
                     fontsize=13, y=1.02)

        for ax, (metric, ylabel, has_zero, better) in zip(axes, [
            ("RMSE", "RMSE (mm/h)", False, "lower"),
            ("MAE",  "MAE (mm/h)",  False, "lower"),
            ("Bias", "Bias (mm/h)", True,  "zero"),
        ]):
            vals      = [cat_metrics[n][cat_label].get(metric) for n in names]
            plot_vals = [v if v is not None else 0.0 for v in vals]

            bars = ax.bar(x, plot_vals, width=w, color=colors,
                          edgecolor="white", linewidth=0.5)
            if has_zero:
                ax.axhline(0, color="black", lw=1.2, ls="--")

            finite = [(i, v) for i, v in enumerate(vals) if v is not None]
            if finite:
                best_i = (min(finite, key=lambda t: abs(t[1]))[0]
                          if better == "zero"
                          else min(finite, key=lambda t: t[1])[0])
                bars[best_i].set_edgecolor("black")
                bars[best_i].set_linewidth(2.5)

            ax.set_title(metric, fontsize=12)
            ax.set_ylabel(ylabel)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [n.replace(" ", "\n").replace("+", "+\n") for n in names], fontsize=8)

            ypad = max((abs(v) for v in plot_vals), default=1) * 0.02 + 0.1
            for bar, val in zip(bars, vals):
                if val is None:
                    ax.text(bar.get_x() + bar.get_width()/2, ypad,
                            "n/a", ha="center", va="bottom",
                            fontsize=8, color="gray")
                else:
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + (ypad if val >= 0 else -ypad*3),
                            f"{val:.2f}", ha="center",
                            va="bottom" if val >= 0 else "top",
                            fontsize=9, fontweight="bold")
        plt.tight_layout()
        save(fig, out, filename)


def plot_categorical_bars(preds, y_true, out, thresholds=(1, 5, 10, 25)):
    names  = list(preds.keys())
    x      = np.arange(len(names))
    w      = 0.55
    colors = [get_color(i) for i in range(len(names))]
    for thr in thresholds:
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        fig.suptitle(f"Categorical Metrics @ {thr} mm/h \u2014 All Models",
                     fontsize=13, y=1.02)
        scores = {n: categorical_scores(y_true, preds[n], thr) for n in names}
        for ax, (metric, ylabel, good) in zip(axes, [
            ("POD", "Probability of Detection", "higher"),
            ("FAR", "False Alarm Ratio",         "lower"),
            ("CSI", "Critical Success Index",    "higher"),
        ]):
            vals = [scores[n][metric] for n in names]
            bars = ax.bar(x, vals, width=w, color=colors,
                          edgecolor="white", linewidth=0.5)
            best_i = int(np.argmax(vals) if good == "higher" else np.argmin(vals))
            bars[best_i].set_edgecolor("black"); bars[best_i].set_linewidth(2.5)
            ax.set_title(f"{metric} @ {thr} mm/h", fontsize=12)
            ax.set_ylabel(f"{ylabel} ({'higher' if good=='higher' else 'lower'} = better)")
            ax.set_ylim(0, 1.18)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [n.replace(" ", "\n").replace("+", "+\n") for n in names], fontsize=8)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.013,
                        f"{val:.3f}", ha="center", va="bottom",
                        fontsize=9, fontweight="bold")
        plt.tight_layout()
        save(fig, out, f"03_categorical_{thr:02d}mmh.png")


def plot_skill_scores(preds, y_true, out, thresholds=(1, 5, 10, 25)):
    names  = list(preds.keys())
    x      = np.arange(len(names))
    w      = 0.55
    colors = [get_color(i) for i in range(len(names))]
    for thr in thresholds:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(f"Skill Scores @ {thr} mm/h \u2014 All Models",
                     fontsize=13, y=1.02)
        scores = {n: categorical_scores(y_true, preds[n], thr) for n in names}
        for ax, (metric, label) in zip(axes, [
            ("HSS", "Heidke Skill Score"),
            ("GSS", "Gilbert Skill Score (ETS)"),
        ]):
            vals = [scores[n][metric] for n in names]
            bars = ax.bar(x, vals, width=w, color=colors,
                          edgecolor="white", linewidth=0.5)
            best_i = int(np.argmax(vals))
            bars[best_i].set_edgecolor("black"); bars[best_i].set_linewidth(2.5)
            ax.axhline(0, color="gray", lw=1.0, ls="--")
            ax.set_title(f"{metric} \u2014 {label} @ {thr} mm/h", fontsize=12)
            ax.set_ylabel(f"{metric} (higher = better)")
            ax.set_xticks(x)
            ax.set_xticklabels(
                [n.replace(" ", "\n").replace("+", "+\n") for n in names], fontsize=8)
            for bar, val in zip(bars, vals):
                ypos = bar.get_height() + 0.012 if val >= 0 else bar.get_height() - 0.06
                ax.text(bar.get_x() + bar.get_width()/2, ypos,
                        f"{val:.3f}", ha="center", va="bottom",
                        fontsize=9, fontweight="bold")
        plt.tight_layout()
        save(fig, out, f"04_skill_{thr:02d}mmh.png")


def plot_pdf(preds, y_true, out):
    colors = [get_color(i) for i in range(len(preds))]
    bins   = np.logspace(np.log10(0.5), np.log10(200), 40)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(y_true, bins=bins, density=True, alpha=0.35,
            color="black", label="Observed", histtype="stepfilled")
    for i, (name, pred) in enumerate(preds.items()):
        ax.hist(pred, bins=bins, density=True, alpha=0.9,
                color=colors[i], histtype="step", lw=2.0, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Rain Rate (mm/h)")
    ax.set_ylabel("Density")
    ax.set_title("Rain Rate Distribution \u2014 Observed vs All Models")
    ax.legend(fontsize=8)
    plt.tight_layout()
    save(fig, out, "05_pdf.png")


def plot_reliability(preds, y_true, out):
    colors = [get_color(i) for i in range(len(preds))]
    bins   = np.percentile(y_true, np.linspace(5, 95, 20))
    bins   = np.unique(np.concatenate([[0], bins, [y_true.max()+1]]))
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot([0, y_true.max()], [0, y_true.max()],
            "k--", lw=2.0, label="Perfect (1:1)", zorder=0)
    for i, (name, pred) in enumerate(preds.items()):
        obs_m, pred_m = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (y_true >= lo) & (y_true < hi)
            if mask.sum() >= 5:
                obs_m.append(float(y_true[mask].mean()))
                pred_m.append(float(pred[mask].mean()))
        ax.plot(obs_m, pred_m,
                marker=get_marker(i), ms=7, lw=2.0,
                color=colors[i], label=name)
    ax.set_xlabel("Mean Observed Rain Rate per Bin (mm/h)")
    ax.set_ylabel("Mean Predicted Rain Rate per Bin (mm/h)")
    ax.set_title("Reliability Diagram (Conditional Mean Bias) \u2014 All Models")
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    save(fig, out, "06_reliability.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data_dir   = _ROOT / "data" / "dataset" / "processed_T4_improved"
    models_dir = _ROOT / "models"
    out_dir    = _ROOT / "eval" / "all_plots_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Output : {out_dir}")
    print("\nLoading predictions...")

    X_val_raw, y_val, preds = load_predictions(data_dir, models_dir, device)
    print(f"\n{len(preds)} models loaded. Generating plots...\n")

    plot_scatter(preds, y_val, out_dir)
    plot_continuous_bars(preds, y_val, out_dir)
    plot_continuous_by_category(preds, y_val, out_dir)
    plot_categorical_bars(preds, y_val, out_dir)
    plot_skill_scores(preds, y_val, out_dir)
    plot_pdf(preds, y_val, out_dir)
    plot_reliability(preds, y_val, out_dir)

    n_plots = 1 + 1 + 5 + 4 + 4 + 1 + 1
    print(f"\nDone. {n_plots} plots saved to: {out_dir}")


if __name__ == "__main__":
    main()