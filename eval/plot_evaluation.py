# -*- coding: utf-8 -*-
"""
plot_evaluation.py
==================
QPE evaluation visualization — Z-R baselines + CNN+KAN + CNN+KAN+LightGBM

Plots generated
---------------
    01_scatter_grouped.png      — 2x3 grid of density scatter plots
    02_continuous_metrics.png   — RMSE / MAE / Bias bar chart
    03_categorical_01mmh.png    — POD / FAR / CSI at 1  mm/h
    03_categorical_05mmh.png    — POD / FAR / CSI at 5  mm/h
    03_categorical_10mmh.png    — POD / FAR / CSI at 10 mm/h
    03_categorical_25mmh.png    — POD / FAR / CSI at 25 mm/h
    04_skill_01mmh.png          — HSS / GSS at 1  mm/h
    04_skill_05mmh.png          — HSS / GSS at 5  mm/h
    04_skill_10mmh.png          — HSS / GSS at 10 mm/h
    04_skill_25mmh.png          — HSS / GSS at 25 mm/h
    05_pdf.png                  — rain rate distribution
    06_reliability.png          — conditional mean bias diagram

Output: eval/plots/

Changes
-------
  1. Dead code removed from build_lgb_features().
     An unreachable 17-feature block after the return statement has been
     deleted. It was a leftover from an earlier feature set and never
     executed, but made the function look broken.

  2. kan_pred_log1p (col 5) aligned with train_lightgbm_corrector.py.
     col 5 is now explicitly log1p(clip(pred_mmh, 0)) — matching what
     the LightGBM model was trained on. The pred_log1p argument from the
     CNN forward pass is no longer passed to build_lgb_features() since
     it is not used there.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch

try:
    from cnn_feature_extractor import RadarCNN
    from kan_rainfall_predictor import RainfallKAN
except ImportError as e:
    sys.exit(f"[FATAL] Cannot import from {_AI_SCRIPT_DIR}\n{e}")


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

MODEL_COLORS  = ["#e74c3c", "#e67e22", "#f1c40f", "#3498db", "#2ecc71"]
MODEL_MARKERS = ["o", "s", "^", "D", "*"]
CENTER = 4


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class QPEModel(torch.nn.Module):
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
    """Z_dbz must be raw dBZ values — NOT normalised."""
    return np.clip((10.0 ** (Z_dbz / 10.0) / a) ** (1.0 / b), 0.0, None)

@torch.no_grad()
def cnn_kan_pred(model, X_norm, device, batch=512):
    """Returns (pred_mmh, pred_log1p). X_norm must be normalised."""
    preds = []
    for i in range(0, len(X_norm), batch):
        x = torch.from_numpy(X_norm[i:i+batch]).to(device)
        preds.append(model(x).squeeze(1).cpu().numpy())
    log1p = np.concatenate(preds)
    return np.clip(np.expm1(log1p), 0.0, None), log1p


def build_lgb_features(X_raw, X_norm, pred_mmh):
    """
    Build (N, 23) LightGBM feature matrix.
    Must exactly match FEATURE_NAMES in train_lightgbm_corrector.py.

    col 5 (kan_pred_log1p) = log1p(clip(pred_mmh, 0)) — matches what
    the LightGBM model was trained on in train_lightgbm_corrector.py.

    Note: pred_log1p from the CNN forward pass is NOT used here because
    train_lightgbm_corrector.py computes col 5 from pred_mmh, not from
    the raw model output.
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
    pred_log1p_pos = np.log1p(pred_mmh_pos)          # col 5 — matches train script
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



def apply_lgb_correction(pred_mmh, epsilon_log):
    """
    Apply log1p-space LightGBM correction.
    R_final = clamp(expm1(log1p(R_hat) + epsilon_log), min=0)
    Matches apply_lgb_correction() in train_lightgbm_corrector.py.
    """
    import numpy as np
    log1p_corrected = np.log1p(np.clip(pred_mmh, 0.0, None)) + epsilon_log
    return np.clip(np.expm1(log1p_corrected), 0.0, None)


def categorical_scores(y_true, y_pred, thr):
    TP = float(np.sum((y_true >= thr) & (y_pred >= thr)))
    FP = float(np.sum((y_true <  thr) & (y_pred >= thr)))
    FN = float(np.sum((y_true >= thr) & (y_pred <  thr)))
    TN = float(np.sum((y_true <  thr) & (y_pred <  thr)))
    N  = TP + FP + FN + TN
    pod = TP / (TP + FN)        if (TP + FN)        > 0 else 0.
    far = FP / (TP + FP)        if (TP + FP)        > 0 else 0.
    csi = TP / (TP + FN + FP)   if (TP + FN + FP)   > 0 else 0.
    dh  = (TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)
    hss = 2 * (TP * TN - FP * FN) / dh if dh > 0 else 0.
    Q   = (TP + FN) * (TP + FP) / N    if N  > 0 else 0.
    dg  = TP + FN + FP - Q
    gss = (TP - Q) / dg if dg > 0 else 0.
    return dict(POD=pod, FAR=far, CSI=csi, HSS=hss, GSS=gss)

def save(fig, path, name):
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


# ---------------------------------------------------------------------------
# 1. Scatter
# ---------------------------------------------------------------------------
def plot_scatter(preds, y_true, out):
    names = list(preds.keys())
    n     = len(names)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    vmax  = min(float(y_true.max()), 150.0)

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 6))
    axes = axes.ravel()

    for i, (name, pred) in enumerate(preds.items()):
        ax = axes[i]
        h  = ax.hist2d(y_true, pred, bins=60,
                       range=[[0, vmax], [0, vmax]],
                       norm=LogNorm(), cmap="plasma")
        plt.colorbar(h[3], ax=ax, label="Count")
        ax.plot([0, vmax], [0, vmax], "w--", lw=1.5, label="1:1")
        r2   = 1 - np.sum((pred-y_true)**2) / np.sum((y_true-y_true.mean())**2)
        rmse = np.sqrt(np.mean((pred-y_true)**2))
        bias = np.mean(pred-y_true)
        corr = np.corrcoef(y_true, pred)[0, 1]
        ax.set_title(
            f"{name}\n"
            f"RMSE={rmse:.2f}  R²={r2:.3f}  Bias={bias:+.2f}  r={corr:.3f}",
            fontsize=10)
        ax.set_xlabel("Observed (mm/h)")
        ax.set_ylabel("Predicted (mm/h)")
        ax.set_xlim(0, vmax); ax.set_ylim(0, vmax)
        ax.legend(fontsize=8)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    save(fig, out, "01_scatter_grouped.png")


# ---------------------------------------------------------------------------
# 2. Continuous metrics
# ---------------------------------------------------------------------------
def plot_continuous_bars(preds, y_true, out):
    names = list(preds.keys())
    x     = np.arange(len(names))
    w     = 0.55

    metrics = {}
    for name, pred in preds.items():
        res = pred - y_true
        metrics[name] = {
            "RMSE": float(np.sqrt(np.mean(res**2))),
            "MAE":  float(np.mean(np.abs(res))),
            "Bias": float(np.mean(res)),
        }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Continuous Metrics — Z-R Baselines vs Deep Learning",
                 fontsize=13, y=1.02)

    for ax, (metric, ylabel) in zip(axes, [
        ("RMSE", "RMSE (mm/h)"),
        ("MAE",  "MAE (mm/h)"),
        ("Bias", "Bias (mm/h)"),
    ]):
        vals = [metrics[n][metric] for n in names]
        bars = ax.bar(x, vals, width=w,
                      color=MODEL_COLORS[:len(names)],
                      edgecolor="white", linewidth=0.5)
        if metric == "Bias":
            ax.axhline(0, color="black", lw=1.2, ls="--")
        best_i = int(np.argmin(np.abs(vals)))
        bars[best_i].set_edgecolor("black")
        bars[best_i].set_linewidth(2.5)
        ax.set_title(metric, fontsize=12)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [n.replace(" ", "\n").replace("+", "+\n") for n in names],
            fontsize=9)
        ypad = max(abs(v) for v in vals) * 0.02
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + ypad,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
    plt.tight_layout()
    save(fig, out, "02_continuous_metrics.png")


# ---------------------------------------------------------------------------
# 3. Categorical bars
# ---------------------------------------------------------------------------
def plot_categorical_bars(preds, y_true, out, thresholds=[1, 5, 10, 25]):
    names = list(preds.keys())
    x = np.arange(len(names))
    w = 0.55

    for thr in thresholds:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Categorical Metrics @ {thr} mm/h",
                     fontsize=13, y=1.02)
        scores = {n: categorical_scores(y_true, preds[n], thr) for n in names}

        for ax, (metric, ylabel, good) in zip(axes, [
            ("POD", "Probability of Detection (higher = better)", "higher"),
            ("FAR", "False Alarm Ratio (lower = better)",         "lower"),
            ("CSI", "Critical Success Index (higher = better)",   "higher"),
        ]):
            vals = [scores[n][metric] for n in names]
            bars = ax.bar(x, vals, width=w,
                          color=MODEL_COLORS[:len(names)],
                          edgecolor="white", linewidth=0.5)
            best_i = int(np.argmax(vals) if good == "higher" else np.argmin(vals))
            bars[best_i].set_edgecolor("black")
            bars[best_i].set_linewidth(2.5)
            ax.set_title(f"{metric} @ {thr} mm/h", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_ylim(0, 1.18)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [n.replace(" ", "\n").replace("+", "+\n") for n in names],
                fontsize=9)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.013,
                        f"{val:.3f}", ha="center", va="bottom",
                        fontsize=10, fontweight="bold")
        plt.tight_layout()
        save(fig, out, f"03_categorical_{thr:02d}mmh.png")


# ---------------------------------------------------------------------------
# 4. Skill scores
# ---------------------------------------------------------------------------
def plot_skill_scores(preds, y_true, out, thresholds=[1, 5, 10, 25]):
    names = list(preds.keys())
    x = np.arange(len(names))
    w = 0.55

    for thr in thresholds:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Skill Scores @ {thr} mm/h", fontsize=13, y=1.02)
        scores = {n: categorical_scores(y_true, preds[n], thr) for n in names}

        for ax, (metric, label) in zip(axes, [
            ("HSS", "Heidke Skill Score"),
            ("GSS", "Gilbert Skill Score (ETS)"),
        ]):
            vals = [scores[n][metric] for n in names]
            bars = ax.bar(x, vals, width=w,
                          color=MODEL_COLORS[:len(names)],
                          edgecolor="white", linewidth=0.5)
            best_i = int(np.argmax(vals))
            bars[best_i].set_edgecolor("black")
            bars[best_i].set_linewidth(2.5)
            ax.axhline(0, color="gray", lw=1.0, ls="--")
            ax.set_title(f"{metric} — {label} @ {thr} mm/h", fontsize=12)
            ax.set_ylabel(f"{metric} (higher = better)", fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [n.replace(" ", "\n").replace("+", "+\n") for n in names],
                fontsize=9)
            for bar, val in zip(bars, vals):
                ypos = bar.get_height() + 0.012 if val >= 0 else bar.get_height() - 0.06
                ax.text(bar.get_x() + bar.get_width()/2,
                        ypos, f"{val:.3f}",
                        ha="center", va="bottom",
                        fontsize=10, fontweight="bold")
        plt.tight_layout()
        save(fig, out, f"04_skill_{thr:02d}mmh.png")


# ---------------------------------------------------------------------------
# 5. PDF
# ---------------------------------------------------------------------------
def plot_pdf(preds, y_true, out):
    bins = np.logspace(np.log10(0.5), np.log10(200), 40)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(y_true, bins=bins, density=True, alpha=0.35,
            color="black", label="Observed", histtype="stepfilled")
    for i, (name, pred) in enumerate(preds.items()):
        ax.hist(pred, bins=bins, density=True, alpha=0.9,
                color=MODEL_COLORS[i], histtype="step", lw=2.0, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Rain Rate (mm/h)")
    ax.set_ylabel("Density")
    ax.set_title("Rain Rate Distribution — Observed vs Predicted")
    ax.legend(fontsize=9)
    plt.tight_layout()
    save(fig, out, "05_pdf.png")


# ---------------------------------------------------------------------------
# 6. Reliability diagram
# ---------------------------------------------------------------------------
def plot_reliability(preds, y_true, out):
    bins = np.percentile(y_true, np.linspace(5, 95, 20))
    bins = np.unique(np.concatenate([[0], bins, [y_true.max()+1]]))

    fig, ax = plt.subplots(figsize=(9, 7))
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
                marker=MODEL_MARKERS[i], ms=7, lw=2.0,
                color=MODEL_COLORS[i], label=name)

    ax.set_xlabel("Mean Observed Rain Rate per Bin (mm/h)")
    ax.set_ylabel("Mean Predicted Rain Rate per Bin (mm/h)")
    ax.set_title("Reliability Diagram (Conditional Mean Bias)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    save(fig, out, "06_reliability.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    data_dir  = _ROOT / "data" / "dataset" / "processed_T4_improved"
    model_dir = _ROOT / "models" / "qpe_cnn_kan"
    out_dir   = _ROOT / "eval" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading data...")
    X_val_raw = np.load(data_dir / "X_val.npy")
    y_val     = np.load(data_dir / "y_val.npy")

    ckpt  = torch.load(model_dir / "best_model.pt", map_location="cpu")
    means = np.array(ckpt["chan_means"], dtype=np.float32)
    stds  = np.array(ckpt["chan_stds"],  dtype=np.float32)

    X_val_norm = normalise(X_val_raw, means, stds)
    Z_raw      = X_val_raw[:, 0, CENTER, CENTER]   # raw dBZ for Z-R

    # Z-R baselines
    print("Z-R predictions...")
    preds = {
        "Z-R Marshall-Palmer":      zr_predict(Z_raw, 200.0, 1.6),
        "Z-R Tropical (NWS Miami)": zr_predict(Z_raw, 300.0, 1.35),
        "Z-R WSR-88D Default":      zr_predict(Z_raw, 300.0, 1.4),
    }

    # CNN+KAN
    print("CNN+KAN...")
    model = QPEModel().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    kan_pred, _ = cnn_kan_pred(model, X_val_norm, device)
    preds["CNN+KAN"] = kan_pred

    # CNN+KAN + LightGBM
    lgb_path = model_dir / "lgb_residual.pkl"
    if lgb_path.exists():
        print("LightGBM...")
        with open(lgb_path, "rb") as f:
            lgb_model = pickle.load(f)
        feats    = build_lgb_features(X_val_raw, X_val_norm, kan_pred)
        epsilon_log = lgb_model.predict(feats, num_iteration=lgb_model.best_iteration)
        preds["CNN+KAN+LightGBM"] = apply_lgb_correction(kan_pred, epsilon_log)
    else:
        print("  WARNING: lgb_residual.pkl not found — skipping LightGBM")

    print(f"\nGenerating plots for {len(preds)} models...")
    plot_scatter(preds, y_val, out_dir)
    plot_continuous_bars(preds, y_val, out_dir)
    plot_categorical_bars(preds, y_val, out_dir)
    plot_skill_scores(preds, y_val, out_dir)
    plot_pdf(preds, y_val, out_dir)
    plot_reliability(preds, y_val, out_dir)

    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()