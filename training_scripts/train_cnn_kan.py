"""
train_cnn_kan.py
================
Joint CNN + KAN training loop for QPE (Quantitative Precipitation Estimation).

Directory structure
-------------------
    QPE_CNN_KAN_LIGHTGBM/
        AI_Script/
            cnn_feature_extractor.py    <- RadarCNN
            kan_rainfall_predictor.py   <- RainfallKAN
        data/
            dataset/
                processed_T4_improved/           <- X_train.npy, y_train.npy, etc.
            models/
                qpe_cnn_kan/            <- best_model.pt, training_history.json
        training_scripts/
            train_cnn_kan.py            <- THIS FILE

Pipeline
--------
    (N, 4, 9, 9) --[RadarCNN]--> (N, 32) --[RainfallKAN]--> (N, 1)
    Input tensor     CNN features          log1p(R) prediction

Loss
----
    Huber loss on log1p(R) targets  (robust to heavy-tailed rain distribution)
    + L1 regularisation on KAN spline weights
    Total = HuberLoss(pred, log1p(y)) + lambda_l1 * spline_l1

Optimiser
---------
    AdamW, lr=1e-3, weight_decay=1e-3
    OneCycleLR: warm-up then cosine anneal

Training
--------
    Batch size  : 512
    Max epochs  : 200
    Early stop  : patience=75 on RMSE (unified criterion)
    Best model  : saved whenever RMSE improves

Changes from previous version
------------------------------
  1. Patience unified on RMSE only.
     Previously patience tracked RMSE but best_val_loss tracked val_loss,
     causing them to diverge — patience could tick up even when the model
     was genuinely improving on val_loss. Now both saving and patience
     use a single criterion: RMSE on the validation set.

  2. patience=75 (was 50).
     With OneCycleLR, the model continues improving during the cosine
     decay phase (epochs 60-200). patience=50 was cutting training
     short before the LR decay had time to refine the solution.

  3. batch_size=512 (was 256).
     Larger batches give more stable gradient estimates per step,
     which helps OneCycleLR's momentum cycling work correctly.
     With 13K+ samples and batch=512, each epoch has ~27 steps —
     enough for stable updates without excessive noise.

Metrics (original mm/h scale)
------------------------------
    RMSE, MAE, Bias, Pearson r
    CSI at 1, 5, 10, 25 mm/h

Outputs
-------
    <output_dir>/best_model.pt           model weights + channel stats + config
    <output_dir>/training_history.json   per-epoch loss and metric curves
    <output_dir>/training_log.txt        full training log

Usage
-----
    python train_cnn_kan.py

    python train_cnn_kan.py \\
        --data_dir   ../data/dataset/processed_T4_improved \\
        --output_dir ../data/models/qpe_cnn_kan \\
        --epochs     200 \\
        --batch_size 512 \\
        --lr         1e-3 \\
        --patience   75

Dependencies
------------
    pip install torch numpy
"""

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

_AI_SCRIPT_DIR = Path(__file__).resolve().parent.parent / "AI_Script"
if str(_AI_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_AI_SCRIPT_DIR))

import argparse
import json
import logging
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from cnn_feature_extractor import RadarCNN
    from kan_rainfall_predictor import (
        RainfallKAN,
        transform_target,
        inverse_transform_target,
    )
except ImportError as e:
    sys.exit(
        f"[FATAL] Could not import model files from {_AI_SCRIPT_DIR}\n"
        f"Error: {e}\n"
        f"Make sure cnn_feature_extractor.py and kan_rainfall_predictor.py "
        f"exist in AI_Script/"
    )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(output_dir: Path) -> logging.Logger:
    log = logging.getLogger("qpe_train")
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    log.addHandler(ch)
    fh = logging.FileHandler(output_dir / "training_log.txt", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class QPEDataset(Dataset):
    """
    T_{4x9x9} radar tensor dataset.
    Normalises each channel using per-channel mean/std (train stats only).
    Applies log1p transform to rain rate targets.
    """

    def __init__(self,
                 X:          np.ndarray,
                 y:          np.ndarray,
                 chan_means: np.ndarray,
                 chan_stds:  np.ndarray,
                 augment=None) -> None:
        X = X.astype(np.float32).copy()
        for c in range(4):
            X[:, c, :, :] = (
                (X[:, c, :, :] - chan_means[c]) / (chan_stds[c] + 1e-8)
            )
        self.X       = torch.from_numpy(X)
        self.y       = torch.from_numpy(y.astype(np.float32))
        self.y_log1p = transform_target(self.y)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        if self.augment is not None:
            x = self.augment(x)
        return x, self.y_log1p[idx]


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------

class RadarAugment:
    """
    Spatial augmentation for T_{4x9x9} radar tensors.
    Rain has no preferred orientation — rotations and flips are physically valid.
    Gaussian noise on Z channels simulates radar calibration uncertainty.
    Applied to training set only.
    """

    def __init__(self,
                 noise_std: float = 0.3,
                 p_flip:    float = 0.5,
                 p_rot:     float = 0.5,
                 p_noise:   float = 0.5) -> None:
        self.noise_std = noise_std
        self.p_flip    = p_flip
        self.p_rot     = p_rot
        self.p_noise   = p_noise

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p_rot:
            k = torch.randint(1, 4, (1,)).item()
            x = torch.rot90(x, k=k, dims=[1, 2])
        if torch.rand(1).item() < self.p_flip:
            x = torch.flip(x, dims=[2])
        if torch.rand(1).item() < self.p_flip:
            x = torch.flip(x, dims=[1])
        if torch.rand(1).item() < self.p_noise:
            noise = torch.randn(2, x.shape[1], x.shape[2]) * self.noise_std
            x = x.clone()
            x[0] = x[0] + noise[0]   # Z_low  (ch0)
            x[2] = x[2] + noise[1]   # Z_high (ch2)
        return x


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class QPEModel(nn.Module):
    """RadarCNN + RainfallKAN. Forward: (N,4,9,9) -> (N,1) log1p(R)."""

    def __init__(self) -> None:
        super().__init__()
        self.cnn = RadarCNN(in_channels=4, feature_dim=32)
        self.kan = RainfallKAN(
            input_dim=32, hidden1=64, hidden2=32,
            grid_size=5, spline_order=3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kan(self.cnn(x))

    def predict_mmh(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.clamp(
                torch.expm1(self.forward(x)).squeeze(1), min=0.0)

    def regularisation_loss(self, lambda_l1: float = 1e-5) -> torch.Tensor:
        return self.kan.regularisation_loss(lambda_l1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    thresholds: List[float] = [1.0, 5.0, 10.0, 25.0]
                    ) -> Dict[str, float]:
    residuals = y_pred - y_true
    metrics = {
        "RMSE":        float(np.sqrt(np.mean(residuals ** 2))),
        "MAE":         float(np.mean(np.abs(residuals))),
        "Bias":        float(np.mean(residuals)),
        "Correlation": float(np.corrcoef(y_true, y_pred)[0, 1]),
    }
    for thr in thresholds:
        obs_pos  = y_true >= thr
        pred_pos = y_pred >= thr
        tp    = float(np.sum( obs_pos &  pred_pos))
        fp    = float(np.sum(~obs_pos &  pred_pos))
        fn    = float(np.sum( obs_pos & ~pred_pos))
        denom = tp + fp + fn
        metrics[f"CSI@{thr:.0f}"] = tp / denom if denom > 0 else 0.0
    return metrics


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimiser, criterion, device,
                lambda_l1, scheduler=None) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimiser.zero_grad()
        pred  = model(X_batch).squeeze(1)
        loss  = criterion(pred, y_batch) + model.regularisation_loss(lambda_l1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.cnn.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model.kan.parameters(), max_norm=0.5)
        optimiser.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device, lambda_l1) -> Tuple[float, Dict]:
    model.eval()
    total_loss = 0.0
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    for X_batch, y_log1p_batch in loader:
        X_batch       = X_batch.to(device)
        y_log1p_batch = y_log1p_batch.to(device)
        pred_log1p    = model(X_batch).squeeze(1)
        loss          = criterion(pred_log1p, y_log1p_batch) + \
                        model.regularisation_loss(lambda_l1)
        total_loss   += loss.item() * len(y_log1p_batch)
        all_pred.append(
            torch.clamp(torch.expm1(pred_log1p), min=0.0).cpu().numpy())
        all_true.append(torch.expm1(y_log1p_batch).cpu().numpy())
    val_loss = total_loss / len(loader.dataset)
    metrics  = compute_metrics(
        np.concatenate(all_true), np.concatenate(all_pred))
    return val_loss, metrics


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(data_dir: Path, output_dir: Path, epochs: int, batch_size: int,
          lr: float, patience: int, lambda_l1: float, huber_delta: float,
          seed: int, log: logging.Logger) -> None:

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Load dataset ───────────────────────────────────────────────────────
    log.info("Loading dataset from: %s", data_dir)
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_val   = np.load(data_dir / "X_val.npy")
    y_val   = np.load(data_dir / "y_val.npy")

    log.info("Train : X=%s  y=%s  y_range=[%.2f, %.2f] mm/h",
             X_train.shape, y_train.shape, y_train.min(), y_train.max())
    log.info("Val   : X=%s  y=%s  y_range=[%.2f, %.2f] mm/h",
             X_val.shape, y_val.shape, y_val.min(), y_val.max())

    with open(data_dir / "dataset_stats.json") as f:
        stats = json.load(f)

    channel_names = ["Z_low", "ZDR_low", "Z_high", "ZDR_high"]
    chan_means = np.array([stats[n]["mean"] for n in channel_names],
                          dtype=np.float32)
    chan_stds  = np.array([stats[n]["std"]  for n in channel_names],
                          dtype=np.float32)

    log.info("Channel normalisation (train stats):")
    for i, n in enumerate(channel_names):
        log.info("  %-10s  mean=%8.4f  std=%8.4f", n, chan_means[i], chan_stds[i])

    # ── DataLoaders ────────────────────────────────────────────────────────
    augment  = RadarAugment(noise_std=0.3, p_flip=0.5, p_rot=0.5, p_noise=0.5)
    train_ds = QPEDataset(X_train, y_train, chan_means, chan_stds, augment)
    val_ds   = QPEDataset(X_val,   y_val,   chan_means, chan_stds)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"))

    log.info("Train batches: %d  |  Val batches: %d  |  Batch size: %d",
             len(train_loader), len(val_loader), batch_size)

    # ── Model ──────────────────────────────────────────────────────────────
    model = QPEModel().to(device)
    n_cnn   = sum(p.numel() for p in model.cnn.parameters() if p.requires_grad)
    n_kan   = sum(p.numel() for p in model.kan.parameters() if p.requires_grad)
    log.info("Parameters:  CNN=%d  KAN=%d  Total=%d", n_cnn, n_kan, n_cnn+n_kan)

    # ── Optimiser and scheduler ────────────────────────────────────────────
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr           = lr,
        epochs           = epochs,
        steps_per_epoch  = len(train_loader),
        pct_start        = 0.3,
        anneal_strategy  = "cos",
        div_factor       = 10.0,
        final_div_factor = 1000.0,
    )
    criterion = nn.HuberLoss(delta=huber_delta)

    log.info("Optimiser : AdamW  lr=%.4f  weight_decay=1e-3", lr)
    log.info("Scheduler : OneCycleLR  max_lr=%.4f  pct_start=0.3", lr)
    log.info("Augment   : rot90+flip+noise  noise_std=0.3")
    log.info("Loss      : HuberLoss(delta=%.1f) + L1_reg(lambda=%.2e)",
             huber_delta, lambda_l1)
    log.info("Patience  : %d epochs on RMSE (unified criterion)", patience)

    # ── Training loop ──────────────────────────────────────────────────────
    history: Dict[str, list] = {
        "train_loss": [], "val_loss": [], "lr": [],
        "RMSE": [], "MAE": [], "Bias": [], "Correlation": [],
        "CSI@1": [], "CSI@5": [], "CSI@10": [], "CSI@25": [],
    }

    best_rmse      = float("inf")
    best_epoch     = 0
    best_val_loss  = float("inf")
    best_metrics:  Dict[str, float] = {}
    patience_count = 0

    log.info("=" * 72)
    log.info("Starting training  max_epochs=%d  patience=%d", epochs, patience)
    log.info("=" * 72)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(
            model, train_loader, optimiser, criterion,
            device, lambda_l1, scheduler)
        val_loss, metrics = validate(
            model, val_loader, criterion, device, lambda_l1)

        current_lr   = scheduler.get_last_lr()[0]
        current_rmse = metrics["RMSE"]
        elapsed      = time.time() - t0

        log.info(
            "Ep %3d/%d | train=%.5f val=%.5f | "
            "RMSE=%.3f MAE=%.3f Bias=%+.3f r=%.3f | "
            "CSI@1=%.3f CSI@5=%.3f CSI@10=%.3f | "
            "lr=%.2e | %.1fs",
            epoch, epochs, train_loss, val_loss,
            metrics["RMSE"], metrics["MAE"], metrics["Bias"],
            metrics["Correlation"],
            metrics.get("CSI@1", 0), metrics.get("CSI@5", 0),
            metrics.get("CSI@10", 0), current_lr, elapsed)

        # Record history
        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["lr"].append(round(current_lr, 8))
        for k in ["RMSE", "MAE", "Bias", "Correlation",
                  "CSI@1", "CSI@5", "CSI@10", "CSI@25"]:
            history[k].append(round(metrics.get(k, 0.0), 5))

        # ── Save best model and track patience — unified on RMSE ──────────
        # Both saving and early stopping use the same criterion (RMSE).
        # This ensures patience only ticks up when the model is genuinely
        # not improving, not due to val_loss / RMSE divergence.
        if current_rmse < best_rmse:
            best_rmse      = current_rmse
            best_epoch     = epoch
            best_val_loss  = val_loss
            best_metrics   = metrics.copy()
            patience_count = 0

            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimiser_state": optimiser.state_dict(),
                "val_loss":        val_loss,
                "metrics":         metrics,
                "chan_means":      chan_means.tolist(),
                "chan_stds":       chan_stds.tolist(),
                "channel_names":   channel_names,
                "config": {
                    "batch_size":  batch_size,
                    "lr":          lr,
                    "lambda_l1":   lambda_l1,
                    "huber_delta": huber_delta,
                    "seed":        seed,
                },
            }, output_dir / "best_model.pt")
            log.info("  *** Best model saved  RMSE=%.4f (val_loss=%.5f) ***",
                     best_rmse, val_loss)
        else:
            patience_count += 1
            if patience_count >= patience:
                log.info(
                    "Early stopping at epoch %d "
                    "(no RMSE improvement for %d epochs)",
                    epoch, patience)
                break

    # ── Final summary ──────────────────────────────────────────────────────
    log.info("=" * 72)
    log.info("Training complete.")
    log.info("Best epoch : %d  |  Best RMSE : %.4f  |  Best val_loss : %.5f",
             best_epoch, best_rmse, best_val_loss)
    log.info("Best metrics:")
    for k, v in best_metrics.items():
        log.info("  %-20s : %.4f", k, v)

    history["best_epoch"]    = best_epoch
    history["best_val_loss"] = round(best_val_loss, 6)
    history["best_rmse"]     = round(best_rmse, 5)
    history["best_metrics"]  = {k: round(v, 5) for k, v in best_metrics.items()}

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    log.info("History saved  : %s", output_dir / "training_history.json")
    log.info("Best model     : %s", output_dir / "best_model.pt")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    _root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(
        description="Train CNN+KAN QPE model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_dir",    type=Path,
                   default=_root / "data" / "dataset" / "processed_T4_improved")
    p.add_argument("--output_dir",  type=Path,
                   default=_root / "models" / "qpe_cnn_kan")
    p.add_argument("--epochs",      type=int,   default=200)
    p.add_argument("--batch_size",  type=int,   default=512)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--patience",    type=int,   default=75)
    p.add_argument("--lambda_l1",   type=float, default=1e-5)
    p.add_argument("--huber_delta", type=float, default=1.0)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(args.output_dir)
    log.info("QPE CNN+KAN Training started.")
    log.info("AI_Script dir : %s", _AI_SCRIPT_DIR)
    log.info("Config        : %s", vars(args))
    train(
        data_dir    = args.data_dir,
        output_dir  = args.output_dir,
        epochs      = args.epochs,
        batch_size  = args.batch_size,
        lr          = args.lr,
        patience    = args.patience,
        lambda_l1   = args.lambda_l1,
        huber_delta = args.huber_delta,
        seed        = args.seed,
        log         = log,
    )


if __name__ == "__main__":
    main()