"""
train_convlstm.py
==================
Training script for ConvLSTMQPE.

Works with the EXISTING T_{4x9x9} dataset — no new dataset needed.
Input: (N, 4, 9, 9) — same files as CNN+KAN and RQPENetD1.

Internally the model reshapes (N,4,9,9) -> (N,4,1,9,9),
treating each dual-pol channel as one ConvLSTM timestep.

Directory structure
-------------------
    QPE_CNN_KAN_LIGHTGBM/
        AI_Script/
            convlstm_qpe.py
        data/dataset/processed_T4_improved/    <- same dataset as always
        models/
            convlstm_qpe/
                best_model.pt
                training_history.json
                training_log.txt
        training_scripts/
            train_convlstm.py         <- THIS FILE

Usage
-----
    python train_convlstm.py

Dependencies
------------
    pip install torch numpy
"""

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

_ROOT          = Path(__file__).resolve().parent.parent
_AI_SCRIPT_DIR = _ROOT / "AI_Script"
if str(_AI_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_AI_SCRIPT_DIR))

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import argparse
import json
import logging
import time
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
try:
    from convlstm_qpe import ConvLSTMQPE, count_parameters
except ImportError as e:
    sys.exit(f"[FATAL] Cannot import convlstm_qpe.py from {_AI_SCRIPT_DIR}\n{e}")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(output_dir: Path) -> logging.Logger:
    log = logging.getLogger("convlstm_train")
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
# Data augmentation — same as CNN+KAN for fair comparison
# ---------------------------------------------------------------------------

class RadarAugment:
    def __init__(self, noise_std=0.3, p_flip=0.5, p_rot=0.5, p_noise=0.5):
        self.noise_std = noise_std
        self.p_flip    = p_flip
        self.p_rot     = p_rot
        self.p_noise   = p_noise

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p_rot:
            x = torch.rot90(x, k=torch.randint(1,4,(1,)).item(), dims=[1,2])
        if torch.rand(1).item() < self.p_flip:
            x = torch.flip(x, dims=[2])
        if torch.rand(1).item() < self.p_flip:
            x = torch.flip(x, dims=[1])
        if torch.rand(1).item() < self.p_noise:
            noise = torch.randn(2, x.shape[1], x.shape[2]) * self.noise_std
            x = x.clone()
            x[0] = x[0] + noise[0]
            x[2] = x[2] + noise[1]
        return x


# ---------------------------------------------------------------------------
# Dataset — identical to CNN+KAN and RQPENetD1
# ---------------------------------------------------------------------------

class QPEDataset(Dataset):
    def __init__(self,
                 X:          np.ndarray,
                 y:          np.ndarray,
                 chan_means: np.ndarray,
                 chan_stds:  np.ndarray,
                 augment=None) -> None:
        X = X.astype(np.float32).copy()
        for c in range(4):
            X[:, c] = (X[:, c] - chan_means[c]) / (chan_stds[c] + 1e-8)
        self.X       = torch.from_numpy(X)
        self.y_log1p = torch.log1p(torch.from_numpy(y.astype(np.float32)))
        self.augment = augment

    def __len__(self):
        return len(self.y_log1p)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment is not None:
            x = self.augment(x)
        return x, self.y_log1p[idx]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    thresholds: List[float] = [1.0, 5.0, 10.0, 25.0]
                    ) -> Dict[str, float]:
    residuals = y_pred - y_true
    ss_res    = np.sum(residuals ** 2)
    ss_tot    = np.sum((y_true - y_true.mean()) ** 2)
    metrics   = {
        "RMSE":        float(np.sqrt(np.mean(residuals ** 2))),
        "MAE":         float(np.mean(np.abs(residuals))),
        "Bias":        float(np.mean(residuals)),
        "Correlation": float(np.corrcoef(y_true, y_pred)[0, 1]),
        "R2":          float(1 - ss_res / (ss_tot + 1e-8)),
    }
    for thr in thresholds:
        obs_p  = y_true >= thr
        pred_p = y_pred >= thr
        tp     = float(np.sum(obs_p & pred_p))
        fp     = float(np.sum(~obs_p & pred_p))
        fn     = float(np.sum(obs_p & ~pred_p))
        denom  = tp + fp + fn
        metrics[f"CSI@{thr:.0f}"] = tp / denom if denom > 0 else 0.0
    return metrics


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimiser, criterion,
                device, scheduler=None) -> float:
    model.train()
    total = 0.0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimiser.zero_grad()
        pred = model(X_b).squeeze(1)
        loss = criterion(pred, y_b)
        loss.backward()
        # Gradient clipping — critical for LSTM stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        if scheduler is not None:
            scheduler.step()
        total += loss.item() * len(y_b)
    return total / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device) -> Tuple[float, Dict]:
    model.eval()
    total    = 0.0
    all_true = []
    all_pred = []
    for X_b, y_log1p_b in loader:
        X_b       = X_b.to(device)
        y_log1p_b = y_log1p_b.to(device)
        pred      = model(X_b).squeeze(1)
        total    += criterion(pred, y_log1p_b).item() * len(y_log1p_b)
        all_pred.append(torch.clamp(torch.expm1(pred), min=0).cpu().numpy())
        all_true.append(torch.expm1(y_log1p_b).cpu().numpy())
    val_loss = total / len(loader.dataset)
    metrics  = compute_metrics(np.concatenate(all_true),
                                np.concatenate(all_pred))
    return val_loss, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(data_dir: Path, output_dir: Path, epochs: int, batch_size: int,
          lr: float, patience: int, hidden_dim: int,
          seed: int, log: logging.Logger) -> None:

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Load existing dataset — same files as CNN+KAN ──────────────────────
    log.info("Loading dataset from %s", data_dir)
    X_train = np.load(data_dir / "X_train.npy")   # (N, 4, 9, 9)
    y_train = np.load(data_dir / "y_train.npy")
    X_val   = np.load(data_dir / "X_val.npy")
    y_val   = np.load(data_dir / "y_val.npy")

    with open(data_dir / "dataset_stats.json") as f:
        stats = json.load(f)

    channel_names = ["Z_low", "ZDR_low", "Z_high", "ZDR_high"]
    chan_means = np.array([stats[n]["mean"] for n in channel_names], np.float32)
    chan_stds  = np.array([stats[n]["std"]  for n in channel_names], np.float32)

    log.info("Train: %s  y=[%.2f, %.2f] mm/h",
             X_train.shape, y_train.min(), y_train.max())
    log.info("Val  : %s", X_val.shape)
    log.info("ConvLSTM will reshape (N,4,9,9) -> (N,4,1,9,9) internally")

    # ── Datasets — identical to CNN+KAN ────────────────────────────────────
    augment  = RadarAugment()
    train_ds = QPEDataset(X_train, y_train, chan_means, chan_stds, augment)
    val_ds   = QPEDataset(X_val,   y_val,   chan_means, chan_stds)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0,
                              pin_memory=(device.type=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0,
                              pin_memory=(device.type=="cuda"))

    # ── Model ──────────────────────────────────────────────────────────────
    model = ConvLSTMQPE(
        in_channels = 1,          # 1 channel per timestep after reshape
        hidden_dim  = hidden_dim,
        num_layers  = 2,
        kernel_size = 3,
        dropout_p   = 0.2,
    ).to(device)

    log.info("ConvLSTMQPE parameters: %d", count_parameters(model))
    log.info("hidden_dim=%d  num_layers=2  kernel=3x3  seq_len=4 (channels)",
             hidden_dim)

    # ── Optimiser — Adam as per paper ──────────────────────────────────────
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=lr, epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3, anneal_strategy="cos",
        div_factor=10.0, final_div_factor=1000.0,
    )
    criterion = nn.MSELoss()

    log.info("Optimiser: Adam  lr=%.4f  OneCycleLR", lr)
    log.info("Loss     : MSELoss on log1p(R)  (paper: MSE)")

    # ── Training loop ──────────────────────────────────────────────────────
    history = {k: [] for k in ["train_loss","val_loss","lr",
                                "RMSE","MAE","Bias","Correlation","R2",
                                "CSI@1","CSI@5","CSI@10","CSI@25"]}

    best_val_loss  = float("inf")
    best_epoch     = 0
    best_metrics: Dict = {}
    patience_count = 0

    log.info("=" * 70)
    log.info("Starting training  max_epochs=%d  patience=%d", epochs, patience)
    log.info("=" * 70)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimiser,
                                 criterion, device, scheduler)
        val_loss, metrics = validate(model, val_loader, criterion, device)
        current_lr = scheduler.get_last_lr()[0]
        elapsed    = time.time() - t0

        log.info(
            "Ep %3d/%d | train=%.5f val=%.5f | "
            "RMSE=%.3f MAE=%.3f Bias=%+.3f r=%.3f | "
            "CSI@5=%.3f CSI@10=%.3f CSI@25=%.3f | "
            "lr=%.2e | %.1fs",
            epoch, epochs, train_loss, val_loss,
            metrics["RMSE"], metrics["MAE"],
            metrics["Bias"], metrics["Correlation"],
            metrics.get("CSI@5",0), metrics.get("CSI@10",0),
            metrics.get("CSI@25",0), current_lr, elapsed,
        )

        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["lr"].append(round(current_lr, 8))
        for k in ["RMSE","MAE","Bias","Correlation","R2",
                  "CSI@1","CSI@5","CSI@10","CSI@25"]:
            history[k].append(round(metrics.get(k, 0.0), 5))

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_epoch     = epoch
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
                "model_config": {
                    "hidden_dim":  hidden_dim,
                    "num_layers":  2,
                    "kernel_size": 3,
                    "in_channels": 1,
                },
                "config": {"batch_size":batch_size,"lr":lr,"seed":seed},
            }, output_dir / "best_model.pt")
            log.info("  *** Best model saved (val_loss=%.5f) ***", val_loss)
        else:
            patience_count += 1
            if patience_count >= patience:
                log.info("Early stopping at epoch %d", epoch)
                break

    log.info("=" * 70)
    log.info("Training complete. Best epoch=%d  val_loss=%.5f",
             best_epoch, best_val_loss)
    for k, v in best_metrics.items():
        log.info("  %-20s : %.4f", k, v)

    history.update({"best_epoch": best_epoch,
                    "best_val_loss": round(best_val_loss, 6),
                    "best_metrics": {k: round(v,5) for k,v in best_metrics.items()}})
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    log.info("Saved: %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    _root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(
        description="Train ConvLSTMQPE on existing T_{4x9x9} dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_dir",   type=Path,
                   default=_root/"data"/"dataset"/"processed_T4_improved")
    p.add_argument("--output_dir", type=Path,
                   default=_root/"models"/"convlstm_qpe")
    p.add_argument("--epochs",     type=int,   default=200)
    p.add_argument("--batch_size", type=int,   default=512)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--patience",   type=int,   default=20)
    p.add_argument("--hidden_dim", type=int,   default=64)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(args.output_dir)
    log.info("ConvLSTMQPE Training started.")
    log.info("Config: %s", vars(args))
    train(data_dir=args.data_dir, output_dir=args.output_dir,
          epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, patience=args.patience,
          hidden_dim=args.hidden_dim, seed=args.seed, log=log)


if __name__ == "__main__":
    main()