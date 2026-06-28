"""
tune_cnn_kan.py
===============
Optuna-based hyperparameter search for the CNN+KAN QPE model.

Imports train_epoch() and validate() directly from train_cnn_kan.py —
no modifications to that script required.

Objective
---------
Weighted combination of two metrics on the validation set:
    score = 0.5 * overall_RMSE + 0.5 * heavy_rain_RMSE (y > 25 mm/h)

Consistent with tune_lightgbm.py so results are comparable across
both stages of the pipeline.

Pruning strategy
----------------
Optuna checks after every epoch whether the trial is unpromising
relative to completed trials and prunes (kills) it early. This is
essential here because each full trial is expensive (up to 200 epochs).
A MedianPruner is used: if the current trial's score is worse than the
median of all completed trials at the same epoch, the trial is pruned.

With 20 trials and aggressive pruning, most bad trials are cut after
10-20 epochs instead of running all 200 — making 20 trials feasible.

Search space
------------
    lr           : 1e-4  – 1e-2   (float, log scale)
    huber_delta  : 0.1   – 5.0    (float, log scale)
    lambda_l1    : 1e-7  – 1e-3   (float, log scale)
    weight_decay : 1e-4  – 1e-1   (float, log scale)

Fixed across all trials
-----------------------
    epochs       = 200   (patience handles early stopping within each trial)
    batch_size   = 512
    patience     = 75
    seed         = 42

Outputs
-------
    <output_dir>/optuna_study.pkl        full Optuna study (all trials)
    <output_dir>/best_params.json        best hyperparameters found
    <output_dir>/tune_log.txt            full tuning log

Usage
-----
    python tune_cnn_kan.py

    python tune_cnn_kan.py \\
        --data_dir   ../data/dataset/processed_T4_improved \\
        --output_dir ../data/models/qpe_cnn_kan/optuna_cnn_kan \\
        --n_trials   20 \\
        --heavy_weight 0.5

Dependencies
------------
    pip install optuna
"""

import sys
from pathlib import Path

_ROOT          = Path(__file__).resolve().parent.parent
_TRAIN_DIR     = Path(__file__).resolve().parent
_AI_SCRIPT_DIR = _ROOT / "AI_Script"

for p in [str(_TRAIN_DIR), str(_AI_SCRIPT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import argparse
import json
import logging
import pickle
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import optuna
from optuna.samplers import TPESampler

# Import directly from train_cnn_kan.py — no changes to that file needed
from train_cnn_kan import (
    QPEDataset,
    QPEModel,
    RadarAugment,
    train_epoch,
    validate,
)

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(output_dir: Path) -> logging.Logger:
    log = logging.getLogger("cnn_kan_tune")
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    log.addHandler(ch)
    fh = logging.FileHandler(output_dir / "tune_log.txt", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def make_objective(
    X_train, y_train, X_val, y_val,
    chan_means, chan_stds,
    heavy_weight: float,
    epochs: int,
    batch_size: int,
    patience: int,
    seed: int,
    device: torch.device,
):
    """
    Returns an Optuna objective closed over the dataset.
    Each trial trains a fresh CNN+KAN from scratch and reports
    the best validation score achieved, with epoch-level pruning
    to kill unpromising trials early.
    """
    overall_weight = 1.0 - heavy_weight
    heavy_mask     = y_val > 25.0
    n_heavy        = int(heavy_mask.sum())

    def objective(trial: optuna.Trial) -> float:

        # ── Sample hyperparameters ─────────────────────────────────────────
        lr           = trial.suggest_float("lr",           1e-4, 1e-2, log=True)
        huber_delta  = trial.suggest_float("huber_delta",  0.1,  5.0,  log=True)
        lambda_l1    = trial.suggest_float("lambda_l1",    1e-7, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)

        torch.manual_seed(seed)
        np.random.seed(seed)

        # ── DataLoaders ────────────────────────────────────────────────────
        augment  = RadarAugment(noise_std=0.3, p_flip=0.5,
                                p_rot=0.5,     p_noise=0.5)
        train_ds = QPEDataset(X_train, y_train, chan_means, chan_stds, augment)
        val_ds   = QPEDataset(X_val,   y_val,   chan_means, chan_stds)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=(device.type == "cuda"))
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=(device.type == "cuda"))

        # ── Model, optimiser, scheduler, loss ─────────────────────────────
        model     = QPEModel().to(device)
        optimiser = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay)
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

        # ── Training loop with epoch-level pruning ─────────────────────────
        best_score     = float("inf")
        best_overall   = float("inf")
        best_heavy     = float("inf")
        patience_count = 0
        last_epoch     = 0

        for epoch in range(1, epochs + 1):
            last_epoch = epoch

            train_epoch(model, train_loader, optimiser, criterion,
                        device, lambda_l1, scheduler)

            _, metrics = validate(model, val_loader, criterion,
                                  device, lambda_l1)

            overall_rmse = metrics["RMSE"]

            # Heavy-rain RMSE: re-run val inference to get per-sample preds
            if n_heavy >= 5:
                model.eval()
                all_pred, all_true = [], []
                with torch.no_grad():
                    for Xb, yb_log in val_loader:
                        pred_log = model(Xb.to(device)).squeeze(1)
                        all_pred.append(
                            torch.clamp(torch.expm1(pred_log),
                                        min=0.0).cpu().numpy())
                        all_true.append(
                            torch.expm1(yb_log).cpu().numpy())
                pred_mmh = np.concatenate(all_pred)
                true_mmh = np.concatenate(all_true)
                heavy_rmse = float(np.sqrt(
                    np.mean((pred_mmh[heavy_mask] - true_mmh[heavy_mask]) ** 2)))
            else:
                heavy_rmse = overall_rmse

            score = overall_weight * overall_rmse + heavy_weight * heavy_rmse

            # Report intermediate score to Optuna for pruning decisions
            trial.report(score, epoch)

            # Track best within this trial (patience on score, not just RMSE,
            # consistent with the objective we are optimising)
            if score < best_score:
                best_score   = score
                best_overall = overall_rmse
                best_heavy   = heavy_rmse
                patience_count = 0
            else:
                patience_count += 1

            # Optuna pruning — kills trial if unpromising vs completed trials
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            # Internal early stopping — mirrors train_cnn_kan.py behaviour
            if patience_count >= patience:
                break

        trial.set_user_attr("best_overall_rmse", round(best_overall, 4))
        trial.set_user_attr("best_heavy_rmse",   round(best_heavy,   4))
        trial.set_user_attr("epochs_run",        last_epoch)

        return best_score

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def tune(
    data_dir:     Path,
    output_dir:   Path,
    n_trials:     int,
    heavy_weight: float,
    epochs:       int,
    batch_size:   int,
    patience:     int,
    seed:         int,
    log:          logging.Logger,
) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Load dataset ───────────────────────────────────────────────────────
    log.info("Loading dataset from %s", data_dir)
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_val   = np.load(data_dir / "X_val.npy")
    y_val   = np.load(data_dir / "y_val.npy")
    log.info("Train: %d  Val: %d", len(y_train), len(y_val))
    log.info("Heavy-rain samples in val (>25 mm/h): %d (%.1f%%)",
             (y_val > 25).sum(), 100 * (y_val > 25).mean())

    with open(data_dir / "dataset_stats.json") as f:
        stats = json.load(f)

    channel_names = ["Z_low", "ZDR_low", "Z_high", "ZDR_high"]
    chan_means = np.array([stats[n]["mean"] for n in channel_names],
                          dtype=np.float32)
    chan_stds  = np.array([stats[n]["std"]  for n in channel_names],
                          dtype=np.float32)

    # ── Log configuration ──────────────────────────────────────────────────
    log.info("=" * 65)
    log.info("Objective = %.2f * overall_RMSE + %.2f * heavy_RMSE (>25mm/h)",
             1.0 - heavy_weight, heavy_weight)
    log.info("Search space:")
    log.info("  lr           : [1e-4,  1e-2]  log scale")
    log.info("  huber_delta  : [0.1,   5.0]   log scale")
    log.info("  lambda_l1    : [1e-7,  1e-3]  log scale")
    log.info("  weight_decay : [1e-4,  1e-1]  log scale")
    log.info("Fixed: epochs=%d  batch_size=%d  patience=%d  seed=%d",
             epochs, batch_size, patience, seed)
    log.info("Pruner: MedianPruner  startup=5  warmup=20 epochs  interval=5")
    log.info("=" * 65)

    # ── Build objective ────────────────────────────────────────────────────
    objective = make_objective(
        X_train, y_train, X_val, y_val,
        chan_means, chan_stds,
        heavy_weight = heavy_weight,
        epochs       = epochs,
        batch_size   = batch_size,
        patience     = patience,
        seed         = seed,
        device       = device,
    )

    # ── Optuna study ───────────────────────────────────────────────────────
    # MedianPruner settings:
    #   n_startup_trials=5  : need 5 complete trials before pruning activates,
    #                         so early trials always run to completion and give
    #                         the pruner a meaningful baseline to compare against.
    #   n_warmup_steps=20   : don't prune before epoch 20 — the OneCycleLR
    #                         warm-up phase runs to epoch ~60, so scores before
    #                         epoch 20 are noisy and not representative.
    #   interval_steps=5    : check every 5 epochs after warmup, not every
    #                         epoch, to reduce pruning overhead.
    sampler = TPESampler(seed=seed, multivariate=True)
    pruner  = optuna.pruners.MedianPruner(
        n_startup_trials = 5,
        n_warmup_steps   = 20,
        interval_steps   = 5,
    )

    study = optuna.create_study(
        direction  = "minimize",
        sampler    = sampler,
        pruner     = pruner,
        study_name = "cnn_kan_qpe",
    )

    def trial_callback(study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            log.info(
                "Trial %3d | score=%.4f | overall_RMSE=%.4f | "
                "heavy_RMSE=%.4f | epochs=%3d | best_so_far=%.4f",
                trial.number,
                trial.value,
                trial.user_attrs.get("best_overall_rmse", -1),
                trial.user_attrs.get("best_heavy_rmse",   -1),
                trial.user_attrs.get("epochs_run",        -1),
                study.best_value,
            )
        elif trial.state == optuna.trial.TrialState.PRUNED:
            log.info("Trial %3d | PRUNED at epoch %s",
                     trial.number,
                     trial.last_step if trial.last_step is not None else "?")

    log.info("Starting Optuna search: n_trials=%d", n_trials)
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
    log.info("Best trial       : #%d", best.number)
    log.info("Best score       : %.4f", best.value)
    log.info("Best overall RMSE: %.4f",
             best.user_attrs.get("best_overall_rmse", -1))
    log.info("Best heavy RMSE  : %.4f",
             best.user_attrs.get("best_heavy_rmse", -1))
    log.info("Epochs run       : %d",
             best.user_attrs.get("epochs_run", -1))
    log.info("Best params:")
    for k, v in best.params.items():
        log.info("  %-20s : %s", k, v)

    # ── Save outputs ───────────────────────────────────────────────────────
    # best_params.json is self-contained — fixed params included so the
    # file can be read and passed directly to train_cnn_kan.py as CLI args
    best_params = {
        **best.params,
        "epochs":      epochs,
        "batch_size":  batch_size,
        "patience":    patience,
        "seed":        seed,
        "best_score":        round(best.value, 5),
        "best_overall_rmse": best.user_attrs.get("best_overall_rmse", -1),
        "best_heavy_rmse":   best.user_attrs.get("best_heavy_rmse",   -1),
        "objective_weights": {
            "overall_rmse": round(1.0 - heavy_weight, 2),
            "heavy_rmse":   round(heavy_weight,        2),
        },
    }

    params_path = output_dir / "best_params.json"
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    log.info("Best params saved : %s", params_path)

    study_path = output_dir / "optuna_study.pkl"
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    log.info("Full study saved  : %s", study_path)

    # ── Top trials summary ─────────────────────────────────────────────────
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value)
    pruned = [t for t in study.trials
              if t.state == optuna.trial.TrialState.PRUNED]

    log.info("")
    log.info("Top %d completed trials:", min(10, len(completed)))
    log.info("  %4s  %8s  %12s  %10s  %7s",
             "Rank", "Score", "Overall RMSE", "Heavy RMSE", "Epochs")
    for rank, t in enumerate(completed[:10], 1):
        log.info("  %4d  %8.4f  %12.4f  %10.4f  %7d",
                 rank, t.value,
                 t.user_attrs.get("best_overall_rmse", -1),
                 t.user_attrs.get("best_heavy_rmse",   -1),
                 t.user_attrs.get("epochs_run",        -1))

    log.info("")
    log.info("Trials complete: %d  |  Pruned: %d  |  Total: %d",
             len(completed), len(pruned), len(study.trials))

    # Print ready-to-use CLI command for retraining with best params
    log.info("")
    log.info("To retrain with best params:")
    cmd = "  python train_cnn_kan.py"
    for k, v in best.params.items():
        cmd += f" \\\n    --{k} {v}"
    log.info(cmd)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna hyperparameter search for CNN+KAN QPE model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_dir",     type=Path,
                   default=_ROOT / "data" / "dataset" / "processed_T4_improved")
    p.add_argument("--output_dir",   type=Path,
                   default=_ROOT / "models" / "qpe_cnn_kan" / "optuna_cnn_kan")
    p.add_argument("--n_trials",     type=int,   default=20)
    p.add_argument("--heavy_weight", type=float, default=0.5,
                   help="Weight for heavy-rain RMSE (>25mm/h) in objective. "
                        "Complement goes to overall RMSE.")
    # Fixed training params — exposed so they can be overridden if needed
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--batch_size",   type=int,   default=512)
    p.add_argument("--patience",     type=int,   default=75)
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(args.output_dir)
    log.info("CNN+KAN Optuna Tuning started.")
    log.info("Config: %s", vars(args))
    tune(
        data_dir     = args.data_dir,
        output_dir   = args.output_dir,
        n_trials     = args.n_trials,
        heavy_weight = args.heavy_weight,
        epochs       = args.epochs,
        batch_size   = args.batch_size,
        patience     = args.patience,
        seed         = args.seed,
        log          = log,
    )


if __name__ == "__main__":
    main()