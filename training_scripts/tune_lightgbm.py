"""
tune_lightgbm.py
================
Optuna-based hyperparameter search for the LightGBM residual corrector.
v5: aligned to log1p residual space and two-stage heavy-rain weighting.

CRITICAL BUG FIXED vs v4
--------------------------
v4's make_objective trained on mm/h residuals but apply_lgb_correction
expects log1p residuals. This meant:
  - The tuner and trainer were using DIFFERENT target spaces
  - The tuner applied correction as `val_pred_mmh + resid_pred` (mm/h addition)
    but the trainer uses `expm1(log1p(R_hat) + epsilon)` (log1p inverse)
  - Hyperparameters found by v4 were optimal for the wrong target space
    and produced a corrector that did almost nothing

v5 fixes:
  1. Tuner trains on log1p residuals (same as trainer)
  2. Tuner applies correction via apply_lgb_correction() (same as trainer)
  3. Heavy-rain boost params (heavy_boost_25/50/75) added to search space
  4. Objective weighted toward heavy-rain RMSE (>25, >50 mm/h separately)
  5. min_child_samples floor lowered to 5 to allow more splits on rare samples

Objective (v5)
--------------
Balanced weighted combination targeting the reliability diagram gap:
    score = 0.25 * overall_RMSE
          + 0.15 * overall_MAE
          + 0.35 * heavy_RMSE_25  (y > 25 mm/h)
          + 0.25 * heavy_RMSE_50  (y > 50 mm/h)

Separating >25 and >50 tiers prevents the optimizer from ignoring the
very extreme tail when >25 samples already have a good RMSE.

Search space (v5)
-----------------
New vs v4:
    heavy_boost_25 : [1.0, 10.0]   (additive weight at >25 mm/h)
    heavy_boost_50 : [3.0, 20.0]   (additive weight at >50 mm/h)
    heavy_boost_75 : [5.0, 40.0]   (additive weight at >75 mm/h)
    min_child_samples : [5, 30]     (lower floor — more tail splits)
    num_leaves     : [31, 255]      (wider — more complex corrections)
    max_depth      : [4, 10]        (deeper — capture non-linear tail)

Unchanged vs v4:
    lr             : [1e-3, 1e-2]
    weight_scale   : [5.0, 20.0]
    huber_delta    : [0.5, 10.0]
    reg_alpha      : [0.01, 5.0]
    reg_lambda     : [0.1, 20.0]
    subsample      : [0.6, 1.0]
    colsample      : [0.6, 1.0]

Fixed:
    n_estimators = 3000
    early_stop   = 200
    use_huber    = True   (Huber searched; MAE is also viable)
    use_dart     = False

Outputs
-------
    <output_dir>/optuna_study_v5.pkl     full Optuna study
    <output_dir>/best_params.json        best hyperparameters
    <output_dir>/tune_log.txt            full log

Usage
-----
    python tune_lightgbm.py

    python tune_lightgbm.py \\
        --data_dir   ../data/dataset/processed_T4_improved \\
        --model_dir  ../data/models/qpe_cnn_kan \\
        --output_dir ../data/models/qpe_cnn_kan/optuna \\
        --n_trials   60
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
import optuna
from optuna.samplers import TPESampler

from train_lightgbm_corrector import (
    load_cnn_kan,
    normalise,
    get_kan_predictions,
    build_features,
    compute_sample_weights,
    compute_metrics,
    apply_lgb_correction,   # v5: import the SINGLE SOURCE OF TRUTH
    FEATURE_NAMES,
    MONOTONE_CONSTRAINTS,
)
import lightgbm as lgb

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(output_dir: Path) -> logging.Logger:
    log = logging.getLogger("lgb_tune")
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
    X_train_raw, y_train, X_val_raw, y_val,
    X_train_norm, X_val_norm,
    train_pred_mmh, train_pred_log1p,
    val_pred_mmh,   val_pred_log1p,
    rmse_weight:       float,
    mae_weight:        float,
    heavy_weight_25:   float,
    heavy_weight_50:   float,
    heavy_weight_75:   float,
    mid_bias_weight:   float,
    n_estimators:      int,
    early_stop:        int,
    log:               logging.Logger,
):
    # ── Build feature matrices once — shared across all trials ─────────────
    X_lgb_train = build_features(X_train_raw, X_train_norm,
                                  train_pred_mmh, train_pred_log1p)
    X_lgb_val   = build_features(X_val_raw,   X_val_norm,
                                  val_pred_mmh,   val_pred_log1p)

    # ── LOG1P-SPACE RESIDUALS — must match train_lightgbm_corrector.py ─────
    # v4 BUG: used mm/h residuals here, making tuner/trainer misaligned.
    # v5 FIX: compute log1p residuals exactly as the trainer does.
    y_train_log1p = np.log1p(np.clip(y_train, 0.0, None)).astype(np.float32)
    y_val_log1p   = np.log1p(np.clip(y_val,   0.0, None)).astype(np.float32)

    train_pred_log1p_pos = np.log1p(np.clip(train_pred_mmh, 0.0, None)).astype(np.float32)
    val_pred_log1p_pos   = np.log1p(np.clip(val_pred_mmh,   0.0, None)).astype(np.float32)

    train_resid_log = (y_train_log1p - train_pred_log1p_pos).astype(np.float32)
    val_resid_log   = (y_val_log1p   - val_pred_log1p_pos  ).astype(np.float32)

    heavy_mask_25  = y_val > 25.0
    heavy_mask_40  = y_val > 40.0
    heavy_mask_50  = y_val > 50.0
    heavy_mask_75  = y_val > 75.0
    mid_mask       = (y_val >= 10.0) & (y_val <= 50.0)  # overcorrection zone
    n_heavy_25     = heavy_mask_25.sum()
    n_heavy_40     = heavy_mask_40.sum()
    n_heavy_50     = heavy_mask_50.sum()
    n_heavy_75     = heavy_mask_75.sum()
    n_mid          = mid_mask.sum()

    def objective(trial: optuna.Trial) -> float:

        # ── Sample hyperparameters (v9 — smooth exponential weight curve) ────
        # 3 params fully describe the weight shape across all rain rates.
        # w(y) = w_min + (w_max-w_min) * expm1(gamma*y) / expm1(gamma*150)
        # No discrete steps, no conflicting penalties — every mm/h weighted smoothly.
        w_min             = trial.suggest_float("w_min",              0.5,   2.0)
        w_max             = trial.suggest_float("w_max",             25.0,  80.0)  # tighter range
        gamma             = trial.suggest_float("gamma",              0.02,  0.12)  # tighter range
        use_huber         = trial.suggest_categorical("use_huber", [True, False])
        huber_delta       = trial.suggest_float("huber_delta",        0.3,   2.0, log=True)
        lr                = trial.suggest_float("lr",                5e-4,  5e-3, log=True)
        num_leaves        = trial.suggest_int(  "num_leaves",         48,   128)
        max_depth         = trial.suggest_int(  "max_depth",           5,     9)
        reg_alpha         = trial.suggest_float("reg_alpha",          0.05,  1.0, log=True)
        reg_lambda        = trial.suggest_float("reg_lambda",         5.0,  30.0, log=True)
        min_child_samples = trial.suggest_int(  "min_child_samples",   5,    20)
        subsample         = trial.suggest_float("subsample",           0.6,   0.9)
        colsample         = trial.suggest_float("colsample",           0.7,   1.0)

        weights = compute_sample_weights(y_train, w_min, w_max, gamma)

        objective_fn = "huber" if use_huber else "regression_l2"
        # LightGBM reports MAE as "l1" in its eval results, NOT "mae".
        # The Optuna pruning callback must match the internal name exactly.
        metric_name      = "huber" if use_huber else "l2"
        metric_name_lgb  = "huber" if use_huber else "mse"   # param key for regression_l2
        alpha_val        = huber_delta if use_huber else 0.9

        lgb_train  = lgb.Dataset(X_lgb_train, label=train_resid_log,
                                  weight=weights, feature_name=FEATURE_NAMES,
                                  free_raw_data=False)
        lgb_val_ds = lgb.Dataset(X_lgb_val,   label=val_resid_log,
                                  reference=lgb_train,
                                  feature_name=FEATURE_NAMES,
                                  free_raw_data=False)

        params = {
            "boosting_type":        "gbdt",
            "objective":            objective_fn,
            "alpha":                alpha_val,
            "metric":               metric_name_lgb,  # "mae" or "huber" for param
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

        pruning_callback = optuna.integration.LightGBMPruningCallback(
            trial, metric_name, valid_name="valid_0")

        try:
            lgb_model = lgb.train(
                params, lgb_train,
                num_boost_round=n_estimators,
                valid_sets=[lgb_val_ds],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=early_stop, verbose=False),
                    lgb.log_evaluation(period=-1),
                    pruning_callback,
                ],
            )
        except optuna.exceptions.TrialPruned:
            raise

        num_iter = (lgb_model.best_iteration
                    if lgb_model.best_iteration > 0
                    else lgb_model.num_trees())

        # ── Apply correction via SINGLE SOURCE OF TRUTH ────────────────────
        # v4 BUG: used `val_pred_mmh + resid_pred` (mm/h addition) — WRONG.
        # v5 FIX: use apply_lgb_correction() = expm1(log1p(R_hat) + epsilon).
        epsilon_log = lgb_model.predict(X_lgb_val, num_iteration=num_iter)
        final_pred  = apply_lgb_correction(val_pred_mmh, epsilon_log)

        overall_rmse = float(np.sqrt(np.mean((final_pred - y_val) ** 2)))
        overall_mae  = float(np.mean(np.abs(final_pred - y_val)))

        if n_heavy_25 >= 5:
            heavy_rmse_25 = float(np.sqrt(
                np.mean((final_pred[heavy_mask_25] - y_val[heavy_mask_25]) ** 2)))
        else:
            heavy_rmse_25 = overall_rmse

        if n_heavy_40 >= 5:
            heavy_rmse_40 = float(np.sqrt(
                np.mean((final_pred[heavy_mask_40] - y_val[heavy_mask_40]) ** 2)))
        else:
            heavy_rmse_40 = heavy_rmse_25

        if n_heavy_50 >= 5:
            heavy_rmse_50 = float(np.sqrt(
                np.mean((final_pred[heavy_mask_50] - y_val[heavy_mask_50]) ** 2)))
        else:
            heavy_rmse_50 = heavy_rmse_40

        if n_heavy_75 >= 5:
            heavy_rmse_75 = float(np.sqrt(
                np.mean((final_pred[heavy_mask_75] - y_val[heavy_mask_75]) ** 2)))
        else:
            heavy_rmse_75 = heavy_rmse_50

        # Bias penalties — both directions:
        #   mid_overfit : penalise OVERcorrection in 10-50 mm/h (bias > 0)
        #   tail_under  : penalise UNDERcorrection in >25 mm/h (bias < 0)
        # Together they squeeze the curve toward the 1:1 line from both sides.
        if n_mid >= 5:
            mid_bias_val = float(np.mean(final_pred[mid_mask] - y_val[mid_mask]))
            mid_overfit_penalty = max(0.0,  mid_bias_val)   # fire only if overcorrecting
        else:
            mid_overfit_penalty = 0.0

        if n_heavy_25 >= 5:
            tail_bias_val = float(np.mean(final_pred[heavy_mask_25] - y_val[heavy_mask_25]))
            tail_under_penalty = max(0.0, -tail_bias_val)   # fire only if undercorrecting
        else:
            tail_under_penalty = 0.0

        # v10 objective: four RMSE tiers + two-sided bias squeeze
        score = (rmse_weight     * overall_rmse       +
                 mae_weight      * overall_mae        +
                 heavy_weight_25 * heavy_rmse_25      +
                 heavy_weight_25 * heavy_rmse_40      +
                 heavy_weight_50 * heavy_rmse_50      +
                 heavy_weight_75 * heavy_rmse_75      +
                 mid_bias_weight * mid_overfit_penalty +
                 mid_bias_weight * 0.3 * tail_under_penalty)  # mild nudge only

        trial.set_user_attr("overall_rmse",       round(overall_rmse,       4))
        trial.set_user_attr("overall_mae",        round(overall_mae,        4))
        trial.set_user_attr("heavy_rmse_25",      round(heavy_rmse_25,      4))
        trial.set_user_attr("heavy_rmse_40",      round(heavy_rmse_40,      4))
        trial.set_user_attr("heavy_rmse_50",      round(heavy_rmse_50,      4))
        trial.set_user_attr("heavy_rmse_75",      round(heavy_rmse_75,      4))
        trial.set_user_attr("mid_bias",           round(mid_overfit_penalty,4))
        trial.set_user_attr("tail_under",         round(tail_under_penalty, 4))
        trial.set_user_attr("n_trees",            num_iter)

        return score

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def tune(
    data_dir:         Path,
    model_dir:        Path,
    output_dir:       Path,
    n_trials:         int,
    rmse_weight:      float,
    mae_weight:       float,
    heavy_weight_25:  float,
    heavy_weight_50:  float,
    heavy_weight_75:  float,
    mid_bias_weight:  float,
    n_estimators:     int,
    early_stop:       int,
    log:              logging.Logger,
) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    log.info("Loading dataset from %s", data_dir)
    X_train_raw = np.load(data_dir / "X_train.npy")
    y_train     = np.load(data_dir / "y_train.npy")
    X_val_raw   = np.load(data_dir / "X_val.npy")
    y_val       = np.load(data_dir / "y_val.npy")
    log.info("Train: %d  Val: %d", len(y_train), len(y_val))
    log.info("Heavy-rain samples in val:")
    for thr in [25, 50, 75]:
        mask = y_val > thr
        log.info("  >%d mm/h : %d (%.1f%%)", thr, mask.sum(), 100 * mask.mean())

    log.info("Loading CNN+KAN from %s", model_dir / "best_model.pt")
    model, chan_means, chan_stds = load_cnn_kan(
        model_dir / "best_model.pt", device, log)

    X_train_norm = normalise(X_train_raw, chan_means, chan_stds)
    X_val_norm   = normalise(X_val_raw,   chan_means, chan_stds)

    log.info("Running CNN+KAN inference (once, shared across all trials)...")
    train_pred_mmh, train_pred_log1p = get_kan_predictions(
        model, X_train_norm, device)
    val_pred_mmh, val_pred_log1p = get_kan_predictions(
        model, X_val_norm, device)

    kan_metrics = compute_metrics(y_val, val_pred_mmh)
    log.info("CNN+KAN baseline — RMSE=%.4f  MAE=%.4f  r=%.4f",
             kan_metrics["RMSE"], kan_metrics["MAE"], kan_metrics["Correlation"])
    for thr in [25, 50]:
        mask = y_val > thr
        if mask.sum() >= 5:
            rmse_thr = float(np.sqrt(np.mean((val_pred_mmh[mask] - y_val[mask]) ** 2)))
            bias_thr = float(np.mean(val_pred_mmh[mask] - y_val[mask]))
            log.info("CNN+KAN >%d mm/h : RMSE=%.4f  Bias=%+.4f  N=%d",
                     thr, rmse_thr, bias_thr, mask.sum())

    objective = make_objective(
        X_train_raw, y_train, X_val_raw, y_val,
        X_train_norm, X_val_norm,
        train_pred_mmh, train_pred_log1p,
        val_pred_mmh,   val_pred_log1p,
        rmse_weight      = rmse_weight,
        mae_weight       = mae_weight,
        heavy_weight_25  = heavy_weight_25,
        heavy_weight_50  = heavy_weight_50,
        heavy_weight_75  = heavy_weight_75,
        mid_bias_weight  = mid_bias_weight,
        n_estimators     = n_estimators,
        early_stop       = early_stop,
        log              = log,
    )

    log.info("=" * 65)
    log.info("Starting v10 search: n_trials=%d", n_trials)
    log.info("Objective = %.2f*RMSE + %.2f*MAE + %.2f*RMSE(>25+>40) + %.2f*RMSE(>50) + %.2f*RMSE(>75) + %.2f*MidBias",
             rmse_weight, mae_weight, heavy_weight_25, heavy_weight_50, heavy_weight_75, mid_bias_weight)
    log.info("v10 changes vs v9:")
    log.info("  [NEW] Two-sided bias squeeze: penalise both over AND undercorrection")
    log.info("  [NEW] tail_under_penalty: fires when >25mm/h mean bias is negative")
    log.info("  [WIDER] w_max ceiling 100->200, gamma ceiling 0.15->0.25")
    log.info("  [WARM] Warm-start with w_max=60, gamma=0.08 (steeper than before)")
    log.info("=" * 65)

    sampler = TPESampler(seed=42, multivariate=True)
    pruner  = optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=50)

    study = optuna.create_study(
        direction  = "minimize",
        sampler    = sampler,
        pruner     = pruner,
        study_name = "lgb_qpe_corrector_v10",
    )

    # Warm-start: w_min=1, w_max=30, gamma=0.05 approximates old step-boost profile
    study.enqueue_trial({
        "w_min":              1.0,
        "w_max":             35.0,   # close to v9 best, slight push
        "gamma":              0.06,   # slight steepening from v9
        "use_huber":         True,
        "huber_delta":        0.6377063394270273,
        "lr":                 0.0037215775240132557,
        "num_leaves":        63,
        "max_depth":          6,
        "reg_alpha":          0.12421030517460693,
        "reg_lambda":        18.824069212509663,
        "min_child_samples": 20,
        "subsample":          0.6874147628356203,
        "colsample":          0.8780843210582354,
    })

    def trial_callback(study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            log.info(
                "Trial %3d | score=%.4f | RMSE=%.4f | MAE=%.4f | "
                "h25=%.4f | h50=%.4f | h75=%.4f | mbias=%+.2f | trees=%4d | best=%.4f",
                trial.number, trial.value,
                trial.user_attrs.get("overall_rmse",  -1),
                trial.user_attrs.get("overall_mae",   -1),
                trial.user_attrs.get("heavy_rmse_25", -1),
                trial.user_attrs.get("heavy_rmse_50", -1),
                trial.user_attrs.get("heavy_rmse_75", -1),
                trial.user_attrs.get("mid_bias",      0.0),
                trial.user_attrs.get("n_trees",       -1),
                study.best_value,
            )
        elif trial.state == optuna.trial.TrialState.PRUNED:
            log.info("Trial %3d | PRUNED", trial.number)

    study.optimize(
        objective,
        n_trials          = n_trials,
        callbacks         = [trial_callback],
        show_progress_bar = False,
    )

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
    log.info("Best mid bias      : %+.4f",best.user_attrs.get("mid_bias",      0.0))
    log.info("Best n_trees       : %d",   best.user_attrs.get("n_trees",       -1))
    log.info("Best params:")
    for k, v in best.params.items():
        log.info("  %-22s : %s", k, v)

    # ── Separate fixed vs tuned params for the retrain command ─────────────
    tuned_params = {k: v for k, v in best.params.items()
                    if k not in ("use_huber", "huber_delta")}

    best_params = {
        **best.params,
        "n_estimators": n_estimators,
        "early_stop":   early_stop,
        "use_dart":     False,
        "best_score":          round(best.value,                                  5),
        "best_overall_rmse":   best.user_attrs.get("overall_rmse",  -1),
        "best_overall_mae":    best.user_attrs.get("overall_mae",   -1),
        "best_heavy_rmse_25":  best.user_attrs.get("heavy_rmse_25", -1),
        "best_heavy_rmse_50":  best.user_attrs.get("heavy_rmse_50", -1),
        "best_heavy_rmse_75":  best.user_attrs.get("heavy_rmse_75", -1),
        "best_mid_bias":       best.user_attrs.get("mid_bias",      0.0),
        "kan_baseline_rmse":   round(kan_metrics["RMSE"], 5),
        "kan_baseline_mae":    round(kan_metrics["MAE"],  5),
        "objective_weights": {
            "overall_rmse":  rmse_weight,
            "overall_mae":   mae_weight,
            "heavy_rmse_25": heavy_weight_25,
            "heavy_rmse_50": heavy_weight_50,
            "heavy_rmse_75": heavy_weight_75,
            "mid_bias":      mid_bias_weight,
            "weight_fn": "smooth_exp: w_min+(w_max-w_min)*expm1(gamma*y)/expm1(gamma*150)",
        },
        "constraint_version": "v4_interactions_unconstrained",
        "residual_space":     "log1p",
        "search_pass":        10,
    }

    params_path = output_dir / "best_params.json"
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    log.info("Best params saved : %s", params_path)

    study_path = output_dir / "optuna_study_v10.pkl"
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    log.info("Full study saved  : %s", study_path)

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value)
    pruned = [t for t in study.trials
              if t.state == optuna.trial.TrialState.PRUNED]

    log.info("")
    log.info("Top 10 trials by score:")
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

    log.info("")
    log.info("To retrain LightGBM with best params:")
    cmd = "  python train_lightgbm_corrector.py"
    for k, v in best.params.items():
        if isinstance(v, bool):
            if v:
                cmd += f" \\\n    --{k}"
            else:
                cmd += f" \\\n    --no-{k}"
        else:
            cmd += f" \\\n    --{k} {v}"
    cmd += f" \\\n    --n_estimators {n_estimators}"
    cmd += f" \\\n    --early_stop {early_stop}"
    log.info(cmd)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna v10 search for LightGBM QPE corrector (two-sided bias squeeze).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_dir",       type=Path,
                   default=_ROOT / "data" / "dataset" / "processed_T4_improved")
    p.add_argument("--model_dir",      type=Path,
                   default=_ROOT / "models" / "qpe_cnn_kan")
    p.add_argument("--output_dir",     type=Path,
                   default=_ROOT / "models" / "qpe_cnn_kan" / "optuna")
    p.add_argument("--n_trials",       type=int,   default=60)

    # Objective weights — v5 adds separate >25 and >50 mm/h tiers
    p.add_argument("--rmse_weight",     type=float, default=0.25,
                   help="Weight for overall RMSE in objective.")
    p.add_argument("--mae_weight",      type=float, default=0.15,
                   help="Weight for overall MAE in objective.")
    p.add_argument("--heavy_weight_25", type=float, default=0.30,
                   help="Weight for >25 and >40 mm/h RMSE in objective.")
    p.add_argument("--heavy_weight_50", type=float, default=0.25,
                   help="Weight for >50 mm/h RMSE in objective.")
    p.add_argument("--heavy_weight_75", type=float, default=0.20,
                   help="Weight for >75 mm/h RMSE in objective.")
    p.add_argument("--mid_bias_weight", type=float, default=3.0,
                   help="Mild penalty for positive bias in 10-50 mm/h range (safety net).")

    p.add_argument("--n_estimators",   type=int,   default=3000)
    p.add_argument("--early_stop",     type=int,   default=200)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(args.output_dir)
    log.info("LightGBM Optuna Tuning v10 (two-sided bias squeeze + wider w_max) started.")
    log.info("Config: %s", vars(args))
    tune(
        data_dir         = args.data_dir,
        model_dir        = args.model_dir,
        output_dir       = args.output_dir,
        n_trials         = args.n_trials,
        rmse_weight      = args.rmse_weight,
        mae_weight       = args.mae_weight,
        heavy_weight_25  = args.heavy_weight_25,
        heavy_weight_50  = args.heavy_weight_50,
        heavy_weight_75  = args.heavy_weight_75,
        mid_bias_weight  = args.mid_bias_weight,
        n_estimators     = args.n_estimators,
        early_stop       = args.early_stop,
        log              = log,
    )


if __name__ == "__main__":
    main()