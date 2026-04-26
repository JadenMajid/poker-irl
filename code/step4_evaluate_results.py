"""
step4_evaluate_results.py
--------------------------
Load IRL estimates and compute evaluation metrics:

  1. Per-parameter MSE and RMSE (alpha, beta, joint)
  2. Held-out Log-Likelihood (HOLL) — secondary metric
  3. HOLL recovery percentage — normalised quality metric
  4. Convergence quality analysis (how many steps to convergence)

Held-out Log-Likelihood (HOLL)
-------------------------------
The HOLL answers: "Under the recovered reward parameters, how likely is the
agent's HELD-OUT behaviour?"

We split collected hands 80/20 train/held-out.  For each held-out hand:

    HOLL = (1/N) Σ log π_{θ̂}(a_t | s_t)

where π_{θ̂} is the Boltzmann-rational policy under estimated parameters.
The Q-value adjustment is the same reparametrised form used in IRL:

    Q_{θ̂}(s, a_obs) = Q₀(s, a_obs) + alpha_norm_hat × A_var_norm + beta_hat × A_pot

Comparison baselines
--------------------
  HOLL_true    — upper bound: log-likelihood under the TRUE parameters
  HOLL_neutral — under (α=0, β=0): no reward shaping beyond base policy
  HOLL_random  — lower bound: uniform random policy = -log(n_legal)

HOLL recovery %:
    100 × (HOLL_est - HOLL_neutral) / |HOLL_true - HOLL_neutral|

This measures what fraction of the gap between neutral and true was recovered.

Output files
------------
  irl_results/evaluation_metrics.json   — aggregate scalar metrics
  irl_results/evaluation_details.json  — per-seat breakdown
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import ActorCriticNetwork, NUM_ACTIONS
from feature_encoder import FeatureEncoder, FEATURE_DIM
from game_state import NUM_PLAYERS
from reward import POT_NORM
from step3_collect_and_run_irl import (
    HandRecord,
    compute_rolling_variance_penalties,
    compute_mc_returns_per_hand,
    fill_var_penalties,
    IRLOptimiser,
    IRLOptimiser,
    CHECKPOINT_DIR,
    IRL_DIR,
    DEVICE,
    HIDDEN_DIM,
    VAR_WINDOW,
)

# ── configuration ──────────────────────────────────────────────────────────

TRAIN_SPLIT = 0.8

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HOLL computation helpers
# ---------------------------------------------------------------------------

def compute_holl_for_seat(
    step_data:   List[Tuple],
    target_net:  ActorCriticNetwork,
    alpha:       float,
    beta:        float,
    var_norm:    float,
    device:      torch.device,
    S:           float,
) -> float:
    """
    Compute mean held-out log-likelihood using the exact same feature extraction
    and logit-shifting logic as the IRLOptimiser.
    """
    if not step_data:
        return 0.0

    opt = IRLOptimiser(
        target_seat=-1,  # dummy seat
        step_data=step_data,
        opponent_models={},
        target_network=target_net,
        device=device,
        var_norm=var_norm,
        S=max(S, 1.0),
    )

    with torch.no_grad():
        opt.theta[0] = alpha
        opt.theta[1] = beta
        idx = torch.arange(opt.n_states, device=device)
        _, ll = opt._posterior_objective(idx)

    return float(ll.item())


def compute_random_holl(step_data: List[Tuple]) -> float:
    """Lower-bound HOLL: E[log π_uniform] = -log(n_legal)."""
    total = 0.0
    n     = 0
    for _, masks, _, _ in step_data:
        n_legal = int(masks[-1].sum())
        if n_legal > 0:
            total += -np.log(n_legal)
            n     += 1
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    is_ablation:  bool = False,
    ablation_tag: str  = "",
) -> Tuple[Dict, Dict]:
    """
    Load IRL estimates and trajectory data, compute all metrics.

    Returns (all_metrics, seat_details) dicts.
    """
    device = torch.device(DEVICE)
    suffix = f"_{ablation_tag}" if ablation_tag else ""

    # ── Load IRL estimates ─────────────────────────────────────────────────
    est_path = os.path.join(IRL_DIR, f"irl_estimates{suffix}.json")
    if not os.path.exists(est_path):
        raise FileNotFoundError(f"IRL estimates not found: {est_path}  "
                                "(run step3 first)")
    with open(est_path) as f:
        raw_ests = json.load(f)
    irl_estimates = {r["seat"]: r for r in raw_ests if "error" not in r}
    log.info("Loaded IRL estimates for %d seats.", len(irl_estimates))

    # ── Load trajectories ──────────────────────────────────────────────────
    traj_path = os.path.join(IRL_DIR, f"hand_records{suffix}.pkl")
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"Hand records not found: {traj_path}  "
                                "(run step3 first)")
    with open(traj_path, "rb") as f:
        import __main__
        import step3_collect_and_run_irl
        __main__.HandRecord = step3_collect_and_run_irl.HandRecord
        __main__.StepRecord = step3_collect_and_run_irl.StepRecord
        hand_records: List[HandRecord] = pickle.load(f)
    log.info("Loaded %d hand records.", len(hand_records))

    # 80/20 split (same split as used during IRL — consistent baselines)
    n_train       = int(len(hand_records) * TRAIN_SPLIT)
    train_records = hand_records[:n_train]
    held_records  = hand_records[n_train:]
    log.info("Split: %d train / %d held-out.", n_train, len(held_records))

    # Variance penalties — compute on TRAINING set for baselines,
    # independently on HELD-OUT set for evaluation
    var_ph_train, var_std_train = compute_rolling_variance_penalties(
        train_records, window=VAR_WINDOW
    )
    mc_train = compute_mc_returns_per_hand(train_records, var_ph_train)

    var_ph_held, var_std_held = compute_rolling_variance_penalties(
        held_records, window=VAR_WINDOW
    )
    mc_held = compute_mc_returns_per_hand(held_records, var_ph_held)

    chip_std_train: Dict[int, float] = {}
    for seat in range(NUM_PLAYERS):
        deltas = [r.chip_deltas.get(seat, 0.0) for r in train_records]
        chip_std_train[seat] = max(float(np.std(deltas)) if len(deltas) > 1 else 1.0, 1.0)

    chip_std_train: Dict[int, float] = {}
    for seat in range(NUM_PLAYERS):
        deltas = [r.chip_deltas.get(seat, 0.0) for r in train_records]
        chip_std_train[seat] = max(float(np.std(deltas)) if len(deltas) > 1 else 1.0, 1.0)

    # ── Load true parameters ───────────────────────────────────────────────
    params_key  = "ablation_agent_params.json" if is_ablation else "perturbed_agent_params.json"
    params_path = os.path.join(CHECKPOINT_DIR, params_key)
    with open(params_path) as f:
        true_params: Dict[int, Tuple[float, float]] = {
            p["seat"]: (p["alpha"], p["beta"]) for p in json.load(f)
        }

    seat_details: Dict[int, Dict] = {}

    # ── Per-seat evaluation ────────────────────────────────────────────────
    for seat in sorted(irl_estimates.keys()):
        est          = irl_estimates[seat]
        α_true, β_true = true_params[seat]
        α_hat        = est["est_alpha"]
        β_hat        = est["est_beta"]
        var_norm     = est.get("var_norm", var_std_train.get(seat, 1.0))

        log.info("\n--- Seat %d ---", seat)
        log.info("  True:  α=%.5f   β=%.4f", α_true, β_true)
        log.info("  Est:   α=%.5f   β=%.4f", α_hat,  β_hat)
        log.info("  VAR_NORM=%.0f", var_norm)

        # Load neutral base agent as Q₀ (same as step3)
        base_path = os.path.join(CHECKPOINT_DIR, "base_agent.pt")
        if not os.path.exists(base_path):
            log.warning("  Base agent checkpoint missing: %s", base_path)
            continue

        ckpt = torch.load(base_path, map_location=device)
        target_net = ActorCriticNetwork(
            input_dim=ckpt.get("feature_dim", FEATURE_DIM),
            hidden_dim=ckpt.get("hidden_dim",  HIDDEN_DIM),
        ).to(device)
        target_net.load_state_dict(ckpt["network_state"])
        target_net.eval()
        for p in target_net.parameters():
            p.requires_grad_(False)

        # Baselines from training set
        train_d = mc_train.get(seat, [])
        V_var_train = float(np.mean([d[3][-1, 1] for d in train_d])) if train_d else 0.0
        V_pot_train = float(np.mean([d[3][-1, 2] for d in train_d])) if train_d else 0.0

        held_d = mc_held.get(seat, [])
        if not held_d:
            log.warning("  No held-out data for seat %d.", seat)
            continue

        # HOLL under four parameter settings
        holl_est = compute_holl_for_seat(
            held_d, target_net, α_hat,   β_hat,   var_norm,
            device, chip_std_train[seat]
        )
        holl_true = compute_holl_for_seat(
            held_d, target_net, α_true,  β_true,  var_norm,
            device, chip_std_train[seat]
        )
        holl_neutral = compute_holl_for_seat(
            held_d, target_net, 0.0, 0.0, var_norm,
            device, chip_std_train[seat]
        )
        holl_random = compute_random_holl(held_d)

        α_mse     = (α_hat - α_true) ** 2
        β_mse     = (β_hat - β_true) ** 2
        joint_mse = (α_mse + β_mse) / 2.0

        holl_gap       = abs(holl_true - holl_neutral)
        holl_recovery  = (
            100.0 * (holl_est - holl_neutral) / holl_gap
            if holl_gap > 1e-8 else float("nan")
        )

        log.info("  α MSE:            %.6f  (error: %+.5f)", α_mse, α_hat - α_true)
        log.info("  β MSE:            %.6f  (error: %+.4f)", β_mse, β_hat - β_true)
        log.info("  Joint MSE:        %.6f", joint_mse)
        log.info("  HOLL estimated:   %.4f", holl_est)
        log.info("  HOLL true:        %.4f  ← upper bound", holl_true)
        log.info("  HOLL neutral:     %.4f", holl_neutral)
        log.info("  HOLL random:      %.4f  ← lower bound", holl_random)
        log.info("  HOLL recovery:    %.1f%%", holl_recovery)

        seat_details[seat] = {
            "seat":              seat,
            "true_alpha":        α_true,
            "true_beta":         β_true,
            "est_alpha":         α_hat,
            "est_beta":          β_hat,
            "alpha_mse":         α_mse,
            "beta_mse":          β_mse,
            "joint_mse":         joint_mse,
            "alpha_rmse":        α_mse ** 0.5,
            "beta_rmse":         β_mse ** 0.5,
            "holl_estimated":    holl_est,
            "holl_true":         holl_true,
            "holl_neutral":      holl_neutral,
            "holl_random":       holl_random,
            "holl_recovery_pct": holl_recovery,
            "n_irl_steps":       est.get("n_steps", None),
            "irl_converged":     est.get("converged", None),
            "var_norm":          var_norm,
        }

    # ── Aggregate ──────────────────────────────────────────────────────────
    all_metrics: Dict = {}
    if seat_details:
        vals = list(seat_details.values())
        all_metrics = {
            "mean_alpha_mse":          float(np.mean([v["alpha_mse"]       for v in vals])),
            "mean_beta_mse":           float(np.mean([v["beta_mse"]        for v in vals])),
            "mean_joint_mse":          float(np.mean([v["joint_mse"]       for v in vals])),
            "mean_alpha_rmse":         float(np.mean([v["alpha_rmse"]      for v in vals])),
            "mean_beta_rmse":          float(np.mean([v["beta_rmse"]       for v in vals])),
            "mean_holl_estimated":     float(np.mean([v["holl_estimated"]  for v in vals])),
            "mean_holl_true":          float(np.mean([v["holl_true"]       for v in vals])),
            "mean_holl_neutral":       float(np.mean([v["holl_neutral"]    for v in vals])),
            "mean_holl_random":        float(np.mean([v["holl_random"]     for v in vals])),
            "mean_holl_recovery_pct":  float(np.nanmean([v["holl_recovery_pct"] for v in vals])),
            "n_seats_evaluated":       len(seat_details),
            "is_ablation":             is_ablation,
        }

        log.info("\n" + "=" * 60)
        log.info("AGGREGATE EVALUATION METRICS")
        log.info("=" * 60)
        log.info("  Mean α MSE:          %.6f  (RMSE %.5f)",
                 all_metrics["mean_alpha_mse"], all_metrics["mean_alpha_rmse"])
        log.info("  Mean β MSE:          %.6f  (RMSE %.4f)",
                 all_metrics["mean_beta_mse"],  all_metrics["mean_beta_rmse"])
        log.info("  Mean joint MSE:      %.6f",  all_metrics["mean_joint_mse"])
        log.info("  Mean HOLL (est):     %.4f",  all_metrics["mean_holl_estimated"])
        log.info("  Mean HOLL (true):    %.4f",  all_metrics["mean_holl_true"])
        log.info("  Mean HOLL recovery:  %.1f%%",all_metrics["mean_holl_recovery_pct"])

    # ── Save ───────────────────────────────────────────────────────────────
    suffix2 = f"_{ablation_tag}" if ablation_tag else ""

    metrics_path = os.path.join(IRL_DIR, f"evaluation_metrics{suffix2}.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    log.info("Saved: %s", metrics_path)

    details_path = os.path.join(IRL_DIR, f"evaluation_details{suffix2}.json")
    with open(details_path, "w") as f:
        json.dump(list(seat_details.values()), f, indent=2)
    log.info("Saved: %s", details_path)

    return all_metrics, seat_details


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_evaluation()
