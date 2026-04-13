"""
step4_evaluate_results.py
--------------------------
Load the IRL estimates and compute evaluation metrics:

  1. Per-parameter MSE and RMSE (alpha, beta, joint)
  2. Held-out Log-Likelihood (HOLL) — the primary secondary metric
  3. Parameter recovery visualisation data (for inspection)
  4. Convergence quality analysis

Held-out Log-Likelihood (HOLL) methodology
-------------------------------------------
The HOLL answers: "Under the recovered reward parameters, how likely is the
agent's HELD-OUT behaviour?"

Setup:
  - Split collected hands into 80% train (used by IRL) / 20% held-out.
  - For the held-out set, for each (seat, step), evaluate:

        HOLL = (1/N_heldout) Σ_t log π_{θ̂}(a_t | s_t)

  where π_{θ̂} is the Boltzmann-rational policy under the estimated parameters:

        log π_θ(a_t | s_t) = Q_θ(s_t, a_t) - log Σ_{a'} exp(Q_θ(s_t, a'))

  with Q_θ(s,a) = Q₀(s,a) + α̂·A_α + β̂·A_pot  (same decomposition as IRL).

Comparison baselines for HOLL:
  (a) HOLL_neutral  — log-likelihood under (α=0, β=0) neutral policy
  (b) HOLL_true     — log-likelihood under the TRUE (α, β) parameters
  (c) HOLL_random   — expected log-likelihood of a uniform random policy
                    = -log(n_legal_actions) ≈ -log(5) ≈ -1.609

A good IRL result should have:
    HOLL_true > HOLL_estimate > HOLL_neutral > HOLL_random

Output files:
  irl_results/evaluation_metrics.json      — all scalar metrics
  irl_results/evaluation_details.json      — per-seat breakdown
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
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import ActorCriticNetwork, NUM_ACTIONS, legal_action_mask
from feature_encoder import FeatureEncoder, FEATURE_DIM
from game_state import NUM_PLAYERS, PlayerObservation
from reward import RewardParams, POT_NORM
from step3_collect_and_run_irl import (
    HandRecord,
    compute_rolling_variance_penalties,
    compute_mc_returns_per_hand,
    CHECKPOINT_DIR,
    IRL_DIR,
    DEVICE,
    HIDDEN_DIM,
)

# ── configuration ──────────────────────────────────────────────────────────

TRAIN_SPLIT = 0.8   # fraction of hands used for IRL (the rest are held-out)

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HOLL computation
# ---------------------------------------------------------------------------

def compute_holl_for_seat(
    seat:         int,
    step_data:    List[Tuple],    # from compute_mc_returns_per_hand
    target_net:   ActorCriticNetwork,
    alpha:        float,
    beta:         float,
    device:       torch.device,
    V_var:        float,
    V_pot:        float,
) -> float:
    """
    Compute the held-out log-likelihood for one set of reward parameters.

    Parameters
    ----------
    step_data  : List of (feats, masks, acts, returns) for held-out hands.
    target_net : Frozen target agent network (provides Q₀ base logits).
    alpha, beta: Reward parameters to evaluate.
    V_var, V_pot: Advantage baselines (estimated on training set).

    Returns
    -------
    Mean log π_θ(a_t | s_t) over all terminal steps in step_data.
    """
    total_ll = 0.0
    n_steps  = 0

    for feats, masks, acts, returns in step_data:
        feat_t = torch.tensor(feats[-1:], dtype=torch.float32, device=device)
        mask_t = torch.tensor(masks[-1:], dtype=torch.bool,    device=device)
        a_obs  = int(acts[-1])

        with torch.no_grad():
            base_logits, _ = target_net(feat_t, mask_t)

        adj_logits  = base_logits[0].clone().float()
        A_var       = float(returns[-1, 1]) - V_var
        A_pot       = float(returns[-1, 2]) - V_pot
        shaping     = alpha * A_var + beta * A_pot
        adj_logits[a_obs] = adj_logits[a_obs] + shaping

        # log π_θ(a_obs | s)
        legal_mask_bool = mask_t[0]
        log_z    = torch.logsumexp(adj_logits[legal_mask_bool], dim=0)
        ll       = (adj_logits[a_obs] - log_z).item()
        total_ll += ll
        n_steps  += 1

    return total_ll / max(n_steps, 1)


def compute_random_holl(step_data: List[Tuple]) -> float:
    """
    Baseline: HOLL under a uniform random policy.
    E[log π_uniform(a | s)] = -log(n_legal_actions)
    """
    total_ll = 0.0
    n        = 0
    for feats, masks, acts, returns in step_data:
        n_legal = int(masks[-1].sum())
        if n_legal > 0:
            total_ll += -np.log(n_legal)
            n += 1
    return total_ll / max(n, 1)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_evaluation(is_ablation: bool = False, ablation_tag: str = "") -> None:
    device = torch.device(DEVICE)
    suffix = f"_{ablation_tag}" if ablation_tag else ""

    # ── Load IRL estimates ─────────────────────────────────────────────────
    estimates_path = os.path.join(IRL_DIR, f"irl_estimates{suffix}.json")
    with open(estimates_path) as f:
        irl_estimates = {r["seat"]: r for r in json.load(f)}
    log.info("Loaded IRL estimates for %d seats.", len(irl_estimates))

    # ── Load trajectory data ───────────────────────────────────────────────
    traj_path = os.path.join(IRL_DIR, f"hand_records{suffix}.pkl")
    with open(traj_path, "rb") as f:
        hand_records: List[HandRecord] = pickle.load(f)
    log.info("Loaded %d hand records.", len(hand_records))

    n_train = int(len(hand_records) * TRAIN_SPLIT)
    train_records = hand_records[:n_train]
    held_records  = hand_records[n_train:]
    log.info("Split: %d train / %d held-out.", len(train_records), len(held_records))

    # Compute variance penalties on TRAINING set (for baselines)
    var_pen_train = compute_rolling_variance_penalties(train_records, window=100)
    mc_train      = compute_mc_returns_per_hand(train_records, var_pen_train)

    # Compute variance penalties on HELD-OUT set
    var_pen_held  = compute_rolling_variance_penalties(held_records, window=100)
    mc_held       = compute_mc_returns_per_hand(held_records, var_pen_held)

    # ── Load true parameters ───────────────────────────────────────────────
    params_path = os.path.join(CHECKPOINT_DIR,
                               "ablation_agent_params.json" if is_ablation
                               else "perturbed_agent_params.json")
    with open(params_path) as f:
        true_params = {p["seat"]: (p["alpha"], p["beta"]) for p in json.load(f)}

    all_metrics   = {}
    seat_details  = {}

    # ── Per-seat evaluation ────────────────────────────────────────────────
    for seat in sorted(irl_estimates.keys()):
        est     = irl_estimates[seat]
        α_true, β_true = true_params[seat]
        α_hat   = est["est_alpha"]
        β_hat   = est["est_beta"]

        log.info("\n--- Seat %d ---", seat)
        log.info("  True: α=%.4f  β=%.4f", α_true, β_true)
        log.info("  Est:  α=%.4f  β=%.4f", α_hat,  β_hat)

        # Load target network
        agent_path = os.path.join(
            CHECKPOINT_DIR,
            f"ablation_perturbed_agent_0.pt" if is_ablation
            else f"perturbed_agent_{seat}.pt"
        )
        ckpt = torch.load(agent_path, map_location=device)
        target_net = ActorCriticNetwork(
            input_dim=ckpt.get("feature_dim", FEATURE_DIM),
            hidden_dim=ckpt.get("hidden_dim",  HIDDEN_DIM),
        ).to(device)
        target_net.load_state_dict(ckpt["network_state"])
        target_net.eval()

        # Compute baselines from TRAINING set
        train_data = mc_train.get(seat, [])
        V_var_train = float(np.mean([d[3][-1, 1] for d in train_data])) if train_data else 0.0
        V_pot_train = float(np.mean([d[3][-1, 2] for d in train_data])) if train_data else 0.0

        held_data = mc_held.get(seat, [])
        if not held_data:
            log.warning("  No held-out data for seat %d.", seat)
            continue

        # HOLL under estimated parameters
        holl_est = compute_holl_for_seat(
            seat, held_data, target_net, α_hat, β_hat, device, V_var_train, V_pot_train
        )
        # HOLL under true parameters
        holl_true = compute_holl_for_seat(
            seat, held_data, target_net, α_true, β_true, device, V_var_train, V_pot_train
        )
        # HOLL under neutral (0, 0)
        holl_neutral = compute_holl_for_seat(
            seat, held_data, target_net, 0.0, 0.0, device, V_var_train, V_pot_train
        )
        # HOLL under random policy
        holl_random = compute_random_holl(held_data)

        α_mse  = (α_hat - α_true) ** 2
        β_mse  = (β_hat - β_true) ** 2
        joint_mse = (α_mse + β_mse) / 2.0

        log.info("  α MSE:  %.6f  (α error: %.4f)", α_mse, α_hat - α_true)
        log.info("  β MSE:  %.6f  (β error: %.4f)", β_mse, β_hat - β_true)
        log.info("  Joint MSE: %.6f", joint_mse)
        log.info("  HOLL_estimated: %.4f", holl_est)
        log.info("  HOLL_true:      %.4f  (upper bound)", holl_true)
        log.info("  HOLL_neutral:   %.4f", holl_neutral)
        log.info("  HOLL_random:    %.4f  (lower bound)", holl_random)
        log.info("  HOLL recovery:  %.2f%%",
                 100 * (holl_est - holl_neutral) / max(abs(holl_true - holl_neutral), 1e-8))

        seat_details[seat] = {
            "seat":          seat,
            "true_alpha":    α_true,
            "true_beta":     β_true,
            "est_alpha":     α_hat,
            "est_beta":      β_hat,
            "alpha_mse":     α_mse,
            "beta_mse":      β_mse,
            "joint_mse":     joint_mse,
            "alpha_rmse":    α_mse ** 0.5,
            "beta_rmse":     β_mse ** 0.5,
            "holl_estimated":holl_est,
            "holl_true":     holl_true,
            "holl_neutral":  holl_neutral,
            "holl_random":   holl_random,
            "holl_recovery_pct": 100 * (holl_est - holl_neutral) / max(abs(holl_true - holl_neutral), 1e-8),
        }

    # ── Aggregate metrics ──────────────────────────────────────────────────
    if seat_details:
        mean_α_mse    = float(np.mean([v["alpha_mse"]  for v in seat_details.values()]))
        mean_β_mse    = float(np.mean([v["beta_mse"]   for v in seat_details.values()]))
        mean_joint    = float(np.mean([v["joint_mse"]  for v in seat_details.values()]))
        mean_holl_est = float(np.mean([v["holl_estimated"] for v in seat_details.values()]))
        mean_holl_tr  = float(np.mean([v["holl_true"]  for v in seat_details.values()]))
        mean_recovery = float(np.mean([v["holl_recovery_pct"] for v in seat_details.values()]))

        all_metrics = {
            "mean_alpha_mse":          mean_α_mse,
            "mean_beta_mse":           mean_β_mse,
            "mean_joint_mse":          mean_joint,
            "mean_alpha_rmse":         mean_α_mse ** 0.5,
            "mean_beta_rmse":          mean_β_mse ** 0.5,
            "mean_holl_estimated":     mean_holl_est,
            "mean_holl_true":          mean_holl_tr,
            "mean_holl_recovery_pct":  mean_recovery,
            "n_seats_evaluated":       len(seat_details),
            "is_ablation":             is_ablation,
        }

        log.info("\n" + "="*60)
        log.info("AGGREGATE EVALUATION METRICS")
        log.info("="*60)
        log.info("  Mean α MSE:          %.6f  (RMSE %.4f)", mean_α_mse, mean_α_mse**0.5)
        log.info("  Mean β MSE:          %.6f  (RMSE %.4f)", mean_β_mse, mean_β_mse**0.5)
        log.info("  Mean joint MSE:      %.6f", mean_joint)
        log.info("  Mean HOLL (est):     %.4f", mean_holl_est)
        log.info("  Mean HOLL (true):    %.4f", mean_holl_tr)
        log.info("  Mean HOLL recovery:  %.2f%%", mean_recovery)

    # ── Save ───────────────────────────────────────────────────────────────
    out_path = os.path.join(IRL_DIR, f"evaluation_metrics{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    log.info("Saved: %s", out_path)

    detail_path = os.path.join(IRL_DIR, f"evaluation_details{suffix}.json")
    with open(detail_path, "w") as f:
        json.dump(list(seat_details.values()), f, indent=2)
    log.info("Saved: %s", detail_path)

    return all_metrics, seat_details


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_evaluation()
