"""
step5b_run_ablation_irl_and_compare.py
---------------------------------------
Ablation study — Phase 2: Collect ablation trajectories, run IRL, compare.

Setup:
  - Seat 0: the fine-tuned ablation agent (α=+0.004, β=+0.25), adapted to
    play against fixed neutral opponents.
  - Seats 1–3: frozen neutral base policy agents.

We collect fresh trajectories using this mix, run the same GABO-IRL on seat 0,
evaluate with step4 metrics, then compare against the main experiment results.

Scientific question answered:
  Does co-adaptation of multiple agents make IRL harder or easier?

  Main:      4 co-adapted agents  → IRL per agent  → MSE_main
  Ablation:  1 adapted + 3 fixed  → IRL on seat 0  → MSE_ablation

Output:
  irl_results/hand_records_ablation.pkl
  irl_results/irl_estimates_ablation.json
  irl_results/irl_convergence_log_ablation.json
  irl_results/evaluation_metrics_ablation.json
  irl_results/evaluation_details_ablation.json
  irl_results/ablation_comparison.json   ← final comparison table
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import ActorCriticNetwork, index_to_action, legal_action_mask
from feature_encoder import FeatureEncoder, FEATURE_DIM
from game_state import ActionType, NUM_PLAYERS, PlayerObservation, Action
from poker_env import PokerEnv
from reward import POT_NORM
from step3_collect_and_run_irl import (
    HandRecord,
    StepRecord,
    BehaviourCloningNet,
    compute_rolling_variance_penalties,
    compute_mc_returns_per_hand,
    fill_var_penalties,
    train_opponent_model,
    run_irl_for_seat,
    CHECKPOINT_DIR,
    IRL_DIR,
    DEVICE,
    HIDDEN_DIM,
    N_COLLECTION_HANDS,
    LOG_COLLECT_EVERY,
    OPP_MIN_SAMPLES,
    VAR_WINDOW,
    IRL_GRAD_ACCUM_STEPS,
    IRL_BATCH_SIZE,
)
from step4_evaluate_results import run_evaluation

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TAG = "ablation"


# ---------------------------------------------------------------------------
# Ablation trajectory collector
# ---------------------------------------------------------------------------

def collect_ablation_trajectories(n_hands: int) -> List[HandRecord]:
    """
    Collect trajectories for the ablation setting:
      Seat 0: fine-tuned ablation agent (adapted, perturbed reward)
      Seats 1–3: frozen neutral base policy

    Returns a list of HandRecord objects.
    """
    device  = torch.device(DEVICE)
    encoder = FeatureEncoder()

    def _load_net(path: str) -> ActorCriticNetwork:
        ckpt = torch.load(path, map_location=device)
        net  = ActorCriticNetwork(
            input_dim=ckpt.get("feature_dim", FEATURE_DIM),
            hidden_dim=ckpt.get("hidden_dim",  HIDDEN_DIM),
        ).to(device)
        net.load_state_dict(ckpt["network_state"])
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        return net

    abl_path  = os.path.join(CHECKPOINT_DIR, "ablation_perturbed_agent_0.pt")
    base_path = os.path.join(CHECKPOINT_DIR, "base_agent.pt")

    if not os.path.exists(abl_path):
        raise FileNotFoundError(f"Ablation agent not found: {abl_path}")
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base agent not found: {base_path}")

    networks = {0: _load_net(abl_path)}
    for s in [1, 2, 3]:
        networks[s] = _load_net(base_path)

    records: List[HandRecord] = []
    start = time.time()

    for hand_i in range(n_hands):
        hand_steps: Dict[int, List] = {i: [] for i in range(NUM_PLAYERS)}

        def make_cb(seat: int, net: ActorCriticNetwork):
            def cb(obs: PlayerObservation) -> Action:
                feat   = encoder.encode(obs)
                mask   = legal_action_mask(obs)
                feat_t = torch.tensor(feat, dtype=torch.float32,
                                      device=device).unsqueeze(0)
                mask_t = mask.unsqueeze(0).to(device)
                with torch.no_grad():
                    logits, _ = net(feat_t, mask_t)
                    dist      = Categorical(logits=logits.squeeze(0))
                    idx       = int(dist.sample().item())
                action = index_to_action(idx, seat)
                hand_steps[seat].append((feat, mask.numpy(), idx))
                return action
            return cb

        env  = PokerEnv(
            [make_cb(i, networks[i]) for i in range(NUM_PLAYERS)],
            record_trajectories=True,
        )
        traj = env.play_hand()

        chip_deltas = {i: float(traj.final_chip_deltas.get(i, 0))
                       for i in range(NUM_PLAYERS)}

        max_pots: Dict[int, float] = {}
        for seat in range(NUM_PLAYERS):
            mp = 0.0
            for step in traj.steps:
                if step.acting_seat == seat:
                    if step.action.action_type in (ActionType.CALL, ActionType.RAISE):
                        mp = max(mp, float(step.observation.pot))
            max_pots[seat] = mp

        steps_by_seat: Dict[int, List[StepRecord]] = {i: [] for i in range(NUM_PLAYERS)}
        for seat in range(NUM_PLAYERS):
            seat_steps = hand_steps[seat]
            n_s        = len(seat_steps)
            for k, (feat, mask_np, idx) in enumerate(seat_steps):
                is_last = (k == n_s - 1)
                steps_by_seat[seat].append(StepRecord(
                    seat=seat,
                    feature=feat,
                    action_idx=idx,
                    legal_mask=mask_np,
                    reward_chip=chip_deltas[seat] if is_last else 0.0,
                    reward_var_pen=0.0,
                    reward_pot=(max_pots[seat] / POT_NORM) if is_last else 0.0,
                    is_terminal=is_last,
                    hand_id=hand_i,
                ))

        records.append(HandRecord(
            hand_id=hand_i,
            steps=steps_by_seat,
            chip_deltas=chip_deltas,
            max_pots=max_pots,
        ))

        if (hand_i + 1) % LOG_COLLECT_EVERY == 0:
            elapsed  = time.time() - start
            hands_ph = (hand_i + 1) / max(elapsed, 1) * 3600
            log.info("  Ablation collect: %6d / %6d | %.0f hands/hr",
                     hand_i + 1, n_hands, hands_ph)

    return records


# ---------------------------------------------------------------------------
# Main ablation IRL + comparison
# ---------------------------------------------------------------------------

def run_ablation_comparison() -> None:
    os.makedirs(IRL_DIR, exist_ok=True)
    device = torch.device(DEVICE)

    # ── Step 1: Collect ablation trajectories ─────────────────────────────
    abl_traj_path = os.path.join(IRL_DIR, f"hand_records_{TAG}.pkl")
    if os.path.exists(abl_traj_path):
        log.info("Loading cached ablation trajectories ...")
        with open(abl_traj_path, "rb") as f:
            hand_records: List[HandRecord] = pickle.load(f)
        log.info("  Loaded %d hand records.", len(hand_records))
    else:
        log.info("Collecting %d ablation trajectory hands ...", N_COLLECTION_HANDS)
        hand_records = collect_ablation_trajectories(N_COLLECTION_HANDS)
        with open(abl_traj_path, "wb") as f:
            pickle.dump(hand_records, f)
        log.info("  Saved → %s", abl_traj_path)

    # ── Step 2: Variance penalties + MC data ──────────────────────────────
    log.info("Computing rolling variance ...")
    var_per_hand, var_std = compute_rolling_variance_penalties(
        hand_records, window=VAR_WINDOW
    )
    fill_var_penalties(hand_records, var_per_hand)
    mc_data = compute_mc_returns_per_hand(hand_records, var_per_hand)

    # ── Step 3: Load ablation agent true params ────────────────────────────
    params_path = os.path.join(CHECKPOINT_DIR, "ablation_agent_params.json")
    with open(params_path) as f:
        abl_params_list = json.load(f)
    # Find the perturbed seat (seat 0, the one that was trained)
    abl_params_map = {p["seat"]: (p["alpha"], p["beta"]) for p in abl_params_list}
    # Only run IRL on seat 0 (the adapted agent)
    target_seat  = 0
    true_alpha, true_beta = abl_params_map[target_seat]

    # ── Step 4: Load neutral base policy as Q₀ ────────────────────────────
    #   Same as step3: logit_base = neutral base network.
    base_net_path = os.path.join(CHECKPOINT_DIR, "base_agent.pt")
    if not os.path.exists(base_net_path):
        raise FileNotFoundError(f"Base agent not found: {base_net_path}")
    base_ckpt = torch.load(base_net_path, map_location=device)
    target_net = ActorCriticNetwork(
        input_dim=base_ckpt.get("feature_dim", FEATURE_DIM),
        hidden_dim=base_ckpt.get("hidden_dim",  HIDDEN_DIM),
    ).to(device)
    target_net.load_state_dict(base_ckpt["network_state"])
    target_net.eval()
    for p in target_net.parameters():
        p.requires_grad_(False)
    log.info("Using neutral base agent as Q₀: %s", base_net_path)

    # ── Step 6: Train opponent models for the 3 fixed neutral seats ───────
    opponent_models: Dict[int, BehaviourCloningNet] = {}
    for opp_seat in [1, 2, 3]:
        opp_data = mc_data.get(opp_seat, [])
        if len(opp_data) < OPP_MIN_SAMPLES:
            log.warning("  Insufficient data for ablation opponent model seat %d (%d samples).",
                        opp_seat, len(opp_data))
            continue
        log.info("  Training ablation opponent model — seat %d (%d hands) ...",
                 opp_seat, len(opp_data))
        all_feats = np.concatenate([d[0] for d in opp_data])
        all_masks = np.concatenate([d[1] for d in opp_data])
        all_acts  = np.concatenate([d[2] for d in opp_data])
        opp_net   = train_opponent_model(all_feats, all_masks, all_acts, device)
        opponent_models[opp_seat] = opp_net
        opp_save  = os.path.join(IRL_DIR, f"opp_model_abl_target0_opp{opp_seat}.pt")
        torch.save(opp_net.state_dict(), opp_save)

    # ── Step 7: Run IRL on seat 0 ─────────────────────────────────────────
    log.info("\n" + "=" * 62)
    log.info("ABLATION IRL — Target seat 0")
    log.info("  (grad_accum=%d, eff. batch=%d hands — inherited from step3)",
             IRL_GRAD_ACCUM_STEPS, IRL_BATCH_SIZE * IRL_GRAD_ACCUM_STEPS)
    log.info("=" * 62)

    abl_result = run_irl_for_seat(
        target_seat=target_seat,
        step_data=mc_data[target_seat],
        opponent_models=opponent_models,
        target_network=target_net,
        device=device,
        true_alpha=true_alpha,
        true_beta=true_beta,
        var_norm=var_std[target_seat],
    )

    # Save ablation IRL estimates (needed by step4 via ablation_tag)
    abl_est_path = os.path.join(IRL_DIR, f"irl_estimates_{TAG}.json")
    summary = [{k: v for k, v in abl_result.items()
                if k not in ("alpha_history", "beta_history", "ll_history")}]
    with open(abl_est_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved: %s", abl_est_path)

    abl_conv_path = os.path.join(IRL_DIR, f"irl_convergence_log_{TAG}.json")
    with open(abl_conv_path, "w") as f:
        json.dump([abl_result], f, indent=2)
    log.info("Saved: %s", abl_conv_path)

    # ── Step 7: Evaluate ablation (use step4 machinery) ───────────────────
    log.info("\nEvaluating ablation IRL ...")
    abl_metrics, abl_details = run_evaluation(is_ablation=True, ablation_tag=TAG)

    # ── Step 8: Load main experiment metrics ──────────────────────────────
    main_metrics_path = os.path.join(IRL_DIR, "evaluation_metrics.json")
    if not os.path.exists(main_metrics_path):
        log.warning("Main experiment metrics not found.  Run step4 first for comparison.")
        main_metrics: Dict = {}
    else:
        with open(main_metrics_path) as f:
            main_metrics = json.load(f)

    # ── Step 9: Build comparison report ───────────────────────────────────
    def _get(d: Dict, key: str):
        return d.get(key, float("nan"))

    abl_joint = _get(abl_metrics, "mean_joint_mse")
    main_joint = _get(main_metrics, "mean_joint_mse")
    ratio = abl_joint / max(main_joint, 1e-10) if not np.isnan(abl_joint) else float("nan")

    if np.isnan(ratio):
        interpretation = "Cannot compare — one or both results are missing."
    elif ratio < 0.85:
        interpretation = (
            "Ablation MSE << Main MSE: co-adaptation HURTS IRL recovery.  "
            "Single-agent setting (fixed opponents) is an easier IRL problem.  "
            "The multi-agent IRL with opponent modelling overcomplicates the simpler case."
        )
    elif ratio > 1.15:
        interpretation = (
            "Ablation MSE >> Main MSE: co-adaptation HELPS IRL recovery, or "
            "the opponent modelling in the multi-agent IRL successfully handles "
            "the richer trajectory variation produced by co-adapted agents.  "
            "The multi-agent setting is informative rather than confounding."
        )
    else:
        interpretation = (
            "Ablation MSE ≈ Main MSE: co-adaptation has minimal effect on IRL "
            "recovery accuracy.  Treating co-players as a static environment "
            "(the Nash equilibrium assumption) is a reasonable approximation.  "
            "GABO-IRL generalises to both settings."
        )

    comparison = {
        "main_experiment": {
            "description":         "4 co-adapted agents; IRL on each individually",
            "mean_alpha_mse":      _get(main_metrics, "mean_alpha_mse"),
            "mean_beta_mse":       _get(main_metrics, "mean_beta_mse"),
            "mean_joint_mse":      _get(main_metrics, "mean_joint_mse"),
            "mean_alpha_rmse":     _get(main_metrics, "mean_alpha_rmse"),
            "mean_beta_rmse":      _get(main_metrics, "mean_beta_rmse"),
            "mean_holl_estimated": _get(main_metrics, "mean_holl_estimated"),
            "mean_holl_recovery_pct": _get(main_metrics, "mean_holl_recovery_pct"),
        },
        "ablation": {
            "description":         "1 adaptive agent vs 3 fixed neutral; IRL on seat 0",
            "mean_alpha_mse":      _get(abl_metrics, "mean_alpha_mse"),
            "mean_beta_mse":       _get(abl_metrics, "mean_beta_mse"),
            "mean_joint_mse":      _get(abl_metrics, "mean_joint_mse"),
            "mean_alpha_rmse":     _get(abl_metrics, "mean_alpha_rmse"),
            "mean_beta_rmse":      _get(abl_metrics, "mean_beta_rmse"),
            "mean_holl_estimated": _get(abl_metrics, "mean_holl_estimated"),
            "mean_holl_recovery_pct": _get(abl_metrics, "mean_holl_recovery_pct"),
        },
        "ablation_vs_main_joint_mse_ratio": float(ratio),
        "interpretation": interpretation,
    }

    comp_path = os.path.join(IRL_DIR, "ablation_comparison.json")
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)
    log.info("Saved: %s", comp_path)

    # ── Print report ───────────────────────────────────────────────────────
    log.info("\n" + "=" * 72)
    log.info("ABLATION vs MAIN EXPERIMENT COMPARISON")
    log.info("=" * 72)
    log.info("")
    log.info("  %-34s  %14s  %14s", "Metric", "Main (4-agent)", "Ablation (1-agent)")
    log.info("  " + "-" * 66)

    metrics_to_print = [
        ("mean_alpha_mse",         "Mean α MSE              "),
        ("mean_beta_mse",          "Mean β MSE              "),
        ("mean_joint_mse",         "Mean Joint MSE          "),
        ("mean_alpha_rmse",        "Mean α RMSE             "),
        ("mean_beta_rmse",         "Mean β RMSE             "),
        ("mean_holl_estimated",    "Mean HOLL (estimated)   "),
        ("mean_holl_recovery_pct", "HOLL Recovery (%)       "),
    ]
    for key, label in metrics_to_print:
        mv = comparison["main_experiment"][key]
        av = comparison["ablation"][key]
        mv_s = f"{mv:.4f}" if not np.isnan(float(mv) if mv is not None else float("nan")) else "N/A"
        av_s = f"{av:.4f}" if not np.isnan(float(av) if av is not None else float("nan")) else "N/A"
        log.info("  %-34s  %14s  %14s", label, mv_s, av_s)

    log.info("")
    log.info("  Ablation / Main joint MSE ratio: %.3f", ratio)
    log.info("")
    log.info("  INTERPRETATION:")
    for line in interpretation.split(".  "):
        if line.strip():
            log.info("  → %s.", line.strip().rstrip("."))
    log.info("")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_ablation_comparison()
