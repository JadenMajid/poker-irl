"""
step5b_run_ablation_irl_and_compare.py
---------------------------------------
Ablation study — Phase 2: IRL + Comparison

Load the ablation agent (seat 0, adapted to fixed neutral opponents), collect
a fresh set of fixed-policy trajectories, run the same GABO-IRL procedure,
and compare the estimated (alpha, beta) MSE to the main experiment.

Scientific question:
  Does co-adaptation of multiple agents make IRL harder?

  Main experiment:  4 co-adapted agents  → IRL on each → MSE_main
  Ablation:         1 adapted + 3 fixed  → IRL on seat 0 → MSE_ablation

If MSE_ablation < MSE_main:
  The co-adaptation assumption DID hurt IRL recovery.  The simpler single-
  agent case (which trivially satisfies the Nash-equilibrium assumption)
  yielded better parameter recovery.  This would suggest that for the main
  experiment, the implicit opponent modelling error compounds the IRL
  difficulty.

If MSE_main ≈ MSE_ablation:
  Our GABO-IRL with opponent modelling successfully handles the multi-agent
  co-adaptation, and the single/multi-agent distinction does not matter
  significantly for IRL recovery accuracy.

If MSE_ablation > MSE_main:
  Surprising — co-adaptation HELPS IRL.  This could occur if the co-adapted
  agents display richer behavioural variation (more informative trajectories)
  than the single agent playing against fixed opponents.

Output:
  irl_results/ablation_comparison.json  — all comparison statistics
  (Also prints a formatted comparison table to the log.)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

import numpy as np


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from step3_collect_and_run_irl import (
    run_collection_and_irl,
    CHECKPOINT_DIR,
    IRL_DIR,
    N_COLLECTION_HANDS,
)
from step4_evaluate_results import run_evaluation

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ablation trajectory collection patch
# ---------------------------------------------------------------------------

def collect_ablation_trajectories(n_hands: int):
    """
    Collect trajectories for the ablation setting:
      - Seat 0: the adapted ablation agent (perturbed reward)
      - Seats 1, 2, 3: frozen neutral base policy agents

    We reuse the main collect_trajectories function but override which
    agent files to load.  This is done by temporarily renaming / symlinking
    the expected checkpoint files — or more cleanly, by reimplementing the
    collector.
    """
    import torch
    from agent import ActorCriticNetwork, index_to_action, legal_action_mask
    from feature_encoder import FeatureEncoder, FEATURE_DIM
    from game_state import NUM_PLAYERS, PlayerObservation, Action
    from poker_env import PokerEnv
    from step3_collect_and_run_irl import HandRecord, StepRecord
    from reward import POT_NORM
    from game_state import ActionType
    import pickle
    from torch.distributions import Categorical

    DEVICE     = "cpu"
    HIDDEN_DIM = 256
    device     = torch.device(DEVICE)
    encoder    = FeatureEncoder()

    # Load ablation agent (seat 0) and base agent (seats 1-3)
    def load_net(path: str) -> ActorCriticNetwork:
        ckpt = torch.load(path, map_location=device)
        net  = ActorCriticNetwork(
            input_dim=ckpt.get("feature_dim", FEATURE_DIM),
            hidden_dim=ckpt.get("hidden_dim",  HIDDEN_DIM),
        ).to(device)
        net.load_state_dict(ckpt["network_state"])
        net.eval()
        return net

    networks = {
        0: load_net(os.path.join(CHECKPOINT_DIR, "ablation_perturbed_agent_0.pt")),
    }
    base_ckpt = os.path.join(CHECKPOINT_DIR, "base_agent.pt")
    for s in [1, 2, 3]:
        networks[s] = load_net(base_ckpt)

    records = []
    start   = time.time()

    LOG_EVERY = 5000
    for hand_i in range(n_hands):
        hand_steps = {i: [] for i in range(NUM_PLAYERS)}

        def make_cb(seat: int, net: ActorCriticNetwork):
            def cb(obs: PlayerObservation) -> Action:
                feat   = encoder.encode(obs)
                mask   = legal_action_mask(obs)
                feat_t = torch.tensor(feat, dtype=torch.float32, device=device).unsqueeze(0)
                mask_t = mask.unsqueeze(0).to(device)
                with torch.no_grad():
                    logits, _ = net(feat_t, mask_t)
                    dist      = Categorical(logits=logits.squeeze(0))
                    idx       = int(dist.sample().item())
                action = index_to_action(idx, seat)
                hand_steps[seat].append((feat, mask.numpy(), idx, obs))
                return action
            return cb

        env  = PokerEnv(
            [make_cb(i, networks[i]) for i in range(NUM_PLAYERS)],
            record_trajectories=True,
        )
        traj = env.play_hand()

        chip_deltas = {i: float(traj.final_chip_deltas.get(i, 0)) for i in range(NUM_PLAYERS)}
        max_pots:  dict = {}
        for seat in range(NUM_PLAYERS):
            mp = 0.0
            for step in traj.steps:
                if step.acting_seat == seat:
                    if step.action.action_type in (ActionType.CALL, ActionType.RAISE):
                        mp = max(mp, float(step.observation.pot))
            max_pots[seat] = mp

        steps_by_seat = {i: [] for i in range(NUM_PLAYERS)}
        for seat in range(NUM_PLAYERS):
            seat_steps = hand_steps[seat]
            n          = len(seat_steps)
            for k, (feat, mask_np, idx, obs) in enumerate(seat_steps):
                is_last = (k == n - 1)
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

        if (hand_i + 1) % LOG_EVERY == 0:
            elapsed  = time.time() - start
            hands_ph = (hand_i + 1) / max(elapsed, 1) * 3600
            log.info("  Ablation collect: %6d / %6d hands | %.0f hands/hr",
                     hand_i + 1, n_hands, hands_ph)

    return records


# ---------------------------------------------------------------------------
# Main ablation IRL + comparison
# ---------------------------------------------------------------------------

def run_ablation_comparison() -> None:
    os.makedirs(IRL_DIR, exist_ok=True)

    # ── Step 1: Collect ablation trajectories ─────────────────────────────
    import pickle
    from step3_collect_and_run_irl import (
        compute_rolling_variance_penalties,
        compute_mc_returns_per_hand,
    )
    from step3_collect_and_run_irl import run_irl_for_seat, IRL_N_STEPS
    import torch

    abl_traj_path = os.path.join(IRL_DIR, "hand_records_ablation.pkl")
    if os.path.exists(abl_traj_path):
        log.info("Loading cached ablation trajectories from %s ...", abl_traj_path)
        with open(abl_traj_path, "rb") as f:
            hand_records = pickle.load(f)
    else:
        log.info("Collecting %d ablation trajectories ...", N_COLLECTION_HANDS)
        hand_records = collect_ablation_trajectories(N_COLLECTION_HANDS)
        with open(abl_traj_path, "wb") as f:
            pickle.dump(hand_records, f)
        log.info("Saved ablation trajectories → %s", abl_traj_path)

    # ── Step 2: Run IRL on seat 0 only ────────────────────────────────────
    log.info("Computing MC return data for ablation ...")
    var_pen = compute_rolling_variance_penalties(hand_records, window=100)
    mc_data = compute_mc_returns_per_hand(hand_records, var_pen)

    device     = torch.device("cpu")
    HIDDEN_DIM = 256
    DEVICE     = "cpu"

    # Load ablation agent network
    agent_path = os.path.join(CHECKPOINT_DIR, "ablation_perturbed_agent_0.pt")
    ckpt       = torch.load(agent_path, map_location=device)
    from agent import ActorCriticNetwork
    target_net = ActorCriticNetwork(
        input_dim=ckpt.get("feature_dim", FEATURE_DIM),
        hidden_dim=ckpt.get("hidden_dim",  HIDDEN_DIM),
    ).to(device)
    target_net.load_state_dict(ckpt["network_state"])
    target_net.eval()
    from feature_encoder import FEATURE_DIM
    for p in target_net.parameters():
        p.requires_grad_(False)

    # Load true params
    with open(os.path.join(CHECKPOINT_DIR, "ablation_agent_params.json")) as f:
        abl_params = {p["seat"]: (p["alpha"], p["beta"]) for p in json.load(f)}

    true_alpha, true_beta = abl_params[0]

    # Train opponent models for ablation
    from step3_collect_and_run_irl import (
        train_opponent_model, OPP_MIN_SAMPLES, BehaviourCloningNet
    )
    opponent_models = {}
    for opp_seat in [1, 2, 3]:
        opp_data = mc_data[opp_seat]
        if len(opp_data) < OPP_MIN_SAMPLES:
            continue
        all_feats = np.concatenate([d[0] for d in opp_data], axis=0)
        all_masks = np.concatenate([d[1] for d in opp_data], axis=0)
        all_acts  = np.concatenate([d[2] for d in opp_data], axis=0)
        log.info("  Training ablation opponent model for seat %d (%d samples) ...",
                 opp_seat, len(all_feats))
        opponent_models[opp_seat] = train_opponent_model(all_feats, all_masks, all_acts, device)

    # Run IRL for seat 0
    abl_result = run_irl_for_seat(
        target_seat=0,
        step_data=mc_data[0],
        opponent_models=opponent_models,
        target_network=target_net,
        device=device,
        true_alpha=true_alpha,
        true_beta=true_beta,
    )

    # Save ablation IRL results
    abl_estimates_path = os.path.join(IRL_DIR, "irl_estimates_ablation.json")
    summary = [{k: v for k, v in abl_result.items() if k not in ("alpha_history", "beta_history", "ll_history")}]
    with open(abl_estimates_path, "w") as f:
        json.dump(summary, f, indent=2)

    abl_conv_path = os.path.join(IRL_DIR, "irl_convergence_log_ablation.json")
    with open(abl_conv_path, "w") as f:
        json.dump([abl_result], f, indent=2)

    # ── Step 3: Evaluate ablation ─────────────────────────────────────────
    log.info("\nEvaluating ablation IRL ...")
    abl_metrics, abl_details = run_evaluation(is_ablation=True, ablation_tag="ablation")

    # ── Step 4: Load main experiment metrics for comparison ───────────────
    main_metrics_path = os.path.join(IRL_DIR, "evaluation_metrics.json")
    if not os.path.exists(main_metrics_path):
        log.warning("Main experiment metrics not found at %s.  Run step4 first.", main_metrics_path)
        main_metrics = {}
    else:
        with open(main_metrics_path) as f:
            main_metrics = json.load(f)

    # ── Step 5: Comparison report ─────────────────────────────────────────
    comparison = {
        "main_experiment":  {
            "mean_alpha_mse":  main_metrics.get("mean_alpha_mse"),
            "mean_beta_mse":   main_metrics.get("mean_beta_mse"),
            "mean_joint_mse":  main_metrics.get("mean_joint_mse"),
            "mean_holl_estimated": main_metrics.get("mean_holl_estimated"),
            "mean_holl_recovery_pct": main_metrics.get("mean_holl_recovery_pct"),
            "description": "4 co-adapted agents, IRL on each individually",
        },
        "ablation": {
            "mean_alpha_mse":  abl_metrics.get("mean_alpha_mse"),
            "mean_beta_mse":   abl_metrics.get("mean_beta_mse"),
            "mean_joint_mse":  abl_metrics.get("mean_joint_mse"),
            "mean_holl_estimated": abl_metrics.get("mean_holl_estimated"),
            "mean_holl_recovery_pct": abl_metrics.get("mean_holl_recovery_pct"),
            "description": "1 adaptive agent vs 3 fixed neutral opponents, IRL on seat 0",
        },
        "ratio_joint_mse": (
            abl_metrics.get("mean_joint_mse", float("nan"))
            / max(main_metrics.get("mean_joint_mse", 1e-9), 1e-9)
        ),
    }

    # Interpretation
    ratio = comparison["ratio_joint_mse"]
    if ratio < 0.9:
        interpretation = (
            "Ablation MSE < Main MSE: Co-adaptation HURTS IRL recovery. "
            "Single-agent fixed-opponent setting is an easier IRL problem."
        )
    elif ratio > 1.1:
        interpretation = (
            "Ablation MSE > Main MSE: Co-adaptation HELPS IRL recovery (or neutral). "
            "Multi-agent opponent modelling successfully handles co-adaptation."
        )
    else:
        interpretation = (
            "Ablation MSE ≈ Main MSE: Co-adaptation has minimal impact on IRL recovery. "
            "The single-agent assumption is a reasonable approximation."
        )
    comparison["interpretation"] = interpretation

    comp_path = os.path.join(IRL_DIR, "ablation_comparison.json")
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)

    # ── Print report ───────────────────────────────────────────────────────
    log.info("\n" + "="*70)
    log.info("ABLATION COMPARISON REPORT")
    log.info("="*70)
    log.info("")
    log.info("  %-35s  %12s  %12s", "Metric", "Main", "Ablation")
    log.info("  " + "-"*63)

    for metric_key, label in [
        ("mean_alpha_mse",         "Mean α MSE         "),
        ("mean_beta_mse",          "Mean β MSE         "),
        ("mean_joint_mse",         "Mean Joint MSE     "),
        ("mean_holl_estimated",    "Mean HOLL (est)    "),
        ("mean_holl_recovery_pct", "HOLL Recovery (%)  "),
    ]:
        main_val = main_metrics.get(metric_key, float("nan"))
        abl_val  = abl_metrics.get(metric_key, float("nan"))
        log.info("  %-35s  %12.4f  %12.4f", label, main_val or float("nan"), abl_val or float("nan"))

    log.info("")
    log.info("  Ablation/Main joint MSE ratio: %.3f", ratio)
    log.info("")
    log.info("  INTERPRETATION:")
    log.info("  %s", interpretation)
    log.info("")
    log.info("Saved: %s", comp_path)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_ablation_comparison()
