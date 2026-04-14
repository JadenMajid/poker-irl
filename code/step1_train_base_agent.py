"""
step1_train_base_agent.py
-------------------------
Train a single shared-parameter poker policy to approximate Nash-equilibrium
play using self-play PPO with parameter sharing across all 4 seats.

Why parameter sharing?
  With parameter sharing, all 4 seats use the same network weights to make
  decisions.  Every hand provides 4x the gradient signal compared to training
  a single-seat agent.  More importantly, the converged fixed point of this
  process is exactly a symmetric Nash equilibrium: each seat is best-responding
  to the same policy (itself), so no seat has an incentive to deviate.  This
  gives us a principled "neutral" base policy rather than an arbitrary one.

Training loop structure:
  1. Run N_STEPS_PER_UPDATE hands, recording every (obs, action, reward, done)
     for all 4 seats into a shared rollout buffer.
  2. Run PPO update epochs on the shared buffer.
  3. Check convergence via policy KL divergence snapshots.
  4. Log progress every LOG_EVERY hands.
  5. Terminate on convergence or MAX_HANDS.
  6. Save the base agent checkpoint.

Output files:
  checkpoints/base_agent.pt          — final base agent weights
  checkpoints/base_agent_config.json — training config and metadata
  checkpoints/base_training_log.json — per-update training statistics
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch

# ── local imports ─────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import (
    ActorCriticNetwork,
    NUM_ACTIONS,
    action_to_index,
    index_to_action,
    legal_action_mask,
)
from feature_encoder import FeatureEncoder, FEATURE_DIM
from game_state import Action, ActionType, NUM_PLAYERS, PlayerObservation
from poker_env import PokerEnv
from ppo_trainer import (
    PPOConfig,
    PPOTrainer,
    ConvergenceDetector,
    FeatureSampleStore,
    RolloutBuffer,
)
from reward import NeutralRewardParams, RewardFunction

# ── configuration ──────────────────────────────────────────────────────────

CHECKPOINT_DIR   = "checkpoints"
LOG_EVERY        = 500       # hands between progress logs
SAVE_EVERY       = 10_000    # hands between intermediate saves
MAX_HANDS        = 2_000_000 # hard ceiling (~1–2 days on CPU)
DEVICE           = "cpu"     # change to "cuda" if available
HIDDEN_DIM       = 256

# PPO hyper-parameters (tuned for poker self-play)
PPO_CFG = PPOConfig(
    n_steps_per_update=4096,
    n_epochs=10,
    mini_batch_size=256,
    clip_range=0.2,
    value_clip_range=0.2,
    value_coef=0.5,
    entropy_coef=0.01,     # keep exploration; poker has high stochasticity
    kl_coef=0.0,           # no KL penalty during base training
    gae_lambda=0.95,
    gamma=1.0,             # no discounting — episodic cash-game
    learning_rate=3e-4,
    max_grad_norm=0.5,
    convergence_window=2000,
    convergence_threshold=3e-4,
    min_hands_before_convergence_check=20_000,
    use_lr_schedule=True,
    lr_schedule_T_max=1_500_000,
)

# Convergence detector settings
CONV_WINDOW     = 2000
CONV_THRESHOLD  = 3e-4
CONV_MIN_HANDS  = 20_000
CONV_CHECK_EVERY= 1000

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared-policy agent wrapper
# ---------------------------------------------------------------------------

class SharedPolicyAgent:
    """
    All 4 seats share this single network.
    Records transitions for the shared PPO buffer.
    """

    def __init__(self, network: ActorCriticNetwork, device: torch.device) -> None:
        self.network = network
        self.device  = device
        self.encoder = FeatureEncoder()
        self.network.eval()

    def act(self, obs: PlayerObservation) -> Action:
        feat   = self.encoder.encode(obs)
        mask   = legal_action_mask(obs)
        feat_t = torch.tensor(feat, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = mask.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, _ = self.network(feat_t, mask_t)
            from torch.distributions import Categorical
            dist      = Categorical(logits=logits.squeeze(0))
            idx       = int(dist.sample().item())

        return index_to_action(idx, obs.observing_seat)

    def act_and_record(
        self,
        obs:     PlayerObservation,
        buffer:  RolloutBuffer,
        reward:  float,
        done:    bool,
    ) -> Action:
        """Act and immediately push the transition into the shared buffer."""
        feat   = self.encoder.encode(obs)
        mask   = legal_action_mask(obs)
        feat_t = torch.tensor(feat, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = mask.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, value = self.network(feat_t, mask_t)
            from torch.distributions import Categorical
            dist          = Categorical(logits=logits.squeeze(0))
            idx_t         = dist.sample()
            log_prob      = dist.log_prob(idx_t)
            idx           = int(idx_t.item())

        buffer.add(
            feature=feat,
            mask=mask.numpy(),
            action=idx,
            log_prob=float(log_prob.item()),
            value=float(value.squeeze().item()),
            reward=reward,
            done=done,
        )
        return index_to_action(idx, obs.observing_seat)


# ---------------------------------------------------------------------------
# Training loop helpers
# ---------------------------------------------------------------------------

def _make_callbacks(
    agent:         SharedPolicyAgent,
    buffer:        RolloutBuffer,
    pending_steps: Dict,   # seat → (obs, action_obj, feat) awaiting reward
    reward_fns:    List[RewardFunction],
):
    """
    Build 4 poker callbacks that record transitions.

    The poker environment calls callback(obs) → Action synchronously.
    Rewards are only known at hand end, so we use a deferred pattern:
      - On call: record the observation/action with reward=0 temporarily.
      - At hand end: back-fill the true reward via post-processing.

    This is handled by the training loop (see run_training_loop) which
    calls env.play_hand() and then back-fills.
    """
    # We can't back-fill mid-hand in the current env design, so instead
    # we collect (obs, action) pairs per seat and push them all at hand end
    # with the correct terminal reward.  The intermediate steps get reward=0
    # and done=False; only the last step per seat gets the true reward and done=True.

    seat_histories: Dict[int, List] = {i: [] for i in range(NUM_PLAYERS)}

    def make_cb(seat: int):
        def callback(obs: PlayerObservation) -> Action:
            feat   = agent.encoder.encode(obs)
            mask   = legal_action_mask(obs)
            feat_t = torch.tensor(feat, dtype=torch.float32, device=agent.device).unsqueeze(0)
            mask_t = mask.unsqueeze(0).to(agent.device)

            with torch.no_grad():
                logits, value = agent.network(feat_t, mask_t)
                from torch.distributions import Categorical
                dist          = Categorical(logits=logits.squeeze(0))
                idx_t         = dist.sample()
                log_prob      = dist.log_prob(idx_t)
                idx           = int(idx_t.item())
                val           = float(value.squeeze().item())

            action = index_to_action(idx, seat)
            seat_histories[seat].append((feat, mask.numpy(), idx, float(log_prob.item()), val))
            return action
        return callback

    callbacks = [make_cb(i) for i in range(NUM_PLAYERS)]
    return callbacks, seat_histories


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def run_training() -> None:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device(DEVICE)

    log.info("Initialising shared-parameter poker network ...")
    network = ActorCriticNetwork(input_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM).to(device)
    network.eval()

    agent    = SharedPolicyAgent(network, device)
    trainer  = PPOTrainer(network, PPO_CFG, device)
    detector = ConvergenceDetector(
        window=CONV_WINDOW,
        threshold=CONV_THRESHOLD,
        min_hands=CONV_MIN_HANDS,
        check_every=CONV_CHECK_EVERY,
    )
    feat_store = FeatureSampleStore(capacity=1024)
    encoder    = FeatureEncoder()

    # Neutral reward functions (alpha=0, beta=0 → just chip delta)
    reward_fns = [RewardFunction(NeutralRewardParams) for _ in range(NUM_PLAYERS)]

    training_log = []
    hand_count   = 0
    update_count = 0
    start_time   = time.time()

    log.info("Starting self-play training.  MAX_HANDS=%d", MAX_HANDS)

    while hand_count < MAX_HANDS:
        # ── Collect rollout ───────────────────────────────────────────────
        # We collect n_steps_per_update transitions from all 4 seats.
        # Each hand generates ~6–12 decision points (1–3 per player per street).

        shared_buffer  = trainer.buffer
        shared_buffer.clear()
        seat_histories: Dict[int, List] = {i: [] for i in range(NUM_PLAYERS)}
        hands_this_rollout = 0

        while len(shared_buffer) < PPO_CFG.n_steps_per_update:
            # Build per-hand callbacks
            this_hand_histories: Dict[int, List] = {i: [] for i in range(NUM_PLAYERS)}

            def make_cb(seat: int):
                def callback(obs: PlayerObservation) -> Action:
                    feat   = encoder.encode(obs)
                    mask   = legal_action_mask(obs)
                    feat_t = torch.tensor(feat, dtype=torch.float32, device=device).unsqueeze(0)
                    mask_t = mask.unsqueeze(0).to(device)

                    with torch.no_grad():
                        logits, value = network(feat_t, mask_t)
                        from torch.distributions import Categorical
                        dist          = Categorical(logits=logits.squeeze(0))
                        idx_t         = dist.sample()
                        lp            = dist.log_prob(idx_t)
                        idx           = int(idx_t.item())
                        val           = float(value.squeeze().item())

                    action = index_to_action(idx, seat)
                    this_hand_histories[seat].append(
                        (feat, mask.numpy(), idx, float(lp.item()), val)
                    )
                    feat_store.add(feat)
                    return action
                return callback

            env = PokerEnv([make_cb(i) for i in range(NUM_PLAYERS)], record_trajectories=True)
            traj = env.play_hand()
            hand_count        += 1
            hands_this_rollout += 1

            # Back-fill rewards: terminal reward for each seat's LAST step,
            # zero for all intermediate steps.
            for seat in range(NUM_PLAYERS):
                chip_delta = float(traj.final_chip_deltas.get(seat, 0))
                rf_components = reward_fns[seat].compute(traj, seat)
                terminal_reward = rf_components.total   # neutral: == chip_delta

                steps = this_hand_histories[seat]
                for k, (feat, mask_np, idx, lp, val) in enumerate(steps):
                    is_last = (k == len(steps) - 1)
                    shared_buffer.add(
                        feature=feat,
                        mask=mask_np,
                        action=idx,
                        log_prob=lp,
                        value=val,
                        reward=terminal_reward if is_last else 0.0,
                        done=is_last,
                    )

            # Convergence check
            sample_feat = feat_store.sample_tensor(256, device)
            converged   = detector.on_hand_end(network, sample_feat, device)

            # Progress log
            if hand_count % LOG_EVERY == 0:
                elapsed  = time.time() - start_time
                hands_ph = hand_count / max(elapsed, 1) * 3600
                mean_kl  = detector.latest_mean_kl()
                log.info(
                    "Hand %7d | Updates %4d | Mean KL %.5f | %.0f hands/hr",
                    hand_count, update_count, mean_kl, hands_ph,
                )

            if converged:
                log.info("Convergence detected at hand %d (mean KL=%.5f < threshold %.5f).",
                         hand_count, detector.latest_mean_kl(), CONV_THRESHOLD)
                break

        # ── PPO update ────────────────────────────────────────────────────
        if len(shared_buffer) >= PPO_CFG.mini_batch_size:
            network.train()
            stats = trainer.update()
            network.eval()
            update_count += 1

            training_log.append({
                "hand":         hand_count,
                "update":       update_count,
                "policy_loss":  stats["policy_loss"],
                "value_loss":   stats["value_loss"],
                "entropy":      stats["entropy"],
                "clip_frac":    stats["clip_frac"],
                "mean_kl":      detector.latest_mean_kl(),
            })

            if update_count % 10 == 0:
                log.info(
                    "  → Update %4d | π-loss %.4f | V-loss %.4f | entropy %.4f | clip %.3f",
                    update_count,
                    stats["policy_loss"],
                    stats["value_loss"],
                    stats["entropy"],
                    stats["clip_frac"],
                )

        # Intermediate save
        if hand_count % SAVE_EVERY == 0:
            _save_checkpoint(
                network, hand_count, CHECKPOINT_DIR, "base_agent_ckpt",
                detector.latest_mean_kl(), stats["policy_loss"], stats["value_loss"], stats["entropy"]
            )

        if detector.converged or hand_count >= MAX_HANDS:
            break

    # ── Final save ────────────────────────────────────────────────────────
    log.info("Training complete.  Total hands: %d  Updates: %d", hand_count, update_count)
    _save_checkpoint(
        network, hand_count, CHECKPOINT_DIR, "base_agent",
        detector.latest_mean_kl(), stats["policy_loss"], stats["value_loss"], stats["entropy"]
    )

    config_meta = {
        "hand_count":    hand_count,
        "update_count":  update_count,
        "converged":     detector.converged,
        "final_mean_kl": detector.latest_mean_kl(),
        "hidden_dim":    HIDDEN_DIM,
        "feature_dim":   FEATURE_DIM,
        "device":        DEVICE,
        "ppo_config":    PPO_CFG.__dict__,
    }
    with open(os.path.join(CHECKPOINT_DIR, "base_agent_config.json"), "w") as f:
        json.dump(config_meta, f, indent=2)

    with open(os.path.join(CHECKPOINT_DIR, "base_training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)

    log.info("Saved:  checkpoints/base_agent.pt")
    log.info("Saved:  checkpoints/base_agent_config.json")
    log.info("Saved:  checkpoints/base_training_log.json")


def _save_checkpoint(
    network:      ActorCriticNetwork,
    step:         int,
    out_dir:      str,
    name:         str,
    mean_kl:      float,
    policy_loss:  float,
    value_loss:   float,
    entropy:      float,
) -> None:
    path = os.path.join(out_dir, f"{name}.pt")
    torch.save({
        "network_state": network.state_dict(),
        "hidden_dim":    network.hidden_dim,
        "feature_dim":   FEATURE_DIM,
        "num_actions":   NUM_ACTIONS,
        "step":          step,
        "mean_kl":       mean_kl,
        "policy_loss":   policy_loss,
        "value_loss":    value_loss,
        "entropy":       entropy,
    }, path)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_training()
