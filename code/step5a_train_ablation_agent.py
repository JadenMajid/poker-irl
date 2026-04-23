"""
step5a_train_ablation_agent.py
-------------------------------
Ablation study — Phase 1: Training

Setup:
  - Load base agent weights into all 4 agents.
  - Agents 1, 2, 3: frozen at the base (neutral) policy — they NEVER update.
  - Agent 0: assigned a POSITIVE (alpha, beta) reward function and allowed
    to fine-tune via PPO + KL regularisation against the base.

This contrasts with the main experiment where ALL 4 agents adapted
simultaneously to their own reward functions.  The ablation tests whether
the co-adaptation assumption is necessary: can we recover reward parameters
just as well when only one agent adapted (to fixed opponents) versus when all
four co-adapted to each other?

By comparing:
  - Main IRL MSE  (all 4 agents co-adapted → IRL on each individually)
  - Ablation IRL MSE (1 agent adapted to fixed opponents → IRL on that 1 agent)

We gain insight into whether the co-adaptation assumption hurts or helps
the IRL recovery, and whether treating co-players as static environment
dynamics is a reasonable simplification.

Reward for the ablation agent:
  We use (alpha=+0.004, beta=+0.25) — the same as Seat 0 in the main experiment
  — so the comparison is apples-to-apples.

Mini-batching:
  Rather than triggering a PPO update whenever the rollout buffer crosses
  n_steps_per_update transitions, we accumulate HANDS_PER_MINI_BATCH complete
  hands before each update.  Rewards are normalised across the full batch of
  hands (zero-mean, unit-variance) before being written into the rollout
  buffer, which substantially reduces gradient noise and gives the optimiser
  a smoother loss landscape.  The larger effective batch also improves GPU/CPU
  utilisation when the environment is the bottleneck.

Output files:
  checkpoints/ablation_perturbed_agent_0.pt    — fine-tuned ablation agent
  checkpoints/ablation_agent_params.json        — (alpha, beta) record
  checkpoints/ablation_training_log.json        — training stats
"""

from __future__ import annotations

import copy
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import (
    ActorCriticNetwork,
    NUM_ACTIONS,
    index_to_action,
    legal_action_mask,
    action_to_index,
)
from feature_encoder import FeatureEncoder, FEATURE_DIM
from game_state import NUM_PLAYERS, PlayerObservation, Action
from poker_env import PokerEnv
from ppo_trainer import (
    PPOConfig,
    PPOTrainer,
    ConvergenceDetector,
    FeatureSampleStore,
    RolloutBuffer,
)
from reward import RewardParams, RewardFunction

# ── configuration ──────────────────────────────────────────────────────────

CHECKPOINT_DIR = "checkpoints"
DEVICE         = "cpu"
HIDDEN_DIM     = 256
LOG_EVERY      = 500
SAVE_EVERY     = 10_000
MAX_HANDS      = 1_000_000

# Number of complete hands to accumulate before each PPO update.
# Larger values → lower gradient variance, smoother trendlines, but less
# frequent parameter updates.  A value of 16–32 hands is a good default for
# a 4-player poker environment where hands average ~6 decision points each,
# giving ~100–200 transitions per update — comparable to the original
# n_steps_per_update=2048 but now aligned to hand boundaries.
HANDS_PER_MINI_BATCH: int = 24

# When True, rewards within each mini-batch are standardised to zero mean /
# unit variance before being stored in the rollout buffer.  This is the
# primary mechanism for noise reduction: it prevents a single high-variance
# hand (e.g. a big pot) from dominating the gradient signal.
NORMALISE_BATCH_REWARDS: bool = True

# Floor standard-deviation used during reward normalisation to avoid
# division-by-zero on degenerate batches where all rewards are identical.
REWARD_NORM_EPS: float = 1e-8

# Ablation agent reward parameters (matching Seat 0 of main experiment)
ABLATION_REWARD_PARAMS = RewardParams(alpha=+0.004, beta=+0.25)

# Fine-tuning PPO config (same as step2)
ABLATION_PPO_CFG = PPOConfig(
    n_steps_per_update=2048,
    n_epochs=8,
    mini_batch_size=128,
    clip_range=0.15,
    value_clip_range=0.15,
    value_coef=0.5,
    entropy_coef=0.005,
    kl_coef=0.05,
    gae_lambda=0.95,
    gamma=1.0,
    learning_rate=1e-4,
    max_grad_norm=0.4,
    convergence_window=2000,
    convergence_threshold=2e-4,
    min_hands_before_convergence_check=15_000,
    use_lr_schedule=True,
    lr_schedule_T_max=900_000,
)

KL_ANNEAL_FACTOR = 0.9995
KL_FLOOR         = 0.005

CONV_THRESHOLD   = 2e-4
CONV_MIN_HANDS   = 15_000
CONV_CHECK_EVERY = 1000
CONV_WINDOW      = 2000

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pending hand: holds one complete hand's transitions before batch commit
# ---------------------------------------------------------------------------

@dataclass
class PendingHand:
    """
    Accumulates all (feature, mask, action, log_prob, value) tuples for a
    single hand, along with the terminal reward.  Kept separate from the
    RolloutBuffer so that batch-level reward normalisation can be applied
    across all pending hands before anything is written to the buffer.
    """
    transitions: List[Tuple] = field(default_factory=list)
    terminal_reward: float   = 0.0

    def add_transition(
        self,
        feat:    np.ndarray,
        mask:    np.ndarray,
        action:  int,
        log_prob: float,
        value:   float,
    ) -> None:
        self.transitions.append((feat, mask, action, log_prob, value))

    def __len__(self) -> int:
        return len(self.transitions)


# ---------------------------------------------------------------------------
# Fixed-policy agent (frozen neutral opponents)
# ---------------------------------------------------------------------------

class FixedAgent:
    """Deterministic greedy or sampled agent with frozen weights."""

    def __init__(self, seat: int, network: ActorCriticNetwork, device: torch.device) -> None:
        self.seat    = seat
        self.network = network
        self.device  = device
        self.encoder = FeatureEncoder()

    def act(self, obs: PlayerObservation) -> Action:
        feat   = self.encoder.encode(obs)
        mask   = legal_action_mask(obs)
        feat_t = torch.tensor(feat, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = mask.unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self.network(feat_t, mask_t)
            dist      = Categorical(logits=logits.squeeze(0))
            idx       = int(dist.sample().item())
        return index_to_action(idx, self.seat)


# ---------------------------------------------------------------------------
# Training agent (single adaptive agent, mini-batch aware)
# ---------------------------------------------------------------------------

class AdaptiveAgent:
    """
    Single fine-tuned agent with mini-batch hand accumulation.

    Instead of writing each hand directly into the RolloutBuffer and
    triggering an update whenever the buffer is full, this agent accumulates
    complete hands in a local ``pending_hands`` list.  Once
    ``HANDS_PER_MINI_BATCH`` hands have been collected the agent:

      1. Optionally normalises rewards across the batch (zero-mean /
         unit-variance), smoothing the gradient signal.
      2. Writes all transitions into the RolloutBuffer in one pass.
      3. Triggers a PPO update if the buffer now meets the step threshold.

    This decouples update frequency from individual hand length, eliminates
    the per-hand reward spike problem, and makes it straightforward to
    parallelise hand generation in future (each worker returns a PendingHand).
    """

    def __init__(
        self,
        seat:        int,
        network:     ActorCriticNetwork,
        ref_network: ActorCriticNetwork,
        reward_fn:   RewardFunction,
        cfg:         PPOConfig,
        device:      torch.device,
    ) -> None:
        self.seat       = seat
        self.network    = network
        self.reward_fn  = reward_fn
        self.device     = device
        self.encoder    = FeatureEncoder()
        self.trainer    = PPOTrainer(network, cfg, device, ref_network=ref_network)
        self.detector   = ConvergenceDetector(
            window=CONV_WINDOW,
            threshold=CONV_THRESHOLD,
            min_hands=CONV_MIN_HANDS,
            check_every=CONV_CHECK_EVERY,
        )
        self.feat_store = FeatureSampleStore(capacity=1024)

        # Mini-batch accumulator: list of complete PendingHand objects
        self._pending_hands: List[PendingHand] = []
        # The hand currently being played
        self._current_hand:  Optional[PendingHand] = None

    # ------------------------------------------------------------------
    # Hand lifecycle
    # ------------------------------------------------------------------

    def begin_hand(self) -> None:
        """Called once at the start of each hand."""
        self._current_hand = PendingHand()

    def act(self, obs: PlayerObservation) -> Action:
        """Select an action and record the transition in the current hand."""
        feat   = self.encoder.encode(obs)
        mask   = legal_action_mask(obs)
        feat_t = torch.tensor(feat, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = mask.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, value = self.network(feat_t, mask_t)
            dist          = Categorical(logits=logits.squeeze(0))
            idx_t         = dist.sample()
            lp            = dist.log_prob(idx_t)
            idx           = int(idx_t.item())
            val           = float(value.squeeze().item())

        self.feat_store.add(feat)
        self._current_hand.add_transition(feat, mask.numpy(), idx, float(lp.item()), val)
        return index_to_action(idx, self.seat)

    def on_hand_end(self, terminal_reward: float) -> bool:
        """
        Finalise the current hand, append it to the pending batch, and
        — once the batch is full — commit all hands to the rollout buffer
        with normalised rewards.

        Returns True if the convergence detector has fired.
        """
        self._current_hand.terminal_reward = terminal_reward
        self._pending_hands.append(self._current_hand)
        self._current_hand = None

        converged = False

        if len(self._pending_hands) >= HANDS_PER_MINI_BATCH:
            converged = self._commit_batch()

        return converged

    # ------------------------------------------------------------------
    # Batch commit
    # ------------------------------------------------------------------

    def _commit_batch(self) -> bool:
        """
        Normalise rewards across the accumulated batch of hands, write all
        transitions into the RolloutBuffer, and update the convergence
        detector.

        Returns True if the convergence detector has fired.
        """
        hands = self._pending_hands
        self._pending_hands = []

        # ── Step 1: collect raw terminal rewards ──────────────────────
        raw_rewards = np.array(
            [h.terminal_reward for h in hands], dtype=np.float32
        )

        # ── Step 2: optionally normalise across the batch ─────────────
        if NORMALISE_BATCH_REWARDS:
            r_mean = raw_rewards.mean()
            r_std  = raw_rewards.std()
            normed_rewards = (raw_rewards - r_mean) / (r_std + REWARD_NORM_EPS)
        else:
            normed_rewards = raw_rewards

        # ── Step 3: write transitions into the rollout buffer ─────────
        buf = self.trainer.buffer
        for hand, reward in zip(hands, normed_rewards):
            n = len(hand.transitions)
            for k, (feat, mask_np, idx, lp, val) in enumerate(hand.transitions):
                is_last = (k == n - 1)
                buf.add(
                    feature  = feat,
                    mask     = mask_np,
                    action   = idx,
                    log_prob = lp,
                    value    = val,
                    reward   = float(reward) if is_last else 0.0,
                    done     = is_last,
                )

        # ── Step 4: update convergence detector once per batch ────────
        sample    = self.feat_store.sample_tensor(256, self.device)
        converged = self.detector.on_hand_end(
            self.network, sample, self.device, num_hands=len(hands)
        )

        return converged

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def maybe_update(self) -> Optional[Dict]:
        """Trigger a PPO update if the buffer has accumulated enough steps."""
        if len(self.trainer.buffer) >= ABLATION_PPO_CFG.n_steps_per_update:
            self.network.train()
            stats = self.trainer.update()
            self.network.eval()
            return stats
        return None

    def anneal_kl(self) -> None:
        self.trainer.cfg.kl_coef = max(
            KL_FLOOR, self.trainer.cfg.kl_coef * KL_ANNEAL_FACTOR
        )

    def pending_batch_size(self) -> int:
        """Number of hands currently waiting in the accumulator."""
        return len(self._pending_hands)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_ablation_training() -> None:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device(DEVICE)

    # ── Load base agent ────────────────────────────────────────────────────
    base_path = os.path.join(CHECKPOINT_DIR, "base_agent.pt")
    if not os.path.exists(base_path):
        log.error("Base agent not found at %s.  Run step1 first.", base_path)
        sys.exit(1)

    log.info("Loading base agent from %s ...", base_path)
    ckpt = torch.load(base_path, map_location=device)

    def load_network() -> ActorCriticNetwork:
        net = ActorCriticNetwork(
            input_dim=ckpt.get("feature_dim", FEATURE_DIM),
            hidden_dim=ckpt.get("hidden_dim",  HIDDEN_DIM),
        ).to(device)
        net.load_state_dict(ckpt["network_state"])
        return net

    base_network = load_network()
    base_network.eval()
    for p in base_network.parameters():
        p.requires_grad_(False)

    # ── Build fixed opponents (seats 1, 2, 3) ─────────────────────────────
    fixed_agents = {
        seat: FixedAgent(seat, load_network(), device)
        for seat in [1, 2, 3]
    }
    for fa in fixed_agents.values():
        for p in fa.network.parameters():
            p.requires_grad_(False)
        fa.network.eval()

    log.info("Fixed opponents (seats 1, 2, 3): frozen neutral base policy.")

    # ── Build adaptive agent (seat 0) ─────────────────────────────────────
    cfg    = copy.deepcopy(ABLATION_PPO_CFG)
    rf     = RewardFunction(ABLATION_REWARD_PARAMS, variance_window=200)
    agent0 = AdaptiveAgent(0, load_network(), base_network, rf, cfg, device)
    agent0.network.eval()

    log.info(
        "Adaptive agent (seat 0): α=%.4f, β=%.4f  |  "
        "mini-batch=%d hands, reward normalisation=%s",
        ABLATION_REWARD_PARAMS.alpha,
        ABLATION_REWARD_PARAMS.beta,
        HANDS_PER_MINI_BATCH,
        NORMALISE_BATCH_REWARDS,
    )

    # ── Training loop ──────────────────────────────────────────────────────
    training_log  = []
    hand_count    = 0
    update_count  = 0
    batch_count   = 0
    start_time    = time.time()

    # Running statistics for logging reward signal health
    reward_history: List[float] = []

    log.info(
        "Starting ablation training.  MAX_HANDS=%d  HANDS_PER_MINI_BATCH=%d",
        MAX_HANDS, HANDS_PER_MINI_BATCH,
    )

    converged = False

    while hand_count < MAX_HANDS and not converged:
        agent0.begin_hand()

        def cb0(obs: PlayerObservation) -> Action:
            return agent0.act(obs)

        callbacks = [
            cb0,
            fixed_agents[1].act,
            fixed_agents[2].act,
            fixed_agents[3].act,
        ]

        env  = PokerEnv(callbacks, record_trajectories=True)
        traj = env.play_hand()
        hand_count += 1

        reward_components = agent0.reward_fn.compute(traj, 0)
        terminal_reward   = reward_components.total
        reward_history.append(terminal_reward)

        # on_hand_end accumulates into the batch; commits when batch is full
        converged = agent0.on_hand_end(terminal_reward)
        if converged:
            # Batch was just committed inside on_hand_end; count it
            batch_count += 1

        agent0.anneal_kl()

        # Track when a batch commit occurred (pending list was just cleared)
        batch_just_committed = (
            agent0.pending_batch_size() == 0
            and hand_count % HANDS_PER_MINI_BATCH == 0
        )
        if batch_just_committed:
            batch_count += 1

        # Trigger PPO update if the rollout buffer is now sufficiently full
        stats = agent0.maybe_update()
        if stats is not None:
            update_count += 1

            # Reward signal stats over the hands that fed this update
            recent_rewards = reward_history[-HANDS_PER_MINI_BATCH * 4:]
            r_arr   = np.array(recent_rewards)
            r_mean  = float(r_arr.mean())
            r_std   = float(r_arr.std())

            entry = {
                "hand":         hand_count,
                "update":       update_count,
                "batch":        batch_count,
                "policy_loss":  stats["policy_loss"],
                "value_loss":   stats["value_loss"],
                "entropy":      stats["entropy"],
                "kl_penalty":   stats["kl_penalty"],
                "mean_kl":      agent0.detector.latest_mean_kl(),
                "reward_mean":  r_mean,
                "reward_std":   r_std,
            }
            training_log.append(entry)

            if update_count % 10 == 0:
                log.info(
                    "  Update %3d | π-loss %.4f | entropy %.4f | "
                    "KL-pen %.4f | r̄=%.3f σ=%.3f",
                    update_count,
                    stats["policy_loss"],
                    stats["entropy"],
                    stats["kl_penalty"],
                    r_mean,
                    r_std,
                )

        if hand_count % LOG_EVERY == 0:
            elapsed  = time.time() - start_time
            hands_ph = hand_count / max(elapsed, 1) * 3600
            log.info(
                "Hand %7d | batch %4d | ConvergeKL %.5f | %.0f hands/hr",
                hand_count,
                batch_count,
                agent0.detector.latest_mean_kl(),
                hands_ph,
            )

        if hand_count % SAVE_EVERY == 0:
            _save_agent(agent0.network, hand_count)

    # ── Final saves ────────────────────────────────────────────────────────
    if converged:
        log.info("Ablation agent converged at hand %d (KL=%.5f).",
                 hand_count, agent0.detector.latest_mean_kl())
    log.info("Ablation training complete.  Total hands: %d", hand_count)

    _save_agent(agent0.network, hand_count, final=True)

    params_record = [
        {"seat": 0, "alpha": ABLATION_REWARD_PARAMS.alpha, "beta": ABLATION_REWARD_PARAMS.beta}
    ]
    for s in [1, 2, 3]:
        params_record.append({"seat": s, "alpha": 0.0, "beta": 0.0})

    with open(os.path.join(CHECKPOINT_DIR, "ablation_agent_params.json"), "w") as f:
        json.dump(params_record, f, indent=2)

    with open(os.path.join(CHECKPOINT_DIR, "ablation_training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)

    log.info("Saved: checkpoints/ablation_perturbed_agent_0.pt")
    log.info("Saved: checkpoints/ablation_agent_params.json")
    log.info("Saved: checkpoints/ablation_training_log.json")


def _save_agent(network: ActorCriticNetwork, step: int, final: bool = False) -> None:
    name = "ablation_perturbed_agent_0.pt" if final else "ablation_ckpt.pt"
    path = os.path.join(CHECKPOINT_DIR, name)
    torch.save({
        "network_state":       network.state_dict(),
        "hidden_dim":          network.hidden_dim,
        "feature_dim":         FEATURE_DIM,
        "num_actions":         NUM_ACTIONS,
        "seat":                0,
        "step":                step,
        "alpha":               ABLATION_REWARD_PARAMS.alpha,
        "beta":                ABLATION_REWARD_PARAMS.beta,
        "hands_per_mini_batch": HANDS_PER_MINI_BATCH,
        "normalise_rewards":   NORMALISE_BATCH_REWARDS,
    }, path)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_ablation_training()
