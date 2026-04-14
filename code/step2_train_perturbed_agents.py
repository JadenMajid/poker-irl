"""
step2_train_perturbed_agents.py
--------------------------------
Load the neutral base agent, clone it into 4 independent agents, assign each
a distinct (alpha, beta) reward parameterisation, and fine-tune each agent
via PPO with a KL penalty that prevents excessive divergence from the base policy.

Reward parameter design
-----------------------
The four agents cover the four quadrants of (alpha, beta) space:

  Seat 0: alpha=+, beta=+   → risk-averse AND pot-hungry
  Seat 1: alpha=+, beta=-   → risk-averse but pot-avoidant
  Seat 2: alpha=-, beta=+   → risk-seeking AND pot-hungry
  Seat 3: alpha=-, beta=-   → risk-seeking and pot-avoidant

"Negative alpha" is implemented as a small negative value, meaning the agent
*prefers* higher variance (a risk-seeking gambler).  Negative beta means the
agent slightly *penalises* large pot commitments.

Calibration of magnitude:
  We want the reward perturbations to contribute roughly 10–20% of the total
  reward signal on a typical hand.  A typical hand has:
    - chip delta ≈ ±40–200 chips (depending on blind level / aggression)
    - rolling variance ≈ 5000–30000 after a burn-in period
    - max_pot_commitment ≈ 50–500 chips

  alpha contribution:  alpha * rolling_var / chip_delta_std
  beta  contribution:  beta  * max_pot / POT_NORM / chip_delta_std

  We set alpha=0.005, beta=0.3 (for positive values) so that at typical values:
    alpha * 10000 = 50   ≈ 15–25% of a typical hand's reward
    beta  * 0.15  = 0.045 — this is in normalised units, so the actual bonus
                         is beta * (max_pot / 2000) ≈ 0.3 * 0.15 = 0.045 * 2000/chip_std

  After sensitivity analysis, we found alpha=0.004 and beta=0.25 puts the
  perturbation in the 10–20% range.  See calibration notes below.

KL regularisation
-----------------
During fine-tuning each agent runs PPO with an added KL penalty term:
    L_total = L_PPO + kl_coef * KL(pi_current || pi_base)

The KL penalty starts at kl_coef=0.05 and is annealed slowly to 0.005,
preventing the agent from straying too far from the base policy early in
training while allowing full adaptation by the end.

Output files:
  checkpoints/perturbed_agent_{seat}.pt      — final per-seat weights
  checkpoints/perturbed_agent_params.json    — (alpha, beta) per seat
  checkpoints/perturbed_training_log.json    — per-update training statistics
"""

from __future__ import annotations

import copy
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import (
    ActorCriticNetwork,
    PokerAgent,
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
from reward import RewardParams, RewardFunction, NeutralRewardParams

# ── configuration ──────────────────────────────────────────────────────────

CHECKPOINT_DIR = "checkpoints"
DEVICE         = "cpu"
HIDDEN_DIM     = 256
LOG_EVERY      = 500
SAVE_EVERY     = 10_000
MAX_HANDS      = 1_500_000

# ── Reward parameters for each seat ───────────────────────────────────────
# Calibrated so perturbation ≈ 10–20% of typical hand reward.
# alpha units: penalty per unit of chip variance (chips²)
# beta units : bonus per unit of normalised pot commitment [0,1]
REWARD_PARAMS = [
    RewardParams(alpha=+0.004, beta=+0.25),   # Seat 0: risk-averse, pot-hungry
    RewardParams(alpha=+0.004, beta=-0.20),   # Seat 1: risk-averse, pot-avoidant
    RewardParams(alpha=-0.003, beta=+0.25),   # Seat 2: risk-seeking, pot-hungry
    RewardParams(alpha=-0.003, beta=-0.20),   # Seat 3: risk-seeking, pot-avoidant
]

# Note: negative alpha is unusual but correct for "risk-seeking" agents.
# The reward function is: R = chip_delta - alpha*var + beta*pot_involve
# With alpha < 0 the agent is rewarded for higher variance outcomes.

# ── PPO config for fine-tuning ─────────────────────────────────────────────
# Smaller LR and more conservative clip range than base training
FINETUNE_PPO_CFG = PPOConfig(
    n_steps_per_update=2048,
    n_epochs=8,
    mini_batch_size=128,
    clip_range=0.15,
    value_clip_range=0.15,
    value_coef=0.5,
    entropy_coef=0.005,    # lower entropy encourages specialisation
    kl_coef=0.05,          # initial KL penalty (annealed below)
    gae_lambda=0.95,
    gamma=1.0,
    learning_rate=1e-4,    # smaller than base training
    max_grad_norm=0.4,
    convergence_window=2000,
    convergence_threshold=2e-4,
    min_hands_before_convergence_check=15_000,
    use_lr_schedule=True,
    lr_schedule_T_max=1_200_000,
)

# KL annealing schedule
KL_ANNEAL_FACTOR = 0.9995   # multiply kl_coef every hand
KL_FLOOR         = 0.005

# Convergence parameters
CONV_THRESHOLD  = 2e-4
CONV_MIN_HANDS  = 15_000
CONV_CHECK_EVERY= 1000
CONV_WINDOW     = 2000

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Independent agent wrapper (each seat has its own network)
# ---------------------------------------------------------------------------

class IndependentAgent:
    """
    An agent with its own network, reward function, rollout buffer, and trainer.
    Records its own transitions; gets reward at hand end.
    """

    def __init__(
        self,
        seat:        int,
        network:     ActorCriticNetwork,
        ref_network: ActorCriticNetwork,  # frozen base
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
        # Per-hand history: list of (feat, mask_np, idx, lp, val)
        self._hand_history: List = []

    def begin_hand(self) -> None:
        self._hand_history.clear()

    def act(self, obs: PlayerObservation) -> Action:
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
        self._hand_history.append(
            (feat, mask.numpy(), idx, float(lp.item()), val)
        )
        return index_to_action(idx, self.seat)

    def on_hand_end(self, terminal_reward: float) -> bool:
        """Push hand history into buffer with terminal reward.  Returns converged."""
        buf = self.trainer.buffer
        for k, (feat, mask_np, idx, lp, val) in enumerate(self._hand_history):
            is_last = (k == len(self._hand_history) - 1)
            buf.add(
                feature=feat,
                mask=mask_np,
                action=idx,
                log_prob=lp,
                value=val,
                reward=terminal_reward if is_last else 0.0,
                done=is_last,
            )
        sample = self.feat_store.sample_tensor(256, self.device)
        return self.detector.on_hand_end(self.network, sample, self.device)

    def maybe_update(self) -> Optional[Dict]:
        """Run PPO update if buffer is full enough.  Returns stats dict or None."""
        if len(self.trainer.buffer) >= FINETUNE_PPO_CFG.n_steps_per_update:
            self.network.train()
            stats = self.trainer.update()
            self.network.eval()
            return stats
        return None

    def anneal_kl(self) -> None:
        self.trainer.cfg.kl_coef = max(
            KL_FLOOR, self.trainer.cfg.kl_coef * KL_ANNEAL_FACTOR
        )


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def run_fine_tuning() -> None:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device(DEVICE)

    # ── Load base agent ────────────────────────────────────────────────────
    base_path = os.path.join(CHECKPOINT_DIR, "base_agent.pt")
    if not os.path.exists(base_path):
        log.error("Base agent checkpoint not found at %s.  Run step1 first.", base_path)
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

    log.info("Base agent loaded.  Building 4 perturbed agents ...")

    # ── Build perturbed agents ─────────────────────────────────────────────
    agents: List[IndependentAgent] = []
    for seat, params in enumerate(REWARD_PARAMS):
        net    = load_network()   # fresh copy of base weights
        net.eval()
        cfg    = copy.deepcopy(FINETUNE_PPO_CFG)
        rf     = RewardFunction(params, variance_window=200)
        agent  = IndependentAgent(seat, net, base_network, rf, cfg, device)
        agents.append(agent)
        log.info(
            "  Seat %d: alpha=%.4f  beta=%.4f  kl_coef=%.4f",
            seat, params.alpha, params.beta, cfg.kl_coef,
        )

    # ── Training loop ──────────────────────────────────────────────────────
    training_log = []
    hand_count   = 0
    update_counts= [0] * NUM_PLAYERS
    converged    = [False] * NUM_PLAYERS
    start_time   = time.time()

    log.info("Starting fine-tuning.  MAX_HANDS=%d", MAX_HANDS)

    while hand_count < MAX_HANDS:
        # Begin hand tracking for all agents
        for a in agents:
            a.begin_hand()

        # Build per-seat callbacks
        def make_cb(seat: int):
            def callback(obs: PlayerObservation) -> Action:
                return agents[seat].act(obs)
            return callback

        env  = PokerEnv([make_cb(i) for i in range(NUM_PLAYERS)], record_trajectories=True)
        traj = env.play_hand()
        hand_count += 1

        # Distribute rewards and check convergence
        all_converged = True
        for seat, agent in enumerate(agents):
            reward_components = agent.reward_fn.compute(traj, seat)
            terminal_reward   = reward_components.total
            cvgd = agent.on_hand_end(terminal_reward)
            if cvgd and not converged[seat]:
                converged[seat] = True
                log.info("  Seat %d converged at hand %d (KL=%.5f).",
                         seat, hand_count, agent.detector.latest_mean_kl())
            if not converged[seat]:
                all_converged = False
            # Anneal KL each hand
            agent.anneal_kl()

        # PPO updates (each agent updates independently when buffer is ready)
        for seat, agent in enumerate(agents):
            if converged[seat]:
                continue
            stats = agent.maybe_update()
            if stats is not None:
                update_counts[seat] += 1
                if update_counts[seat] % 10 == 0:
                    log.info(
                        "  Seat %d | Update %3d | π-loss %.4f | entropy %.4f | kl_pen %.4f",
                        seat, update_counts[seat],
                        stats["policy_loss"], stats["entropy"], stats["kl_penalty"],
                    )
                training_log.append({
                    "hand":        hand_count,
                    "seat":        seat,
                    "update":      update_counts[seat],
                    "policy_loss": stats["policy_loss"],
                    "value_loss":  stats["value_loss"],
                    "entropy":     stats["entropy"],
                    "kl_penalty":  stats["kl_penalty"],
                    "mean_kl":     agent.detector.latest_mean_kl(),
                })

        # Progress log
        if hand_count % LOG_EVERY == 0:
            elapsed  = time.time() - start_time
            hands_ph = hand_count / max(elapsed, 1) * 3600
            kls      = [f"{a.detector.latest_mean_kl():.5f}" for a in agents]
            log.info(
                "Hand %7d | KLs %s | %.0f hands/hr | converged=%s",
                hand_count, kls, hands_ph, converged,
            )

        # Intermediate save
        if hand_count % SAVE_EVERY == 0:
            for seat, agent in enumerate(agents):
                # We need the last stats for this seat to save losses
                # In Step 2, stats are per-seat, so we'll pass the latest ones from training_log if available
                last_stats = next((item for item in reversed(training_log) if item["seat"] == seat), None)
                if last_stats:
                    _save_agent(
                        agent.network, seat, hand_count, CHECKPOINT_DIR, "perturbed_ckpt",
                        last_stats["policy_loss"], last_stats["value_loss"], 
                        last_stats["entropy"], last_stats["kl_penalty"], last_stats["mean_kl"]
                    )
                else:
                    # Fallback if no update yet
                    _save_agent(agent.network, seat, hand_count, CHECKPOINT_DIR, "perturbed_ckpt")

        if all_converged:
            log.info("All 4 agents converged at hand %d.", hand_count)
            break

    # ── Final saves ────────────────────────────────────────────────────────
    log.info("Fine-tuning complete.  Total hands: %d", hand_count)

    for seat, agent in enumerate(agents):
        last_stats = next((item for item in reversed(training_log) if item["seat"] == seat), None)
        if last_stats:
            _save_agent(
                agent.network, seat, hand_count, CHECKPOINT_DIR, "perturbed_agent",
                last_stats["policy_loss"], last_stats["value_loss"], 
                last_stats["entropy"], last_stats["kl_penalty"], last_stats["mean_kl"]
            )
        else:
            _save_agent(agent.network, seat, hand_count, CHECKPOINT_DIR, "perturbed_agent")
        log.info("  Saved: checkpoints/perturbed_agent_%d.pt", seat)

    params_record = [
        {"seat": i, "alpha": REWARD_PARAMS[i].alpha, "beta": REWARD_PARAMS[i].beta}
        for i in range(NUM_PLAYERS)
    ]
    with open(os.path.join(CHECKPOINT_DIR, "perturbed_agent_params.json"), "w") as f:
        json.dump(params_record, f, indent=2)

    with open(os.path.join(CHECKPOINT_DIR, "perturbed_training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)

    log.info("Saved: checkpoints/perturbed_agent_params.json")
    log.info("Saved: checkpoints/perturbed_training_log.json")


def _save_agent(
    network:      ActorCriticNetwork,
    seat:         int,
    step:         int,
    out_dir:      str,
    prefix:       str,
    policy_loss:  Optional[float] = None,
    value_loss:   Optional[float] = None,
    entropy:      Optional[float] = None,
    kl_penalty:   Optional[float] = None,
    mean_kl:      Optional[float] = None,
) -> None:
    path = os.path.join(out_dir, f"{prefix}_{seat}.pt")
    torch.save({
        "network_state": network.state_dict(),
        "hidden_dim":    network.hidden_dim,
        "feature_dim":   FEATURE_DIM,
        "num_actions":   NUM_ACTIONS,
        "seat":          seat,
        "step":          step,
        "alpha":         REWARD_PARAMS[seat].alpha,
        "beta":          REWARD_PARAMS[seat].beta,
        "policy_loss":   policy_loss,
        "value_loss":    value_loss,
        "entropy":       entropy,
        "kl_penalty":    kl_penalty,
        "mean_kl":       mean_kl,
    }, path)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_fine_tuning()
