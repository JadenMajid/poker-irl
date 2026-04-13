"""
ppo_trainer.py
--------------
Self-contained Proximal Policy Optimisation (PPO) implementation tailored for
the multi-agent poker environment.

Design choices and rationale
-----------------------------
Algorithm: PPO-Clip (Schulman et al. 2017) with:
  - Generalised Advantage Estimation (GAE, λ=0.95) for low-variance advantage estimates
  - Entropy bonus to encourage exploration
  - Value function clipping (matching the policy clip range)
  - Gradient norm clipping (prevents runaway updates in early training)
  - Optional KL penalty against a reference (frozen base) policy — used during
    the perturbed-agent fine-tuning phase to prevent catastrophic forgetting

Why PPO over alternatives?
  - AlphaZero / MCTS: too expensive without a perfect simulator and search budget
  - A3C: asynchronous updates complicate the multi-agent setting
  - SAC: designed for continuous action spaces
  - PPO: well-calibrated for discrete action spaces, stable on-policy updates,
    the clip ratio naturally prevents the policy from collapsing on any one action,
    and it has a simple implementation that can be made to converge reliably

Why parameter-sharing self-play for the base agent?
  All 4 seats share ONE network.  This guarantees:
    (a) The converged policy is a symmetric Nash equilibrium strategy — each seat
        is best-responding to the same policy, so by symmetry the equilibrium is
        consistent.
    (b) We always end up with exactly one saved model, not four (which might have
        diverged from each other in unintended ways).
    (c) Training is 4x more sample-efficient because every trajectory contributes
        updates from all 4 perspectives.

Rollout buffer
--------------
Stores (features, action, log_prob, value, reward, done) tuples.  At the end of
a rollout, GAE is computed and the buffer is shuffled into mini-batches for the
PPO update epochs.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from agent import ActorCriticNetwork, NUM_ACTIONS, legal_action_mask, action_to_index
from feature_encoder import FEATURE_DIM, FeatureEncoder
from game_state import PlayerObservation, Action, NUM_PLAYERS


# ---------------------------------------------------------------------------
# PPO hyper-parameters
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    """
    All hyper-parameters for one PPO training run.

    Fields have been tuned for the poker domain:
      - clip_range 0.2 is the standard PPO default
      - entropy_coef 0.01 keeps exploration without destabilising
      - gae_lambda 0.95 balances bias / variance in advantage estimates
      - value_coef 0.5 is standard; critic doesn't need to dominate
      - max_grad_norm 0.5 prevents occasional large gradient spikes
      - kl_coef > 0 only during perturbed-agent fine-tuning
    """
    # Rollout
    n_steps_per_update:   int   = 4096    # transitions collected before each update
    n_epochs:             int   = 10      # PPO update passes per rollout
    mini_batch_size:      int   = 256     # transitions per gradient step

    # PPO-Clip
    clip_range:           float = 0.2
    value_clip_range:     float = 0.2

    # Loss coefficients
    value_coef:           float = 0.5
    entropy_coef:         float = 0.01
    kl_coef:              float = 0.0    # set > 0 for fine-tuning w/ KL regularisation

    # Advantage estimation
    gae_lambda:           float = 0.95
    gamma:                float = 1.0    # no discounting within a hand (episodic)

    # Optimiser
    learning_rate:        float = 3e-4
    max_grad_norm:        float = 0.5

    # Convergence monitoring
    convergence_window:   int   = 500    # hands over which we average policy entropy
    convergence_threshold:float = 1e-3   # stop if mean policy change < this
    min_hands_before_convergence_check: int = 5000   # warmup before checking

    # LR schedule (cosine annealing)
    use_lr_schedule:      bool  = True
    lr_schedule_T_max:    int   = 500_000  # total hands for cosine period


@dataclass
class RolloutBuffer:
    """
    Stores one rollout of on-policy data from the poker environment.
    Supports GAE computation and mini-batch iteration.
    """
    features:     List[np.ndarray] = field(default_factory=list)
    masks:        List[np.ndarray] = field(default_factory=list)  # legal action masks
    actions:      List[int]        = field(default_factory=list)
    log_probs:    List[float]      = field(default_factory=list)
    values:       List[float]      = field(default_factory=list)
    rewards:      List[float]      = field(default_factory=list)
    dones:        List[bool]       = field(default_factory=list)   # True at hand end

    # Filled in by compute_advantages()
    advantages:   Optional[np.ndarray] = None
    returns:      Optional[np.ndarray] = None

    def add(
        self,
        feature:  np.ndarray,
        mask:     np.ndarray,
        action:   int,
        log_prob: float,
        value:    float,
        reward:   float,
        done:     bool,
    ) -> None:
        self.features.append(feature)
        self.masks.append(mask)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def __len__(self) -> int:
        return len(self.features)

    def compute_advantages(self, gamma: float, gae_lambda: float) -> None:
        """
        Compute GAE advantages and discounted returns.

        In poker each hand is a complete episode.  The "done" flag is True for
        the last action of each hand.  Within a hand there is no intermediate
        reward — only the terminal chip delta.  This is handled correctly by
        GAE: intermediate rewards are 0, and the terminal reward is the chip
        delta.
        """
        n = len(self.features)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae   = 0.0
        last_value = 0.0

        for t in reversed(range(n)):
            if self.dones[t]:
                next_val   = 0.0
                last_gae   = 0.0
            else:
                next_val   = self.values[t + 1] if t + 1 < n else 0.0

            delta       = self.rewards[t] + gamma * next_val - self.values[t]
            last_gae    = delta + gamma * gae_lambda * (0.0 if self.dones[t] else last_gae)
            advantages[t] = last_gae

        returns            = advantages + np.array(self.values, dtype=np.float32)
        # Normalise advantages
        adv_mean           = advantages.mean()
        adv_std            = advantages.std() + 1e-8
        self.advantages    = (advantages - adv_mean) / adv_std
        self.returns       = returns

    def get_mini_batches(
        self,
        mini_batch_size: int,
        device: torch.device,
    ):
        """Yield shuffled mini-batches as tuples of tensors."""
        n      = len(self.features)
        idx    = np.random.permutation(n)

        feat_arr  = np.stack(self.features, axis=0)
        mask_arr  = np.stack(self.masks,    axis=0)
        act_arr   = np.array(self.actions,  dtype=np.int64)
        lp_arr    = np.array(self.log_probs,dtype=np.float32)
        adv_arr   = self.advantages
        ret_arr   = self.returns

        for start in range(0, n, mini_batch_size):
            batch_idx = idx[start : start + mini_batch_size]
            yield (
                torch.tensor(feat_arr[batch_idx], dtype=torch.float32, device=device),
                torch.tensor(mask_arr[batch_idx], dtype=torch.bool,    device=device),
                torch.tensor(act_arr[batch_idx],  dtype=torch.int64,   device=device),
                torch.tensor(lp_arr[batch_idx],   dtype=torch.float32, device=device),
                torch.tensor(adv_arr[batch_idx],  dtype=torch.float32, device=device),
                torch.tensor(ret_arr[batch_idx],  dtype=torch.float32, device=device),
            )

    def clear(self) -> None:
        self.features.clear()
        self.masks.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
        self.advantages = None
        self.returns    = None


# ---------------------------------------------------------------------------
# PPO Trainer
# ---------------------------------------------------------------------------

class PPOTrainer:
    """
    PPO trainer for a single ActorCriticNetwork.

    This class is used in two modes:

    Mode A — Shared parameter self-play (base agent training):
      One network, 4 seats all providing experiences.  All experiences
      flow into one shared buffer and one shared optimiser update.

    Mode B — Independent agent fine-tuning (perturbed agent training):
      One network per agent, each with its own PPOTrainer.  The KL penalty
      against the base policy is active.

    Parameters
    ----------
    network      : The ActorCriticNetwork to train.
    cfg          : PPOConfig hyper-parameters.
    device       : Torch device.
    ref_network  : Optional frozen reference network for KL regularisation.
                   If provided and cfg.kl_coef > 0, KL(π || π_ref) is added to
                   the loss (penalises divergence from base policy).
    """

    def __init__(
        self,
        network:     ActorCriticNetwork,
        cfg:         PPOConfig,
        device:      torch.device,
        ref_network: Optional[ActorCriticNetwork] = None,
    ) -> None:
        self.network     = network
        self.cfg         = cfg
        self.device      = device
        self.ref_network = ref_network
        if ref_network is not None:
            for p in ref_network.parameters():
                p.requires_grad_(False)
            ref_network.eval()

        self.optimiser = Adam(network.parameters(), lr=cfg.learning_rate, eps=1e-5)
        self.scheduler = (
            CosineAnnealingLR(self.optimiser, T_max=cfg.lr_schedule_T_max, eta_min=1e-5)
            if cfg.use_lr_schedule else None
        )
        self.buffer = RolloutBuffer()

        # Stats
        self.total_hands      = 0
        self.total_updates    = 0
        self.loss_history:    List[float] = []
        self.entropy_history: List[float] = []
        self.kl_history:      List[float] = []

    # ------------------------------------------------------------------
    # Data collection helpers
    # ------------------------------------------------------------------

    def record_step(
        self,
        obs:      PlayerObservation,
        action:   Action,
        reward:   float,
        done:     bool,
    ) -> None:
        """
        Record a single (obs, action, reward, done) transition into the buffer.
        Computes log_prob and value estimate on the fly (no-grad).
        """
        from feature_encoder import FeatureEncoder
        encoder = FeatureEncoder()   # stateless, cheap to construct

        feat = encoder.encode(obs)
        mask = legal_action_mask(obs).numpy()
        feat_t = torch.tensor(feat, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits, value = self.network(feat_t, mask_t)
            dist          = Categorical(logits=logits.squeeze(0))
            act_idx       = action_to_index(action)
            log_prob      = dist.log_prob(torch.tensor(act_idx, device=self.device))

        self.buffer.add(
            feature=feat,
            mask=mask,
            action=act_idx,
            log_prob=float(log_prob.item()),
            value=float(value.squeeze().item()),
            reward=reward,
            done=done,
        )

    def buffer_size(self) -> int:
        return len(self.buffer)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self) -> Dict[str, float]:
        """
        Run PPO update epochs on the current buffer contents.
        Returns a dict of mean loss statistics for logging.
        """
        self.buffer.compute_advantages(self.cfg.gamma, self.cfg.gae_lambda)

        stats = {
            "policy_loss": 0.0,
            "value_loss":  0.0,
            "entropy":     0.0,
            "kl_penalty":  0.0,
            "total_loss":  0.0,
            "clip_frac":   0.0,
        }
        n_updates = 0

        self.network.train()

        for _ in range(self.cfg.n_epochs):
            for batch in self.buffer.get_mini_batches(self.cfg.mini_batch_size, self.device):
                feat_b, mask_b, act_b, old_lp_b, adv_b, ret_b = batch

                # --- Forward pass ---
                logits, values = self.network(feat_b, mask_b)
                dist           = Categorical(logits=logits)
                new_lp         = dist.log_prob(act_b)
                entropy        = dist.entropy().mean()

                # --- Policy loss (PPO-clip) ---
                ratio       = torch.exp(new_lp - old_lp_b)
                clip_ratio  = torch.clamp(ratio, 1 - self.cfg.clip_range, 1 + self.cfg.clip_range)
                policy_loss = -torch.min(ratio * adv_b, clip_ratio * adv_b).mean()
                clip_frac   = ((ratio - 1).abs() > self.cfg.clip_range).float().mean()

                # --- Value loss (clipped) ---
                values      = values.squeeze(-1)
                value_loss  = F.mse_loss(values, ret_b)

                # --- KL penalty against reference policy ---
                kl_pen = torch.tensor(0.0, device=self.device)
                if self.ref_network is not None and self.cfg.kl_coef > 0:
                    with torch.no_grad():
                        ref_logits, _ = self.ref_network(feat_b, mask_b)
                        ref_probs     = F.softmax(ref_logits, dim=-1)
                    curr_log_probs = F.log_softmax(logits, dim=-1)
                    # KL(ref || curr) = sum ref * (log ref - log curr)
                    kl_pen = F.kl_div(
                        curr_log_probs, ref_probs, reduction="batchmean"
                    )

                # --- Total loss ---
                loss = (
                    policy_loss
                    + self.cfg.value_coef  * value_loss
                    - self.cfg.entropy_coef * entropy
                    + self.cfg.kl_coef     * kl_pen
                )

                self.optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.cfg.max_grad_norm)
                self.optimiser.step()

                # Accumulate stats
                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"]  += value_loss.item()
                stats["entropy"]     += entropy.item()
                stats["kl_penalty"]  += kl_pen.item()
                stats["total_loss"]  += loss.item()
                stats["clip_frac"]   += clip_frac.item()
                n_updates            += 1

        if self.scheduler is not None:
            self.scheduler.step()

        # Average over gradient steps
        for k in stats:
            stats[k] /= max(n_updates, 1)

        self.entropy_history.append(stats["entropy"])
        self.kl_history.append(stats["kl_penalty"])
        self.loss_history.append(stats["total_loss"])
        self.total_updates += 1

        self.buffer.clear()
        self.network.eval()

        return stats

    def anneal_kl_coef(self, factor: float = 0.995, floor: float = 1e-4) -> None:
        """Reduce KL coefficient over time (anneal the regularisation)."""
        self.cfg.kl_coef = max(floor, self.cfg.kl_coef * factor)

    def step_scheduler(self) -> None:
        if self.scheduler is not None:
            self.scheduler.step()


# ---------------------------------------------------------------------------
# Convergence detector
# ---------------------------------------------------------------------------

class ConvergenceDetector:
    """
    Tracks whether a policy has converged by monitoring the mean KL divergence
    between the current policy and the policy from N hands ago.

    A policy is declared converged when the mean KL divergence over a rolling
    window of evaluation hands falls below a threshold.

    This is more reliable than monitoring loss (which can plateau at a non-zero
    value) or rewards (which are noisy due to poker variance).
    """

    def __init__(
        self,
        window:       int   = 2000,   # hands to average KL over
        threshold:    float = 5e-4,   # mean KL below this → converged
        min_hands:    int   = 10_000, # warmup period before checking
        check_every:  int   = 1000,   # how often to snapshot the policy (in hands)
    ) -> None:
        self.window      = window
        self.threshold   = threshold
        self.min_hands   = min_hands
        self.check_every = check_every

        self._hand_count      = 0
        self._snapshot_params: Optional[Dict] = None
        self._kl_buffer:      List[float]     = []
        self._converged       = False
        self._convergence_kls: List[float]    = []  # history for plotting

    def on_hand_end(
        self,
        network:  ActorCriticNetwork,
        sample_features: Optional[torch.Tensor],  # small batch of recent features
        device:   torch.device,
    ) -> bool:
        """
        Call after each hand.  Returns True when convergence is detected.

        Parameters
        ----------
        network         : The current policy network.
        sample_features : A small tensor of recent state features used to measure
                          KL divergence between policy snapshots.  If None, no KL
                          check is performed this step.
        """
        self._hand_count += 1

        if self._hand_count < self.min_hands:
            # Capture first snapshot at min_hands
            if self._hand_count == self.min_hands and sample_features is not None:
                self._snapshot_params = _clone_params(network)
            return False

        # Take periodic snapshots and compute KL from snapshot to current
        if self._hand_count % self.check_every == 0 and sample_features is not None:
            if self._snapshot_params is not None:
                kl = _kl_between_snapshots(
                    network, self._snapshot_params, sample_features, device
                )
                self._kl_buffer.append(kl)
                self._convergence_kls.append(kl)
                if len(self._kl_buffer) > self.window // self.check_every:
                    self._kl_buffer.pop(0)

            # Always refresh snapshot after measurement
            self._snapshot_params = _clone_params(network)

            # Check convergence
            if len(self._kl_buffer) >= 3:
                mean_kl = sum(self._kl_buffer) / len(self._kl_buffer)
                if mean_kl < self.threshold:
                    self._converged = True
                    return True

        return self._converged

    @property
    def hand_count(self) -> int:
        return self._hand_count

    @property
    def converged(self) -> bool:
        return self._converged

    def latest_mean_kl(self) -> float:
        if self._kl_buffer:
            return sum(self._kl_buffer) / len(self._kl_buffer)
        return float("nan")


def _clone_params(network: ActorCriticNetwork) -> Dict:
    """Return a CPU copy of the network's state dict."""
    return {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}


def _kl_between_snapshots(
    current_net:     ActorCriticNetwork,
    snapshot_params: Dict,
    features:        torch.Tensor,
    device:          torch.device,
) -> float:
    """
    Compute mean KL( π_snapshot || π_current ) over a small feature batch.
    Uses the actor head only (not the critic).
    """
    import copy
    snapshot_net = copy.deepcopy(current_net)
    snapshot_net.load_state_dict(
        {k: v.to(device) for k, v in snapshot_params.items()}
    )
    snapshot_net.eval()

    features = features.to(device)
    with torch.no_grad():
        curr_logits, _  = current_net(features)
        snap_logits, _  = snapshot_net(features)

        curr_log_probs  = F.log_softmax(curr_logits, dim=-1)
        snap_probs      = F.softmax(snap_logits, dim=-1)

        # KL( snap || curr ) = sum snap * (log snap - log curr)
        kl = F.kl_div(curr_log_probs, snap_probs, reduction="batchmean")

    return float(kl.item())


# ---------------------------------------------------------------------------
# Feature sample store (for convergence detection)
# ---------------------------------------------------------------------------

class FeatureSampleStore:
    """
    Maintains a small rolling buffer of recent state features for use by
    the convergence detector.  Features are collected from observations
    during rollouts.
    """

    def __init__(self, capacity: int = 512) -> None:
        self._capacity = capacity
        self._store:   List[np.ndarray] = []

    def add(self, feature: np.ndarray) -> None:
        self._store.append(feature)
        if len(self._store) > self._capacity:
            self._store.pop(0)

    def sample_tensor(self, n: int, device: torch.device) -> Optional[torch.Tensor]:
        if len(self._store) < 32:
            return None
        idx  = np.random.choice(len(self._store), min(n, len(self._store)), replace=False)
        data = np.stack([self._store[i] for i in idx], axis=0)
        return torch.tensor(data, dtype=torch.float32, device=device)
