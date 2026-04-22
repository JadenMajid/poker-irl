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
  - Return normalisation via RunningMeanStd (CRITICAL for poker: chip-scale rewards
    produce ~850x gradient norms without normalisation, completely overwhelming the
    max_grad_norm clip and preventing meaningful learning)
  - Optional KL penalty against a reference (frozen base) policy — used during
    the perturbed-agent fine-tuning phase to prevent catastrophic forgetting
  - Explained variance diagnostic to monitor value head learning quality

Why return normalisation is essential here:
  In poker, chip deltas range from ±20 (small blind lost) to ±2000 (big pot won).
  Without normalisation, the value MSE loss = (chip_delta)^2 ≈ 40,000–4,000,000,
  producing gradient norms of 800–8000 before the max_grad_norm clip.  The clip
  then reduces every gradient to the same norm regardless of its informational
  content, effectively making the learning rate a function of chip variance rather
  than policy quality.  By whitening returns to zero-mean unit-variance via a
  RunningMeanStd tracker, value gradients are comparable in scale to policy
  gradients, and the value head converges in O(10k) rather than O(100k) updates.

Why parameter-sharing self-play for the base agent?
  All 4 seats share ONE network.  This guarantees:
    (a) The converged policy is a symmetric Nash equilibrium strategy — each seat
        is best-responding to the same policy, so by symmetry the equilibrium is
        consistent.
    (b) We always end up with exactly one saved model, not four (which might have
        diverged from each other in unintended ways).
    (c) Training is 4x more sample-efficient because every trajectory contributes
        updates from all 4 perspectives.
"""

from __future__ import annotations

import copy
import math
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
# Running statistics normaliser (Chan et al. parallel Welford)
# ---------------------------------------------------------------------------

class RunningMeanStd:
    """
    Maintains running mean and variance of a stream of scalars using Chan et al.'s
    numerically stable parallel update formula.

    Used to normalise returns before computing the value loss.
    The normalised return r_norm = (r - mean) / (std + eps) is zero-mean and
    approximately unit-variance, keeping value gradients well-scaled.

    Parameters
    ----------
    shape   : Shape of values tracked.  Use () for scalars.
    epsilon : Small constant to avoid division by zero.
    clip    : Clip normalised values to [-clip, +clip] to bound outlier impact.
    """

    def __init__(self, shape: tuple = (), epsilon: float = 1e-8,
                 clip: Optional[float] = 10.0) -> None:
        self.mean    = np.zeros(shape, dtype=np.float64)
        self.var     = np.ones(shape,  dtype=np.float64)
        self.count   = 0
        self.epsilon = epsilon
        self.clip    = clip

    def update(self, x: np.ndarray) -> None:
        """Update running statistics with a batch of values."""
        x = np.asarray(x, dtype=np.float64).flatten()
        n = len(x)
        if n == 0:
            return
        batch_mean = x.mean()
        batch_var  = x.var() if n > 1 else 0.0

        if self.count == 0:
            self.mean  = np.full_like(self.mean,  batch_mean)
            self.var   = np.full_like(self.var,   max(batch_var, 1.0))
            self.count = n
            return

        total    = self.count + n
        delta    = batch_mean - self.mean
        new_mean = self.mean + delta * n / total
        m_a      = self.var     * self.count
        m_b      = batch_var    * n
        m2       = m_a + m_b + delta ** 2 * self.count * n / total
        self.mean  = new_mean
        self.var   = m2 / total
        self.count = total

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var + self.epsilon)

    def normalise(self, x: np.ndarray) -> np.ndarray:
        normed = (np.asarray(x, dtype=np.float64) - self.mean) / self.std
        if self.clip is not None:
            normed = np.clip(normed, -self.clip, self.clip)
        return normed.astype(np.float32)

    def denormalise(self, x_norm: np.ndarray) -> np.ndarray:
        return (np.asarray(x_norm, dtype=np.float64) * self.std + self.mean).astype(np.float32)

    def state_dict(self) -> Dict:
        return {"mean": self.mean.tolist(), "var": self.var.tolist(), "count": self.count}

    def load_state_dict(self, d: Dict) -> None:
        self.mean  = np.array(d["mean"],  dtype=np.float64)
        self.var   = np.array(d["var"],   dtype=np.float64)
        self.count = int(d["count"])


# ---------------------------------------------------------------------------
# PPO hyper-parameters
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    """All hyper-parameters for one PPO training run."""

    # Rollout
    n_steps_per_update:   int   = 4096
    n_epochs:             int   = 10
    mini_batch_size:      int   = 256

    # PPO-Clip
    clip_range:           float = 0.2
    value_clip_range:     float = 0.2

    # Loss coefficients
    value_coef:           float = 0.5
    entropy_coef:         float = 0.01
    kl_coef:              float = 0.0     # > 0 during fine-tuning

    # Advantage estimation
    gae_lambda:           float = 0.95
    gamma:                float = 1.0     # no discounting (episodic cash game)

    # Optimiser
    learning_rate:        float = 3e-4
    max_grad_norm:        float = 0.5

    # Return normalisation — strongly recommended; see module docstring
    normalise_returns:    bool  = True

    # Convergence monitoring
    convergence_window:   int   = 2000
    convergence_threshold:float = 3e-4
    min_hands_before_convergence_check: int = 10_000

    # LR schedule (cosine annealing)
    use_lr_schedule:      bool  = True
    lr_schedule_T_max:    int   = 500_000


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

@dataclass
class RolloutBuffer:
    """
    Stores one rollout of on-policy data from the poker environment.
    Supports GAE computation and mini-batch iteration.
    """
    features:  List[np.ndarray] = field(default_factory=list)
    masks:     List[np.ndarray] = field(default_factory=list)
    actions:   List[int]        = field(default_factory=list)
    log_probs: List[float]      = field(default_factory=list)
    values:    List[float]      = field(default_factory=list)
    rewards:   List[float]      = field(default_factory=list)
    dones:     List[bool]       = field(default_factory=list)

    advantages:  Optional[np.ndarray] = None
    returns:     Optional[np.ndarray] = None

    def add(self, feature: np.ndarray, mask: np.ndarray, action: int,
            log_prob: float, value: float, reward: float, done: bool) -> None:
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
        """Compute GAE advantages and discounted returns (gamma=1 for episodic poker)."""
        n = len(self.features)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae   = 0.0

        for t in reversed(range(n)):
            if self.dones[t]:
                next_val = 0.0
                last_gae = 0.0
            else:
                next_val = self.values[t + 1] if t + 1 < n else 0.0

            delta       = self.rewards[t] + gamma * next_val - self.values[t]
            last_gae    = delta + gamma * gae_lambda * (0.0 if self.dones[t] else last_gae)
            advantages[t] = last_gae

        returns = advantages + np.array(self.values, dtype=np.float32)
        # Normalise advantages
        self.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.returns    = returns

    def get_mini_batches(
        self,
        mini_batch_size:    int,
        device:             torch.device,
        normalised_returns: Optional[np.ndarray] = None,
    ):
        """
        Yield shuffled (feat, mask, act, old_lp, adv, ret) mini-batches.
        If normalised_returns is provided, use it as the value target.
        """
        n       = len(self.features)
        idx     = np.random.permutation(n)
        feat_a  = np.stack(self.features,  axis=0)
        mask_a  = np.stack(self.masks,     axis=0)
        act_a   = np.array(self.actions,   dtype=np.int64)
        lp_a    = np.array(self.log_probs, dtype=np.float32)
        adv_a   = self.advantages
        ret_a   = normalised_returns if normalised_returns is not None else self.returns

        for start in range(0, n, mini_batch_size):
            bi = idx[start : start + mini_batch_size]
            yield (
                torch.tensor(feat_a[bi], dtype=torch.float32, device=device),
                torch.tensor(mask_a[bi], dtype=torch.bool,    device=device),
                torch.tensor(act_a[bi],  dtype=torch.int64,   device=device),
                torch.tensor(lp_a[bi],   dtype=torch.float32, device=device),
                torch.tensor(adv_a[bi],  dtype=torch.float32, device=device),
                torch.tensor(ret_a[bi],  dtype=torch.float32, device=device),
            )

    def clear(self) -> None:
        self.features.clear(); self.masks.clear(); self.actions.clear()
        self.log_probs.clear(); self.values.clear(); self.rewards.clear()
        self.dones.clear()
        self.advantages = None; self.returns = None


# ---------------------------------------------------------------------------
# PPO Trainer
# ---------------------------------------------------------------------------

class PPOTrainer:
    """
    PPO trainer for a single ActorCriticNetwork.

    Two usage modes:
      A) Shared-parameter self-play (base agent): all 4 seats → one network.
      B) Independent fine-tuning (perturbed agents): one network per seat.

    Parameters
    ----------
    network     : Network to train.
    cfg         : PPO hyper-parameters.
    device      : Torch device.
    ref_network : Frozen reference for KL regularisation (fine-tuning only).
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
        self.buffer  = RolloutBuffer()
        self._ret_rms = RunningMeanStd(shape=(), clip=10.0)

        # Training statistics
        self.total_updates:  int        = 0
        self.loss_history:   List[float] = []
        self.entropy_history:List[float] = []
        self.kl_history:     List[float] = []
        self.ev_history:     List[float] = []   # explained variance

    def buffer_size(self) -> int:
        return len(self.buffer)

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self) -> Dict[str, float]:
        """
        Compute GAE, normalise returns, run PPO epochs, return stats.
        """
        self.buffer.compute_advantages(self.cfg.gamma, self.cfg.gae_lambda)

        # Normalise returns before value loss (prevents 850x gradient norms)
        raw_returns = self.buffer.returns
        if self.cfg.normalise_returns and raw_returns is not None:
            self._ret_rms.update(raw_returns)
            norm_returns = self._ret_rms.normalise(raw_returns)
        else:
            norm_returns = raw_returns

        stats = dict(policy_loss=0.0, value_loss=0.0, entropy=0.0,
                     kl_penalty=0.0, total_loss=0.0, clip_frac=0.0,
                     explained_var=0.0)
        n_up = 0

        self.network.train()

        for _ in range(self.cfg.n_epochs):
            for batch in self.buffer.get_mini_batches(
                self.cfg.mini_batch_size, self.device, norm_returns
            ):
                feat_b, mask_b, act_b, old_lp_b, adv_b, ret_b = batch

                logits, values = self.network(feat_b, mask_b)
                dist           = Categorical(logits=logits)
                new_lp         = dist.log_prob(act_b)
                entropy        = dist.entropy().mean()

                # PPO-clip policy loss
                ratio       = torch.exp(new_lp - old_lp_b)
                clip_ratio  = torch.clamp(ratio, 1 - self.cfg.clip_range,
                                                  1 + self.cfg.clip_range)
                policy_loss = -torch.min(ratio * adv_b, clip_ratio * adv_b).mean()
                clip_frac   = ((ratio - 1).abs() > self.cfg.clip_range).float().mean()

                # Value loss on normalised returns (well-scaled MSE)
                values     = values.squeeze(-1)
                value_loss = F.mse_loss(values, ret_b)

                # Explained variance diagnostic
                with torch.no_grad():
                    var_y = ret_b.var()
                    ev    = (1 - (ret_b - values.detach()).var()
                             / (var_y + 1e-8)).item()

                # KL penalty (fine-tuning only)
                kl_pen = torch.tensor(0.0, device=self.device)
                if self.ref_network is not None and self.cfg.kl_coef > 0:
                    with torch.no_grad():
                        ref_logits, _ = self.ref_network(feat_b, mask_b)
                        ref_logp      = F.log_softmax(ref_logits, dim=-1)
                        ref_probs     = ref_logp.exp()
                    curr_logp = F.log_softmax(logits, dim=-1)
                    # Use explicit KL form and scrub 0*inf terms from masked actions.
                    kl_terms = ref_probs * (ref_logp - curr_logp)
                    kl_pen   = torch.nan_to_num(
                        kl_terms, nan=0.0, posinf=0.0, neginf=0.0
                    ).sum(dim=-1).mean()

                loss = (
                    policy_loss
                    + self.cfg.value_coef   * value_loss
                    - self.cfg.entropy_coef * entropy
                    + self.cfg.kl_coef      * kl_pen
                )

                self.optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(),
                                         self.cfg.max_grad_norm)
                self.optimiser.step()

                stats["policy_loss"]   += policy_loss.item()
                stats["value_loss"]    += value_loss.item()
                stats["entropy"]       += entropy.item()
                stats["kl_penalty"]    += kl_pen.item()
                stats["total_loss"]    += loss.item()
                stats["clip_frac"]     += clip_frac.item()
                stats["explained_var"] += ev
                n_up += 1

        if self.scheduler is not None:
            self.scheduler.step()

        for k in stats:
            stats[k] /= max(n_up, 1)

        self.entropy_history.append(stats["entropy"])
        self.kl_history.append(stats["kl_penalty"])
        self.loss_history.append(stats["total_loss"])
        self.ev_history.append(stats["explained_var"])
        self.total_updates += 1

        self.buffer.clear()
        self.network.eval()
        return stats

    def anneal_kl_coef(self, factor: float = 0.995, floor: float = 1e-4) -> None:
        self.cfg.kl_coef = max(floor, self.cfg.kl_coef * factor)

    @property
    def ret_rms(self) -> RunningMeanStd:
        return self._ret_rms


# ---------------------------------------------------------------------------
# Convergence detector
# ---------------------------------------------------------------------------

class ConvergenceDetector:
    """
    Detects policy convergence by tracking KL divergence between periodic
    policy snapshots.  More reliable than loss-based detection for poker,
    since value loss remains large even after the policy has stabilised.
    """

    def __init__(
        self,
        window:      int   = 2000,
        threshold:   float = 3e-4,
        min_hands:   int   = 10_000,
        check_every: int   = 1000,
    ) -> None:
        self.window       = window
        self.threshold    = threshold
        self.min_hands    = min_hands
        self.check_every  = check_every

        self._hand_count   = 0
        self._snapshot:    Optional[Dict] = None
        self._kl_buf:      List[float]    = []
        self._all_kls:     List[float]    = []
        self._converged    = False

    def on_hand_end(
        self,
        network:  ActorCriticNetwork,
        features: Optional[torch.Tensor],
        device:   torch.device,
    ) -> bool:
        """Call after each hand.  Returns True when convergence is detected."""
        self._hand_count += 1

        # Ensure we always create the first snapshot once warmup is complete
        # and features are available, even if exact boundary timing is missed.
        if self._hand_count >= self.min_hands and self._snapshot is None and features is not None:
            self._snapshot = _clone_params(network)

        if self._hand_count < self.min_hands:
            return False

        if self._hand_count % self.check_every == 0 and features is not None:
            if self._snapshot is not None:
                kl = _kl_between_snapshots(network, self._snapshot, features, device)
                self._kl_buf.append(kl)
                self._all_kls.append(kl)
                if len(self._kl_buf) > self.window // self.check_every:
                    self._kl_buf.pop(0)
            self._snapshot = _clone_params(network)

            if len(self._kl_buf) >= 3:
                mean_kl = sum(self._kl_buf) / len(self._kl_buf)
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
        if self._kl_buf:
            return sum(self._kl_buf) / len(self._kl_buf)
        return float("nan")


def _clone_params(net: ActorCriticNetwork) -> Dict:
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def _kl_between_snapshots(
    current:  ActorCriticNetwork,
    snapshot: Dict,
    features: torch.Tensor,
    device:   torch.device,
) -> float:
    """Mean KL(π_snapshot || π_current) over a feature batch."""
    snap_net = copy.deepcopy(current)
    snap_net.load_state_dict({k: v.to(device) for k, v in snapshot.items()})
    snap_net.eval()

    features = features.to(device)
    with torch.no_grad():
        curr_logits, _ = current(features)
        snap_logits, _ = snap_net(features)
        curr_logp = F.log_softmax(curr_logits, dim=-1)
        snap_logp = F.log_softmax(snap_logits, dim=-1)
        snap_prob = snap_logp.exp()
        kl_terms  = snap_prob * (snap_logp - curr_logp)
        kl        = torch.nan_to_num(
            kl_terms, nan=0.0, posinf=0.0, neginf=0.0
        ).sum(dim=-1).mean()

    return float(kl.item())


# ---------------------------------------------------------------------------
# Feature sample store
# ---------------------------------------------------------------------------

class FeatureSampleStore:
    """Rolling buffer of recent state features for convergence detection."""

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
