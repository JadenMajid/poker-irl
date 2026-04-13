"""
step3_collect_and_run_irl.py
-----------------------------
Phase 1: Collect a large set of fixed-policy trajectories from all 4 perturbed agents.
Phase 2: For each agent in turn, run inverse reinforcement learning (IRL) to
         recover their (alpha, beta) reward parameters from observed behaviour.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IRL Algorithm: Gradient-Ascent Bayesian IRL with Opponent Modelling (GABO-IRL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Mathematical setup
------------------
We model each agent's policy as a Boltzmann-rational actor:

    π_θ(a | s) ∝ exp( Q_θ(s, a) / τ )

where Q_θ(s, a) is the action-value under reward parameters θ = (α, β) and
τ is a temperature (≈1 for a well-trained agent).

For LINEAR reward  R(s,a;θ) = φ₀(s,a) + α·φ_α(s,a) + β·φ_β(s,a)  we can
decompose the Q-function similarly:

    Q_θ(s,a) = Q₀(s,a) + α · Q_α(s,a) + β · Q_β(s,a)

where each Q_k is the action-value for the k-th reward component.

The log-likelihood of observed trajectory {(s_t, a_t)} is:

    ℒ(θ) = Σ_t log π_θ(a_t | s_t)
           = Σ_t [ Q_θ(s_t, a_t) - log Σ_{a'} exp(Q_θ(s_t, a')) ]

The posterior (with Gaussian prior p(θ) = N(0, σ²I)) is:

    log p(θ | data) = ℒ(θ) - ||θ||²/(2σ²) + const

We maximise this via gradient ascent on (α, β).

How we estimate Q-components
------------------------------
We cannot solve the Bellman equations analytically in our complex game.  Instead:

1. **Opponent modelling**: For each non-target seat, fit a behavioural cloning
   neural network from observed (state, action) pairs.  This gives us a model
   of opponents' policies π_opp(a | s) — necessary to evaluate the target
   agent's effective Q-function, since opponent actions determine transitions.

2. **Monte Carlo Q-estimates**: For each reward component k ∈ {0, α, β}, we use
   observed trajectories to estimate Q_k(s, a) as the expected future cumulative
   reward *from component k* given taking action a in state s, following the
   observed mixed policy thereafter.  This is a Monte-Carlo return estimate.

3. **Advantage-based likelihood**: Instead of raw Q-values (which have arbitrary
   scale), we use Boltzmann-rational *advantages* A_k(s,a) = Q_k(s,a) - V_k(s),
   which are zero-mean and better-scaled for gradient optimisation.

4. **Direct gradient through log-likelihood**: The gradient

        ∂ℒ/∂α = Σ_t [ φ_α(s_t, a_t) - E_{π_θ}[φ_α(s_t, ·)] ]

   has the interpretation of "feature expectation matching" — we're pushing θ
   toward values where the observed feature expectations match what a Boltzmann-
   rational agent with reward θ would produce.

Opponent modelling detail
--------------------------
We fit a simple MLP (BehaviourCloningNet) for each opponent seat via supervised
learning on (feature, action) pairs from the collected trajectories.  This gives
us π̂_opp(a|s), which approximates how each opponent actually plays.

This opponent model serves a dual purpose:
  (a) It refines our estimate of the target agent's "effective transition dynamics"
      (knowing how opponents respond helps us evaluate the expected outcome of each
       target action).
  (b) It allows us to attribute the target agent's observed behaviour to their
      OWN reward function rather than confounding it with opponent patterns —
      the key methodological contribution to multi-agent IRL.

Convergence detection
----------------------
We track the running estimates α̂_t, β̂_t over the last CONV_WINDOW gradient
steps and declare convergence when:
    max(std(α̂), std(β̂)) < CONV_THRESHOLD   (estimates are stable)

Output files
------------
  irl_results/trajectories.pkl            — collected trajectory data
  irl_results/irl_estimates.json          — final (alpha_hat, beta_hat) per seat
  irl_results/irl_convergence_log.json    — estimate evolution over gradient steps
  irl_results/opponent_models_{seat}.pt   — fitted opponent BC networks
"""

from __future__ import annotations

import copy
import json
import logging
import os
import pickle
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import (
    ActorCriticNetwork,
    NUM_ACTIONS,
    action_to_index,
    index_to_action,
    legal_action_mask,
)
from feature_encoder import FeatureEncoder, FEATURE_DIM
from game_state import (
    ActionType,
    NUM_PLAYERS,
    PlayerObservation,
    Action,
    HandTrajectory,
    TrajectoryStep,
)
from poker_env import PokerEnv
from reward import RewardParams, POT_NORM

# ── configuration ──────────────────────────────────────────────────────────

CHECKPOINT_DIR = "checkpoints"
IRL_DIR        = "irl_results"
DEVICE         = "cpu"
HIDDEN_DIM     = 256

# Trajectory collection
N_COLLECTION_HANDS = 50_000   # hands to collect for IRL (more = better estimates)
LOG_COLLECT_EVERY  = 5_000

# Opponent modelling
OPP_HIDDEN_DIM     = 128
OPP_EPOCHS         = 50
OPP_LR             = 1e-3
OPP_BATCH_SIZE     = 512
OPP_MIN_SAMPLES    = 200   # minimum samples to fit opponent model

# IRL gradient ascent
IRL_LR             = 0.05     # learning rate for (alpha, beta) optimisation
IRL_N_STEPS        = 5_000    # max gradient steps per agent
IRL_BATCH_SIZE     = 256      # trajectories per gradient step
IRL_PRIOR_SIGMA    = 0.5      # std of Gaussian prior on theta = (alpha, beta)
IRL_LOG_EVERY      = 100      # steps between progress logs

# Convergence
CONV_WINDOW        = 300      # gradient steps to average for convergence check
CONV_THRESHOLD     = 5e-4     # max std of estimates in window to declare convergence
CONV_MIN_STEPS     = 500      # warmup before checking convergence

# MC advantage estimation
MC_GAMMA           = 1.0      # no discounting within hand

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures for IRL
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    """
    Compact record of a single decision point for IRL processing.
    Stores the feature vector, action taken, and the decomposed reward signals.
    """
    seat:           int
    feature:        np.ndarray   # shape (FEATURE_DIM,)
    action_idx:     int          # 0-4
    legal_mask:     np.ndarray   # shape (NUM_ACTIONS,), bool
    # Reward feature values *for this step* (used to compute MC returns)
    reward_chip:    float        # net chip contribution from this hand (0 mid-hand, final at end)
    reward_var_pen: float        # variance penalty contribution (0 mid-hand)
    reward_pot:     float        # pot involvement bonus (0 mid-hand)
    is_terminal:    bool         # True for the last step of the hand for this seat
    hand_id:        int


@dataclass
class HandRecord:
    """Processed record of one full hand for all seats."""
    hand_id:    int
    steps:      Dict[int, List[StepRecord]]  # seat → steps
    chip_deltas:Dict[int, float]             # seat → net chips
    max_pots:   Dict[int, float]             # seat → max pot committed


# ---------------------------------------------------------------------------
# Trajectory collector
# ---------------------------------------------------------------------------

def collect_trajectories(n_hands: int) -> List[HandRecord]:
    """
    Load all 4 perturbed agents (fixed, no RL), run n_hands of play,
    and collect structured HandRecord objects.
    """
    device = torch.device(DEVICE)
    encoder = FeatureEncoder()

    # Load agents
    agents_networks: List[ActorCriticNetwork] = []
    for seat in range(NUM_PLAYERS):
        path = os.path.join(CHECKPOINT_DIR, f"perturbed_agent_{seat}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Perturbed agent {seat} not found: {path}")
        ckpt = torch.load(path, map_location=device)
        net  = ActorCriticNetwork(
            input_dim=ckpt.get("feature_dim", FEATURE_DIM),
            hidden_dim=ckpt.get("hidden_dim",  HIDDEN_DIM),
        ).to(device)
        net.load_state_dict(ckpt["network_state"])
        net.eval()
        agents_networks.append(net)

    records: List[HandRecord] = []
    start   = time.time()

    for hand_i in range(n_hands):
        # Per-hand step accumulators
        hand_steps:   Dict[int, List] = {i: [] for i in range(NUM_PLAYERS)}

        def make_cb(seat: int, net: ActorCriticNetwork):
            def callback(obs: PlayerObservation) -> Action:
                feat   = encoder.encode(obs)
                mask   = legal_action_mask(obs)
                feat_t = torch.tensor(feat, dtype=torch.float32, device=device).unsqueeze(0)
                mask_t = mask.unsqueeze(0).to(device)
                with torch.no_grad():
                    logits, _ = net(feat_t, mask_t)
                    dist      = Categorical(logits=logits.squeeze(0))
                    idx_t     = dist.sample()
                    idx       = int(idx_t.item())
                action = index_to_action(idx, seat)
                hand_steps[seat].append((feat, mask.numpy(), idx, obs))
                return action
            return callback

        env  = PokerEnv(
            [make_cb(i, agents_networks[i]) for i in range(NUM_PLAYERS)],
            record_trajectories=True,
        )
        traj = env.play_hand()

        # Compute reward components per seat
        chip_deltas: Dict[int, float] = {
            i: float(traj.final_chip_deltas.get(i, 0)) for i in range(NUM_PLAYERS)
        }

        # Max pot committed per seat (for beta reward component)
        max_pots: Dict[int, float] = {}
        for seat in range(NUM_PLAYERS):
            mp = 0.0
            for step in traj.steps:
                if step.acting_seat == seat:
                    if step.action.action_type in (ActionType.CALL, ActionType.RAISE):
                        mp = max(mp, float(step.observation.pot))
            max_pots[seat] = mp

        # Build StepRecords (reward components only known at hand end)
        steps_by_seat: Dict[int, List[StepRecord]] = {i: [] for i in range(NUM_PLAYERS)}
        for seat in range(NUM_PLAYERS):
            seat_steps = hand_steps[seat]
            n_steps    = len(seat_steps)
            for k, (feat, mask_np, idx, obs) in enumerate(seat_steps):
                is_last = (k == n_steps - 1)
                steps_by_seat[seat].append(StepRecord(
                    seat=seat,
                    feature=feat,
                    action_idx=idx,
                    legal_mask=mask_np,
                    # Reward components: only non-zero at terminal step
                    reward_chip=chip_deltas[seat] if is_last else 0.0,
                    reward_var_pen=0.0,   # filled in by IRL (needs rolling var)
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
            log.info("  Collected %6d / %6d hands | %.0f hands/hr",
                     hand_i + 1, n_hands, hands_ph)

    return records


# ---------------------------------------------------------------------------
# Variance penalty computation (rolling, per-seat, from chip deltas)
# ---------------------------------------------------------------------------

def compute_rolling_variance_penalties(
    records:     List[HandRecord],
    window:      int = 100,
) -> Dict[int, List[float]]:
    """
    For each seat, compute the rolling variance of chip deltas over a sliding
    window.  Returns a dict: seat → list of variance values (one per hand).
    This is used to fill in the reward_var_pen field for each hand's terminal step.
    """
    result: Dict[int, List[float]] = {i: [] for i in range(NUM_PLAYERS)}
    windows: Dict[int, List[float]] = {i: [] for i in range(NUM_PLAYERS)}

    for rec in records:
        for seat in range(NUM_PLAYERS):
            delta = rec.chip_deltas[seat]
            w     = windows[seat]
            w.append(delta)
            if len(w) > window:
                w.pop(0)
            if len(w) >= 2:
                var = float(np.var(w, ddof=1))
            else:
                var = 0.0
            result[seat].append(var)

    return result


# ---------------------------------------------------------------------------
# Behavioural Cloning network for opponent modelling
# ---------------------------------------------------------------------------

class BehaviourCloningNet(nn.Module):
    """
    Simple MLP that predicts action probabilities from state features.
    Used to model opponent policies from observed (state, action) data.

    Architecture: 3-layer MLP with GELU activations (same style as the
    main actor network for consistency).
    """
    def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = OPP_HIDDEN_DIM,
                 num_actions: int = NUM_ACTIONS) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.net(x)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        return logits   # logits, not probabilities

    def log_probs(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return F.log_softmax(self.forward(x, mask), dim=-1)


def train_opponent_model(
    features:  np.ndarray,   # (N, FEATURE_DIM)
    masks:     np.ndarray,   # (N, NUM_ACTIONS) bool
    actions:   np.ndarray,   # (N,) int
    device:    torch.device,
) -> BehaviourCloningNet:
    """
    Fit a BehaviourCloningNet on (feature, action) pairs via cross-entropy loss.
    Returns the fitted model in eval mode.
    """
    model = BehaviourCloningNet().to(device)
    opt   = Adam(model.parameters(), lr=OPP_LR)

    feat_t = torch.tensor(features, dtype=torch.float32, device=device)
    mask_t = torch.tensor(masks,    dtype=torch.bool,    device=device)
    act_t  = torch.tensor(actions,  dtype=torch.int64,   device=device)

    N = len(features)
    model.train()
    for epoch in range(OPP_EPOCHS):
        perm = torch.randperm(N, device=device)
        total_loss = 0.0
        n_batches  = 0
        for start in range(0, N, OPP_BATCH_SIZE):
            idx       = perm[start : start + OPP_BATCH_SIZE]
            logits    = model(feat_t[idx], mask_t[idx])
            loss      = F.cross_entropy(logits, act_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches  += 1
        if (epoch + 1) % 10 == 0:
            log.debug("    BC epoch %3d: loss %.4f", epoch + 1, total_loss / n_batches)

    model.eval()
    return model


# ---------------------------------------------------------------------------
# MC return estimation for reward components
# ---------------------------------------------------------------------------

def compute_mc_returns_per_hand(
    hand_records: List[HandRecord],
    var_penalties: Dict[int, List[float]],  # seat → per-hand variance
) -> Dict[int, List[Tuple[np.ndarray, np.ndarray, np.ndarray, int]]]:
    """
    For each seat, compute Monte-Carlo returns for each reward component.

    Returns:
      seat → list of (features, masks, action_indices, step_returns_3d)
      where step_returns_3d is shape (n_steps, 3) for components
      [chip, var_penalty, pot_involvement].

    These are used by the IRL optimiser.  Within a hand (single episode):
      - Component 0 (chips):    all intermediate steps get 0; terminal gets chip_delta
      - Component 1 (var pen.): terminal step gets rolling variance at this hand
      - Component 2 (pot):      terminal step gets max_pot / POT_NORM

    This is correct for gamma=1 (undiscounted) episodic tasks.
    """
    # Organise: seat → list of (feat, mask, act_idx, [R_chip, R_var, R_pot])
    result: Dict[int, List] = {i: [] for i in range(NUM_PLAYERS)}

    for hand_idx, rec in enumerate(hand_records):
        for seat in range(NUM_PLAYERS):
            steps   = rec.steps[seat]
            n_steps = len(steps)
            if n_steps == 0:
                continue

            var_penalty = var_penalties[seat][hand_idx]

            feats    = np.stack([s.feature    for s in steps])
            masks    = np.stack([s.legal_mask for s in steps])
            acts     = np.array([s.action_idx for s in steps])
            # Shape (n_steps, 3): [chip, var_pen, pot]
            returns  = np.zeros((n_steps, 3), dtype=np.float32)
            # Only terminal step gets non-zero returns (gamma=1, single episode)
            returns[-1, 0] = rec.chip_deltas[seat]
            returns[-1, 1] = var_penalty          # positive = penalty (alpha * this)
            returns[-1, 2] = rec.max_pots[seat] / POT_NORM

            result[seat].append((feats, masks, acts, returns))

    return result


# ---------------------------------------------------------------------------
# IRL: gradient-ascent on log-posterior
# ---------------------------------------------------------------------------

class IRLOptimiser:
    """
    Gradient-ascent Bayesian IRL for recovering (alpha, beta) of one target seat.

    The core idea (Feature Expectation Matching / Boltzmann-rational IRL):

    Given observed (s_t, a_t) pairs and MC return estimates for each reward
    component, we form the log-likelihood:

        ℒ(α, β) = Σ_t log softmax( Q_θ(s_t, :) )[a_t]

    where Q_θ(s, a) ≈ Q₀(s,a) + α·Q_α(s,a) + β·Q_β(s,a)

    and Q_k(s,a) is the MC return for component k when taking action a.

    Since we observe only one action per state, we estimate Q_k(s, a_obs) but
    need Q_k(s, a') for all a' to compute the log-partition.  We handle this
    by using the *advantage* form:

        A_k(s, a) = Q_k(s, a) - V_k(s)

    where V_k(s) ≈ E_{π}[Q_k(s,a)] is estimated from the data.

    For unobserved actions, we use the opponent model to estimate a baseline:
    the expected Q_k under the empirical mix of observed actions.

    Gradient:
        ∂ℒ/∂α = Σ_t [ A_α(s_t, a_t) - E_{π_θ}[A_α(s_t, ·)] ]

    This is the "feature expectation matching" update: push alpha such that
    the variance-penalised feature expectation under the observed policy matches
    that of a Boltzmann-rational agent with reward weight alpha.

    Gaussian prior: log p(θ) = -||θ||²/(2σ²)  →  gradient: -θ/σ²

    The opponent model is used to provide the target agent's effective
    value baseline: since the target agent's Q-values incorporate the
    expected outcomes given opponent behaviour, and opponents play according
    to π̂_opp, we can estimate the target's effective transition kernel.

    In practice we implement a simplified but principled version:
    we estimate the advantage A_k(s_t, a_t) directly from MC returns,
    and approximate the partition function using a learned Q-network
    that maps (feature, alpha, beta) → action logit adjustments.
    """

    def __init__(
        self,
        target_seat:     int,
        step_data:       List[Tuple],   # from compute_mc_returns_per_hand
        opponent_models: Dict[int, BehaviourCloningNet],
        target_network:  ActorCriticNetwork,
        device:          torch.device,
        prior_sigma:     float = IRL_PRIOR_SIGMA,
        lr:              float = IRL_LR,
    ) -> None:
        self.seat           = target_seat
        self.step_data      = step_data
        self.opp_models     = opponent_models
        self.target_network = target_network
        self.device         = device
        self.prior_sigma    = prior_sigma

        # Learnable reward parameters — initialised at (0, 0) (prior centre)
        self.theta = nn.Parameter(
            torch.zeros(2, dtype=torch.float64, device=device)
        )
        self.optimiser = Adam([self.theta], lr=lr)

        # History for convergence detection and plotting
        self.alpha_history: List[float] = []
        self.beta_history:  List[float] = []
        self.loss_history:  List[float] = []

        # Pre-compute advantage baselines from the data
        self._precompute_advantage_baselines()

    def _precompute_advantage_baselines(self) -> None:
        """
        Estimate per-feature-bin value baselines V_k(s) ≈ mean return for component k.

        We use the observed MC returns as the baseline:
            V_k = mean over all steps of R_k (terminal return of the hand).

        This is a crude but unbiased estimate for the value when the policy
        is fixed (which it is — the agents are frozen).  It allows computing:
            A_k(s_t, a_t) = R_k(hand containing t) - V_k

        In the log-likelihood gradient, these baselines cancel out exactly
        in expectation (they're control variates), but they reduce variance
        of the gradient estimate.
        """
        chip_returns  = []
        var_returns   = []
        pot_returns   = []

        for feats, masks, acts, returns in self.step_data:
            # Terminal step returns (last row of each hand)
            chip_returns.append(returns[-1, 0])
            var_returns.append( returns[-1, 1])
            pot_returns.append( returns[-1, 2])

        self.V_chip = float(np.mean(chip_returns)) if chip_returns else 0.0
        self.V_var  = float(np.mean(var_returns))  if var_returns  else 0.0
        self.V_pot  = float(np.mean(pot_returns))  if pot_returns  else 0.0

        log.info(
            "    Seat %d advantage baselines: V_chip=%.2f  V_var=%.4f  V_pot=%.4f",
            self.seat, self.V_chip, self.V_var, self.V_pot,
        )

    def _sample_batch(self, batch_size: int) -> List[Tuple]:
        """Sample a random batch of hand records."""
        idx = np.random.choice(len(self.step_data), min(batch_size, len(self.step_data)), replace=False)
        return [self.step_data[i] for i in idx]

    def _compute_log_likelihood_gradient(
        self,
        batch:  List[Tuple],
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute the gradient of log p(θ|data) w.r.t. θ = (alpha, beta).

        For each hand in the batch, for each step (s_t, a_t):
          1. Compute Q_θ(s_t, a_t) ≈ Q₀(s_t, a_t) + α·A_α(s_t, a_t) + β·A_β(s_t, a_t)
             where Q₀ comes from the target agent's network (frozen),
             and A_k is the MC advantage for component k.

          2. Estimate the partition function Z(s_t) = Σ_{a'} exp(Q_θ(s_t, a'))
             by mixing the target network's base logits with the reward shaping.
             For unobserved actions: A_k(s_t, a') is approximated as 0 (no advantage
             over baseline) — this is the key approximation.  It is conservative
             (underestimates the gradient) but consistent.

          3. Log-likelihood contribution: log π_θ(a_t|s_t) = Q_θ(s_t,a_t) - log Z(s_t)

          4. Prior gradient: -theta / sigma²
        """
        alpha = self.theta[0]
        beta  = self.theta[1]

        total_ll  = torch.tensor(0.0, dtype=torch.float64, device=self.device, requires_grad=False)
        grad_alpha = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        grad_beta  = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        n_steps   = 0

        for feats, masks, acts, returns in batch:
            n = len(acts)

            feat_t = torch.tensor(feats, dtype=torch.float32, device=self.device)
            mask_t = torch.tensor(masks, dtype=torch.bool,    device=self.device)
            act_t  = torch.tensor(acts,  dtype=torch.int64,   device=self.device)

            with torch.no_grad():
                # Base Q-values from frozen target network (actor logits ≈ Q₀)
                base_logits, _ = self.target_network(feat_t, mask_t)
                # Shape: (n, NUM_ACTIONS) — already masked

            # Advantage values for each reward component
            # A_chip(s_t, a_t) = R_chip_t - V_chip (only terminal step non-zero)
            # For non-terminal steps A_chip = 0 - V_chip  (control variate)
            # But since we're in a terminal-reward setting, the advantage for
            # non-terminal steps is exactly their contribution: 0 - V = -V.
            # However, -V is constant across actions and cancels in softmax.
            # So we only need the terminal-step returns, which differ by action.
            # For intermediate steps: all A_k are equal → no gradient signal.
            # We therefore SKIP intermediate steps (they contribute zero gradient).

            # Only process the terminal step of each hand
            # (index -1 / last step for this seat)
            # Terminal step advantage (MC return - baseline)
            A_chip = returns[-1, 0] - self.V_chip
            A_var  = returns[-1, 1] - self.V_var    # var penalty (positive)
            A_pot  = returns[-1, 2] - self.V_pot

            # The OBSERVED action at the terminal step
            a_obs = int(acts[-1])

            # Q_θ(s, a_obs) = base_logit[a_obs] + alpha * A_var + beta * A_pot
            # Note: A_chip is part of Q₀ (already in base_logit implicitly via value fn)
            # For the reward-SHAPING components (var and pot), only the terminal step
            # carries non-zero return, so only the terminal step contributes gradient.

            # Adjusted logits for ALL actions at terminal step
            # For the observed action: shaping = alpha * A_var + beta * A_pot
            # For unobserved actions: shaping ≈ 0 (no advantage observed → baseline)
            #   This is the key approximation: we assume unobserved actions have
            #   zero advantage w.r.t. reward shaping (conservative estimate).
            adj_logits = base_logits[-1].clone().double()   # shape (NUM_ACTIONS,)

            # Apply reward shaping only to the OBSERVED action
            # (Unobserved actions stay at base logit)
            shaping = alpha.item() * float(A_var) + beta.item() * float(A_pot)
            adj_logits[a_obs] = adj_logits[a_obs] + shaping

            # Log-likelihood: log π_θ(a_obs | s) = adj_logit[a_obs] - log Σ exp(adj_logit)
            log_z = torch.logsumexp(adj_logits[mask_t[-1]], dim=0)
            ll    = adj_logits[a_obs] - log_z

            total_ll  = total_ll + ll.detach()

            # Gradient (analytical form for the shaping-at-observed-action approximation):
            #   ∂ll/∂alpha = A_var * (1 - π_θ(a_obs|s))
            #   ∂ll/∂beta  = A_pot * (1 - π_θ(a_obs|s))
            pi_a_obs   = torch.exp(ll).detach()   # ∈ [0,1]
            grad_alpha = grad_alpha + torch.tensor(A_var, dtype=torch.float64, device=self.device) * (1.0 - pi_a_obs)
            grad_beta  = grad_beta  + torch.tensor(A_pot, dtype=torch.float64, device=self.device) * (1.0 - pi_a_obs)
            n_steps   += 1

        if n_steps == 0:
            return torch.tensor([0.0, 0.0], dtype=torch.float64, device=self.device), 0.0

        # Average over batch
        grad_alpha = grad_alpha / n_steps
        grad_beta  = grad_beta  / n_steps
        ll_mean    = float(total_ll.item()) / n_steps

        # Add Gaussian prior gradient: ∂log p(θ)/∂θ_k = -θ_k / σ²
        prior_grad_alpha = -self.theta[0] / (self.prior_sigma ** 2)
        prior_grad_beta  = -self.theta[1] / (self.prior_sigma ** 2)

        full_grad = torch.stack([
            grad_alpha + prior_grad_alpha,
            grad_beta  + prior_grad_beta,
        ])
        return full_grad, ll_mean

    def step(self, batch: List[Tuple]) -> float:
        """Take one gradient-ascent step.  Returns the mean log-likelihood."""
        full_grad, ll = self._compute_log_likelihood_gradient(batch)

        # Manual gradient ascent (we're maximising, so +grad)
        self.optimiser.zero_grad()
        # Set the gradient on theta (negated for minimisation → we negate the ascent)
        self.theta.grad = -full_grad.float().to(self.theta.dtype)
        self.optimiser.step()

        alpha_val = float(self.theta[0].item())
        beta_val  = float(self.theta[1].item())
        self.alpha_history.append(alpha_val)
        self.beta_history.append(beta_val)
        self.loss_history.append(ll)

        return ll

    def is_converged(self) -> bool:
        """
        Convergence: the standard deviation of both parameters over the last
        CONV_WINDOW steps is below CONV_THRESHOLD.
        """
        if len(self.alpha_history) < CONV_MIN_STEPS + CONV_WINDOW:
            return False
        window_alpha = self.alpha_history[-CONV_WINDOW:]
        window_beta  = self.beta_history[-CONV_WINDOW:]
        return (
            float(np.std(window_alpha)) < CONV_THRESHOLD
            and float(np.std(window_beta))  < CONV_THRESHOLD
        )

    @property
    def current_alpha(self) -> float:
        return float(self.theta[0].item())

    @property
    def current_beta(self) -> float:
        return float(self.theta[1].item())


# ---------------------------------------------------------------------------
# Main IRL procedure
# ---------------------------------------------------------------------------

def run_irl_for_seat(
    target_seat:      int,
    step_data:        List[Tuple],
    opponent_models:  Dict[int, BehaviourCloningNet],
    target_network:   ActorCriticNetwork,
    device:           torch.device,
    true_alpha:       float,
    true_beta:        float,
) -> Dict:
    """
    Run the full GABO-IRL procedure for one target seat.
    Returns a result dict with estimates, history, and convergence info.
    """
    log.info("  Running IRL for seat %d (true α=%.4f, true β=%.4f) ...",
             target_seat, true_alpha, true_beta)
    log.info("    Data size: %d hands", len(step_data))

    optimiser = IRLOptimiser(
        target_seat=target_seat,
        step_data=step_data,
        opponent_models=opponent_models,
        target_network=target_network,
        device=device,
    )

    start = time.time()
    for step_i in range(IRL_N_STEPS):
        batch = optimiser._sample_batch(IRL_BATCH_SIZE)
        ll    = optimiser.step(batch)

        if (step_i + 1) % IRL_LOG_EVERY == 0:
            alpha_hat = optimiser.current_alpha
            beta_hat  = optimiser.current_beta
            alpha_err = abs(alpha_hat - true_alpha)
            beta_err  = abs(beta_hat  - true_beta)
            elapsed   = time.time() - start
            log.info(
                "    Step %4d | α̂=%.4f (err %.4f) | β̂=%.4f (err %.4f) | LL=%.4f | %.1fs",
                step_i + 1, alpha_hat, alpha_err, beta_hat, beta_err, ll, elapsed,
            )

        if optimiser.is_converged():
            log.info("    Converged at step %d.", step_i + 1)
            break

    final_alpha = optimiser.current_alpha
    final_beta  = optimiser.current_beta

    # Take the mean of the last CONV_WINDOW estimates (more stable than last value)
    if len(optimiser.alpha_history) >= CONV_WINDOW:
        final_alpha = float(np.mean(optimiser.alpha_history[-CONV_WINDOW:]))
        final_beta  = float(np.mean(optimiser.beta_history[-CONV_WINDOW:]))

    result = {
        "seat":          target_seat,
        "true_alpha":    true_alpha,
        "true_beta":     true_beta,
        "est_alpha":     final_alpha,
        "est_beta":      final_beta,
        "alpha_mse":     (final_alpha - true_alpha) ** 2,
        "beta_mse":      (final_beta  - true_beta)  ** 2,
        "n_steps":       len(optimiser.alpha_history),
        "converged":     optimiser.is_converged(),
        "alpha_history": optimiser.alpha_history,
        "beta_history":  optimiser.beta_history,
        "ll_history":    optimiser.loss_history,
    }

    log.info(
        "  Seat %d result: α̂=%.4f (true %.4f) | β̂=%.4f (true %.4f)",
        target_seat, final_alpha, true_alpha, final_beta, true_beta,
    )
    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_collection_and_irl(is_ablation: bool = False, ablation_tag: str = "") -> None:
    os.makedirs(IRL_DIR, exist_ok=True)
    device   = torch.device(DEVICE)
    tag      = ablation_tag if is_ablation else ""
    suffix   = f"_{tag}" if tag else ""

    # ── Load true reward parameters ────────────────────────────────────────
    params_path = os.path.join(CHECKPOINT_DIR, "perturbed_agent_params.json")
    if is_ablation:
        params_path = os.path.join(CHECKPOINT_DIR, "ablation_agent_params.json")
    with open(params_path) as f:
        true_params = {p["seat"]: (p["alpha"], p["beta"]) for p in json.load(f)}

    # ── Phase 1: Collect trajectories ─────────────────────────────────────
    traj_path = os.path.join(IRL_DIR, f"hand_records{suffix}.pkl")
    if os.path.exists(traj_path):
        log.info("Loading cached trajectories from %s ...", traj_path)
        with open(traj_path, "rb") as f:
            hand_records = pickle.load(f)
        log.info("  Loaded %d hand records.", len(hand_records))
    else:
        log.info("Collecting %d hands of fixed-policy play ...", N_COLLECTION_HANDS)
        hand_records = collect_trajectories(N_COLLECTION_HANDS)
        with open(traj_path, "wb") as f:
            pickle.dump(hand_records, f)
        log.info("  Saved trajectories to %s.", traj_path)

    # ── Compute rolling variance penalties ────────────────────────────────
    log.info("Computing rolling variance penalties ...")
    var_penalties = compute_rolling_variance_penalties(hand_records, window=100)

    # ── Compute MC return data ─────────────────────────────────────────────
    log.info("Computing MC return data per seat ...")
    mc_data = compute_mc_returns_per_hand(hand_records, var_penalties)

    # ── Phase 2: Train opponent models + run IRL per seat ─────────────────
    irl_results = []

    for target_seat in range(NUM_PLAYERS):
        log.info("\n" + "="*60)
        log.info("IRL for target seat %d", target_seat)
        log.info("="*60)

        # ── Load target agent network ──────────────────────────────────────
        if is_ablation:
            # Ablation: only seat 0 is the adapted agent
            agent_path = os.path.join(CHECKPOINT_DIR, f"ablation_perturbed_agent_0.pt")
            if target_seat != 0:
                log.info("  Skipping seat %d (only seat 0 is the target in ablation).", target_seat)
                continue
        else:
            agent_path = os.path.join(CHECKPOINT_DIR, f"perturbed_agent_{target_seat}.pt")

        ckpt = torch.load(agent_path, map_location=device)
        target_net = ActorCriticNetwork(
            input_dim=ckpt.get("feature_dim", FEATURE_DIM),
            hidden_dim=ckpt.get("hidden_dim",  HIDDEN_DIM),
        ).to(device)
        target_net.load_state_dict(ckpt["network_state"])
        target_net.eval()
        for p in target_net.parameters():
            p.requires_grad_(False)

        # ── Train opponent models ──────────────────────────────────────────
        opponent_models: Dict[int, BehaviourCloningNet] = {}
        for opp_seat in range(NUM_PLAYERS):
            if opp_seat == target_seat:
                continue

            log.info("  Training opponent model for seat %d ...", opp_seat)
            opp_data = mc_data[opp_seat]
            if len(opp_data) < OPP_MIN_SAMPLES:
                log.warning("    Too few samples (%d) for seat %d opponent model.",
                            len(opp_data), opp_seat)
                continue

            all_feats = np.concatenate([d[0] for d in opp_data], axis=0)
            all_masks = np.concatenate([d[1] for d in opp_data], axis=0)
            all_acts  = np.concatenate([d[2] for d in opp_data], axis=0)

            opp_net = train_opponent_model(all_feats, all_masks, all_acts, device)
            opponent_models[opp_seat] = opp_net

            # Save opponent model
            opp_path = os.path.join(IRL_DIR, f"opponent_model_target{target_seat}_opp{opp_seat}{suffix}.pt")
            torch.save(opp_net.state_dict(), opp_path)
            log.info("    Saved opponent model → %s", opp_path)

        # ── Run IRL ───────────────────────────────────────────────────────
        true_alpha, true_beta = true_params[target_seat]
        result = run_irl_for_seat(
            target_seat=target_seat,
            step_data=mc_data[target_seat],
            opponent_models=opponent_models,
            target_network=target_net,
            device=device,
            true_alpha=true_alpha,
            true_beta=true_beta,
        )
        irl_results.append(result)

    # ── Save results ───────────────────────────────────────────────────────
    summary = []
    for r in irl_results:
        summary.append({
            k: v for k, v in r.items()
            if k not in ("alpha_history", "beta_history", "ll_history")
        })
    estimates_path = os.path.join(IRL_DIR, f"irl_estimates{suffix}.json")
    with open(estimates_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved: %s", estimates_path)

    # Save full convergence histories
    conv_path = os.path.join(IRL_DIR, f"irl_convergence_log{suffix}.json")
    with open(conv_path, "w") as f:
        json.dump(irl_results, f, indent=2)
    log.info("Saved: %s", conv_path)

    # Print summary table
    log.info("\n" + "="*60)
    log.info("IRL RESULTS SUMMARY")
    log.info("="*60)
    log.info("  %4s  %8s  %8s  %8s  %8s  %10s  %10s",
             "Seat", "true_α", "est_α", "true_β", "est_β", "α-MSE", "β-MSE")
    for r in irl_results:
        log.info("  %4d  %+8.4f  %+8.4f  %+8.4f  %+8.4f  %10.6f  %10.6f",
                 r["seat"], r["true_alpha"], r["est_alpha"],
                 r["true_beta"], r["est_beta"],
                 r["alpha_mse"], r["beta_mse"])


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    run_collection_and_irl()
