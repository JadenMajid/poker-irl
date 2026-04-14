"""
reward.py
---------
Reward functions for the multi-agent IRL poker project.

Each agent has a personalised reward function parameterised by:
  alpha : risk-aversion coefficient  (higher → penalise variance in chip outcomes)
  beta  : pot-involvement bias       (higher → reward getting involved in big pots)

The canonical reward is:
    R = E[chips] - alpha * Var(chips) + beta * pot_involvement_bonus(hand)

Since poker hands are one-shot episodes (not sequential MDPs in the conventional
sense), the "E[chips]" term reduces to the actual net chip change for that hand.
The variance term is approximated across a rolling window of recent hands, making
it a proper measure of result volatility rather than just within-hand variance.

This module provides:
  - RewardParams            : (alpha, beta) container with serialisation helpers.
  - RewardFunction          : Callable reward computer given a hand outcome and context.
  - RollingRewardTracker    : Maintains a running history and computes the full reward
                              including the variance penalty term.
  - NeutralRewardParams     : alpha=0, beta=0 — used for base agent training.

Design notes:
  - The variance penalty uses a **rolling window** (default 100 hands) so that:
      (a) it is computable during training without requiring full episode storage, and
      (b) it captures behavioural variance over the agent's recent history, which is
          what matters for IRL recovery (an agent consistently choosing low-variance
          lines will have low rolling variance regardless of individual outcomes).
  - The pot_involvement_bonus is computed per-hand as:
        bonus = max_pot_level_reached / POT_NORM
    where max_pot_level_reached is the largest pot size at the moment the agent
    either called or raised — representing how big a pot they voluntarily
    committed to.  Folding before the flop yields a very small bonus; calling
    a big river bet yields a large one.
  - We deliberately keep the reward function differentiable with respect to
    (alpha, beta) so that gradient-based IRL methods can optimise these directly.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np

from game_state import (
    ActionType,
    BetRecord,
    HandTrajectory,
    NUM_PLAYERS,
    TrajectoryStep,
)


# ---------------------------------------------------------------------------
# Normalisation constant (matches feature_encoder.py)
# ---------------------------------------------------------------------------

POT_NORM = 2000.0


# ---------------------------------------------------------------------------
# Reward parameters
# ---------------------------------------------------------------------------

@dataclass
class RewardParams:
    """
    (alpha, beta) reward parameterisation for one agent.

    Attributes
    ----------
    alpha : Risk-aversion coefficient.  alpha=0 → risk-neutral (maximise EV).
            alpha > 0 → agent penalises variance in chip returns.
            Typical range: [0.0, 1.0].
    beta  : Pot-involvement bias.  beta=0 → agent ignores pot size.
            beta > 0 → agent receives a bonus for committing to large pots
            (encouraging aggressive / pot-building behaviour).
            Typical range: [0.0, 2.0].
    """
    alpha: float = 0.0
    beta:  float = 0.0

    def __post_init__(self) -> None:
        # alpha and beta may be negative to represent risk-seeking / pot-avoidant agents.
        # No validation constraints on sign — the IRL recovery experiment requires
        # agents that span all four quadrants of (alpha, beta) space.
        pass

    def to_array(self) -> np.ndarray:
        """Serialise to a 2-float numpy array for optimisation."""
        return np.array([self.alpha, self.beta], dtype=np.float64)

    @staticmethod
    def from_array(arr: np.ndarray) -> "RewardParams":
        """Deserialise from a 2-float numpy array."""
        arr = np.asarray(arr, dtype=np.float64)
        if arr.shape != (2,):
            raise ValueError(f"Expected shape (2,), got {arr.shape}.")
        return RewardParams(alpha=float(arr[0]), beta=float(arr[1]))

    def __repr__(self) -> str:
        return f"RewardParams(alpha={self.alpha:.4f}, beta={self.beta:.4f})"

    def perturb(self, delta_alpha: float = 0.0, delta_beta: float = 0.0) -> "RewardParams":
        """Return a new RewardParams with small perturbations (clamped ≥ 0)."""
        return RewardParams(
            alpha=max(0.0, self.alpha + delta_alpha),
            beta =max(0.0, self.beta  + delta_beta),
        )


# Canonical neutral params used for base-agent training
NeutralRewardParams = RewardParams(alpha=0.0, beta=0.0)


# ---------------------------------------------------------------------------
# Per-hand reward components
# ---------------------------------------------------------------------------

@dataclass
class HandRewardComponents:
    """
    Decomposition of the reward into its constituent parts for one hand.

    Useful for debugging, logging, and IRL gradient computation.

    Attributes
    ----------
    chip_delta          : Net chips won/lost this hand.
    variance_penalty    : alpha * rolling_variance (positive → penalty applied).
    pot_involvement_bonus: beta * normalised_max_pot_commitment.
    total               : The scalar reward fed to the RL/IRL system.
    rolling_variance    : The variance estimate used this step (for logging).
    max_pot_commitment  : The largest pot the agent actively committed chips into.
    """
    chip_delta:            float
    variance_penalty:      float
    pot_involvement_bonus: float
    total:                 float
    rolling_variance:      float
    max_pot_commitment:    float


# ---------------------------------------------------------------------------
# Pot involvement computation from trajectory
# ---------------------------------------------------------------------------

def compute_pot_involvement(
    trajectory: HandTrajectory,
    seat: int,
) -> float:
    """
    Extract the maximum pot size at which player *seat* actively committed chips
    (called or raised) during this hand.

    Returns 0.0 if the player folded before committing anything beyond blinds,
    or if they were never involved (shouldn't happen in a normal hand).

    Rationale:
      - Calling/raising into a large pot signals pot-building intent.
      - Folding preflop yields near-zero involvement even if the pot was large.
      - This is asymmetric: the player gets credit for the pot size *at the
        moment they committed*, rewarding late-street aggression most heavily.
    """
    max_pot = 0.0
    for step in trajectory.steps_for_player(seat):
        action = step.action
        if action.action_type in (ActionType.CALL, ActionType.RAISE):
            pot_at_action = step.observation.pot
            if pot_at_action > max_pot:
                max_pot = float(pot_at_action)
    return max_pot


# ---------------------------------------------------------------------------
# Rolling variance tracker
# ---------------------------------------------------------------------------

class RollingVarianceTracker:
    """
    Maintains a sliding window of chip deltas and computes their sample variance.

    Implements Welford's online algorithm for numerical stability.

    Parameters
    ----------
    window_size : Number of recent hands to include in the variance estimate.
                  Smaller windows → more reactive but noisier; larger → smoother.
    """

    def __init__(self, window_size: int = 100) -> None:
        self._window_size = window_size
        self._history: Deque[float] = deque(maxlen=window_size)

    def update(self, chip_delta: float) -> None:
        """Push a new chip delta observation into the window."""
        self._history.append(chip_delta)

    def variance(self) -> float:
        """
        Return the sample variance of chip deltas in the current window.
        Returns 0.0 if fewer than 2 observations are present.
        """
        n = len(self._history)
        if n < 2:
            return 0.0
        data = list(self._history)
        mean = sum(data) / n
        var  = sum((x - mean) ** 2 for x in data) / (n - 1)  # Bessel-corrected
        return var

    def std(self) -> float:
        return math.sqrt(self.variance())

    @property
    def num_observations(self) -> int:
        return len(self._history)

    def reset(self) -> None:
        self._history.clear()


# ---------------------------------------------------------------------------
# Reward function — main class
# ---------------------------------------------------------------------------

class RewardFunction:
    """
    Computes the per-hand scalar reward for one agent given a HandTrajectory.

    The reward function is fully determined by a RewardParams object:

        R = chip_delta
            - alpha * rolling_var(chip_deltas)
            + beta  * (max_pot_committed / POT_NORM)

    The variance term uses a rolling window maintained internally, so this
    object must be called once per hand in order (it accumulates history).
    Call .reset() to clear the history between experiment runs.

    Parameters
    ----------
    params         : The (alpha, beta) reward parameters for this agent.
    variance_window: Rolling window size for variance estimation.
    """

    def __init__(
        self,
        params:          RewardParams,
        variance_window: int = 100,
    ) -> None:
        self._params  = params
        self._tracker = RollingVarianceTracker(window_size=variance_window)

    @property
    def params(self) -> RewardParams:
        return self._params

    @params.setter
    def params(self, new_params: RewardParams) -> None:
        """Allow updating params (e.g. during IRL optimisation)."""
        self._params = new_params

    def compute(
        self,
        trajectory: HandTrajectory,
        seat:       int,
    ) -> HandRewardComponents:
        """
        Compute the reward for player *seat* from a completed hand trajectory.

        Parameters
        ----------
        trajectory : The full hand trajectory (all players, all actions).
        seat       : Which player's reward to compute.

        Returns
        -------
        HandRewardComponents with all decomposed fields populated.
        """
        # --- Chip delta ---
        chip_delta = float(trajectory.final_chip_deltas.get(seat, 0))

        # --- Update rolling variance with this outcome ---
        self._tracker.update(chip_delta)
        rolling_var = self._tracker.variance()

        # --- Pot involvement ---
        max_pot = compute_pot_involvement(trajectory, seat)

        # --- Compose reward ---
        variance_penalty = self._params.alpha * rolling_var
        pot_bonus        = self._params.beta  * (max_pot / POT_NORM)

        total = chip_delta - variance_penalty + pot_bonus

        return HandRewardComponents(
            chip_delta=chip_delta,
            variance_penalty=variance_penalty,
            pot_involvement_bonus=pot_bonus,
            total=total,
            rolling_variance=rolling_var,
            max_pot_commitment=max_pot,
        )

    def scalar_reward(self, trajectory: HandTrajectory, seat: int) -> float:
        """Convenience method: compute and return just the scalar total reward."""
        return self.compute(trajectory, seat).total

    def reset(self) -> None:
        """Clear rolling history (call between experiment episodes)."""
        self._tracker.reset()


# ---------------------------------------------------------------------------
# Reward evaluator for IRL — stateless, given explicit variance estimate
# ---------------------------------------------------------------------------

def compute_reward_stateless(
    params:       RewardParams,
    chip_delta:   float,
    rolling_var:  float,
    max_pot:      float,
) -> float:
    """
    Compute the reward for given scalar inputs without any stateful tracking.

    Used by IRL methods that sweep over candidate (alpha, beta) values and need
    to evaluate the reward for many parameter combinations quickly.

    Parameters
    ----------
    params      : Candidate reward parameters.
    chip_delta  : Net chips for this hand.
    rolling_var : Pre-computed rolling variance estimate.
    max_pot     : Maximum pot the player committed into this hand.
    """
    return (
        chip_delta
        - params.alpha * rolling_var
        + params.beta  * (max_pot / POT_NORM)
    )


def reward_gradient_wrt_params(
    chip_delta:  float,
    rolling_var: float,
    max_pot:     float,
) -> np.ndarray:
    """
    Gradient of the reward with respect to (alpha, beta).

        dR/d_alpha = -rolling_var
        dR/d_beta  =  max_pot / POT_NORM

    Returns a (2,) numpy array.  Useful for gradient-based IRL.
    """
    return np.array([
        -rolling_var,
        max_pot / POT_NORM,
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# KL-regularised reward for fine-tuning agents from base policy
# ---------------------------------------------------------------------------

@dataclass
class RegularisedRewardConfig:
    """
    Configuration for reward regularisation during the agent fine-tuning phase.

    After training a neutral base agent, each agent is fine-tuned with their
    personalised (alpha, beta) reward.  To prevent them drifting too far from
    the base policy (and thus staying in a region where their behaviour is
    still "poker-like"), we add a KL penalty:

        R_regularised = R_personalised - kl_coeff * KL(pi || pi_base)

    This is implemented as a callback the RL trainer calls after each batch.
    The KL divergence is computed by the RL trainer (it has access to the
    base policy logits), not by this module.  This dataclass just carries the
    configuration.
    """
    reward_params:  RewardParams
    kl_coeff:       float = 0.01    # coefficient on the KL penalty
    kl_anneal_rate: float = 0.99    # multiply kl_coeff by this each epoch
    min_kl_coeff:   float = 0.0001  # floor for kl_coeff after annealing

    def anneal(self) -> None:
        """Reduce the KL coefficient by one annealing step."""
        self.kl_coeff = max(self.min_kl_coeff, self.kl_coeff * self.kl_anneal_rate)
