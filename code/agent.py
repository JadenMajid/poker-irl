"""
agent.py
--------
Neural network architecture and agent class for multi-agent poker IRL.

Architecture overview
---------------------
Each agent maintains a single ActorCriticNetwork with:

  Trunk (shared):
    - Input: fixed-length feature vector from FeatureEncoder (165 floats)
    - 3 hidden layers with LayerNorm + GELU activation
    - Residual connections on hidden layers 2 and 3

  Actor head (policy):
    - Projects from trunk to action logits
    - Actions: FOLD(0), CALL(1), RAISE_20(2), RAISE_100(3), RAISE_500(4)
    - When a player cannot raise (has already raised this street), the raise
      logits are masked to −∞ before softmax so the policy still outputs a
      valid distribution.

  Critic head (value):
    - Projects from trunk to a scalar state-value estimate V(s)
    - Used by the RL trainer (PPO); not needed for IRL inference.

Design choices and rationale:
  1. Shared trunk avoids duplicate computation and encourages the actor and critic
     to share useful representations (standard in PPO implementations).
  2. LayerNorm (not BatchNorm) is used because we operate with small, variable
     batch sizes during online RL updates.
  3. GELU activation tends to outperform ReLU on policy-gradient tasks (smoother
     gradients, no dead-neuron problem).
  4. Residual connections prevent gradient vanishing for the deeper trunk layers
     and make it easier for the network to learn near-identity functions when
     features are already informative.
  5. Fixed action space of 5 actions keeps the interface simple and explicit.
     The environment guarantees that at most {FOLD, CALL, RAISE_20, RAISE_100,
     RAISE_500} are ever needed.
  6. The agent's act() method is synchronous (no batching internally), which
     keeps the environment simple.  Batched inference is handled by the RL
     trainer when it collects rollouts.

IRL compatibility:
  - act_with_log_prob() returns both the action and log π(a|s), which is the
    primary quantity consumed by maximum-likelihood IRL.
  - action_log_probs() evaluates log π(a|s) for an externally supplied action,
    enabling importance sampling and trajectory-level likelihood computation.
  - The network is a standard nn.Module and can be checkpointed / loaded via
    PyTorch's standard mechanisms.
"""

from __future__ import annotations

import copy
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from cards import Card
from feature_encoder import FeatureEncoder, FEATURE_DIM
from game_state import (
    Action,
    ActionType,
    FIXED_RAISE_SIZES,
    PlayerObservation,
)
from reward import RewardFunction, RewardParams


# ---------------------------------------------------------------------------
# Action space constants
# ---------------------------------------------------------------------------

# Index → (ActionType, raise_amount)
ACTION_INDEX_TO_SPEC: Dict[int, Tuple[ActionType, int]] = {
    0: (ActionType.FOLD,  0),
    1: (ActionType.CALL,  0),
    2: (ActionType.RAISE, FIXED_RAISE_SIZES[0]),   # 20
    3: (ActionType.RAISE, FIXED_RAISE_SIZES[1]),   # 100
    4: (ActionType.RAISE, FIXED_RAISE_SIZES[2]),   # 500
}
NUM_ACTIONS = len(ACTION_INDEX_TO_SPEC)   # 5


def action_to_index(action: Action) -> int:
    """Convert an Action object to its integer index in ACTION_INDEX_TO_SPEC."""
    for idx, (atype, ramt) in ACTION_INDEX_TO_SPEC.items():
        if action.action_type == atype and action.raise_amount == ramt:
            return idx
    raise ValueError(f"Action {action} does not map to any known index.")


def index_to_action(idx: int, seat: int) -> Action:
    """Convert an action index back to an Action for *seat*."""
    if idx not in ACTION_INDEX_TO_SPEC:
        raise ValueError(f"Action index {idx} is out of range [0, {NUM_ACTIONS}).")
    atype, ramt = ACTION_INDEX_TO_SPEC[idx]
    return Action(atype, seat, raise_amount=ramt)


def legal_action_mask(obs: PlayerObservation) -> torch.Tensor:
    """
    Build a boolean mask of shape (NUM_ACTIONS,) where True = legal.
    Used to zero out illegal action logits before sampling.
    """
    legal_types = {a.action_type for a in obs.legal_actions}
    mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
    for idx, (atype, _) in ACTION_INDEX_TO_SPEC.items():
        if atype in legal_types:
            mask[idx] = True
    return mask


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    A two-layer MLP block with a residual skip connection.
    Input and output have the same dimension *dim*.
    """
    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln1  = nn.LayerNorm(dim)
        self.fc1  = nn.Linear(dim, dim)
        self.ln2  = nn.LayerNorm(dim)
        self.fc2  = nn.Linear(dim, dim)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.drop(F.gelu(self.fc1(self.ln1(x))))
        out = self.fc2(self.ln2(out))
        return F.gelu(out + residual)


class ActorCriticNetwork(nn.Module):
    """
    Shared trunk + actor head + critic head.

    Parameters
    ----------
    input_dim   : Feature vector size (FEATURE_DIM from feature_encoder.py).
    hidden_dim  : Width of all hidden layers.
    num_actions : Number of discrete actions (NUM_ACTIONS = 5).
    dropout     : Dropout probability (applied in residual blocks during training).
    """

    def __init__(
        self,
        input_dim:   int = FEATURE_DIM,
        hidden_dim:  int = 256,
        num_actions: int = NUM_ACTIONS,
        dropout:     float = 0.05,
    ) -> None:
        super().__init__()
        self.input_dim   = input_dim
        self.hidden_dim  = hidden_dim
        self.num_actions = num_actions

        # --- Input projection ---
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # --- Trunk: 2 residual blocks ---
        self.trunk = nn.Sequential(
            ResidualBlock(hidden_dim, dropout=dropout),
            ResidualBlock(hidden_dim, dropout=dropout),
        )

        # --- Actor head ---
        self.actor_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )

        # --- Critic head ---
        self.critic_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Weight initialisation — small actor output → near-uniform initial policy
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor_head[-1].bias)
        nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.0)
        nn.init.zeros_(self.critic_head[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        features    : Float tensor of shape (batch, input_dim) or (input_dim,).
        action_mask : Boolean tensor of shape (batch, num_actions) or (num_actions,).
                      True where actions are legal.  If None, all actions are legal.

        Returns
        -------
        logits : Action logits of shape (batch, num_actions)  (masked).
        value  : State value of shape (batch, 1).
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)
            squeeze  = True
        else:
            squeeze = False

        trunk_out = self.trunk(self.input_proj(features))
        logits    = self.actor_head(trunk_out)
        value     = self.critic_head(trunk_out)

        # Apply mask: set illegal actions to large negative value before softmax
        if action_mask is not None:
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0).expand_as(logits)
            logits = logits.masked_fill(~action_mask, float("-inf"))

        if squeeze:
            logits = logits.squeeze(0)
            value  = value.squeeze(0)

        return logits, value

    def policy_distribution(
        self,
        features: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Categorical:
        """Return a Categorical distribution over actions (masked)."""
        logits, _ = self.forward(features, action_mask)
        return Categorical(logits=logits)

    def value_estimate(self, features: torch.Tensor) -> torch.Tensor:
        """Return just the scalar value estimate (for PPO critic loss)."""
        _, value = self.forward(features)
        return value


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class PokerAgent:
    """
    A poker-playing agent encapsulating:
      - A unique seat index.
      - A neural network (ActorCriticNetwork) for decision-making.
      - A FeatureEncoder for converting observations to tensors.
      - A RewardFunction parameterised by (alpha, beta).

    The agent is the primary interface between the PokerEnv and the RL/IRL
    training loops.  It exposes:
      - act(obs)                → Action  (for environment step)
      - act_with_log_prob(obs)  → (Action, log_prob)  (for PPO loss)
      - action_log_probs(obs, action) → log_prob  (for IRL likelihood)
      - action_probs(obs)       → np.ndarray of shape (5,)  (full distribution)

    Parameters
    ----------
    seat            : Seat index in [0, NUM_PLAYERS).
    reward_params   : (alpha, beta) personalised reward parameters.
    hidden_dim      : Width of the neural network hidden layers.
    device          : Torch device ("cpu" or "cuda").
    deterministic   : If True, act() always picks the argmax action (for evaluation).
                      If False (default), samples stochastically (for training).
    """

    def __init__(
        self,
        seat:            int,
        reward_params:   RewardParams,
        hidden_dim:      int = 256,
        device:          str = "cpu",
        deterministic:   bool = False,
    ) -> None:
        self.seat          = seat
        self.device        = torch.device(device)
        self.deterministic = deterministic

        self._encoder    = FeatureEncoder()
        self._network    = ActorCriticNetwork(
            input_dim=FEATURE_DIM,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self._reward_fn  = RewardFunction(params=reward_params)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def network(self) -> ActorCriticNetwork:
        return self._network

    @property
    def reward_params(self) -> RewardParams:
        return self._reward_fn.params

    @reward_params.setter
    def reward_params(self, new_params: RewardParams) -> None:
        self._reward_fn.params = new_params

    @property
    def reward_function(self) -> RewardFunction:
        return self._reward_fn

    @property
    def encoder(self) -> FeatureEncoder:
        return self._encoder

    # ------------------------------------------------------------------
    # Observation → feature tensor
    # ------------------------------------------------------------------

    def _obs_to_tensor(self, obs: PlayerObservation) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (feature_tensor, action_mask_tensor) on the agent's device."""
        feat = self._encoder.encode(obs)
        feat_t = torch.from_numpy(feat).to(self.device)
        mask   = legal_action_mask(obs).to(self.device)
        return feat_t, mask

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(self, obs: PlayerObservation) -> Action:
        """
        Select an action for the given observation.

        Called by the PokerEnv during a hand.  Operates under torch.no_grad().
        """
        with torch.no_grad():
            feat_t, mask = self._obs_to_tensor(obs)
            logits, _    = self._network(feat_t, mask)
            if self.deterministic:
                action_idx = int(logits.argmax().item())
            else:
                dist       = Categorical(logits=logits)
                action_idx = int(dist.sample().item())
        return index_to_action(action_idx, self.seat)

    def act_with_log_prob(
        self,
        obs: PlayerObservation,
    ) -> Tuple[Action, torch.Tensor]:
        """
        Sample an action and return its log-probability.

        Used during PPO rollout collection.

        Returns
        -------
        action   : The sampled Action.
        log_prob : Scalar tensor: log π(action | obs).
        """
        feat_t, mask = self._obs_to_tensor(obs)
        logits, _    = self._network(feat_t, mask)
        dist         = Categorical(logits=logits)
        action_idx_t = dist.sample()
        log_prob     = dist.log_prob(action_idx_t)
        action       = index_to_action(int(action_idx_t.item()), self.seat)
        return action, log_prob

    def action_log_probs(
        self,
        obs:    PlayerObservation,
        action: Action,
    ) -> torch.Tensor:
        """
        Evaluate log π(action | obs) for a given (observation, action) pair.

        This is the primary quantity used by maximum-likelihood IRL:
            L(theta) = sum_t log π_theta(a_t | s_t)

        Parameters
        ----------
        obs    : The observation at the decision point.
        action : The action whose log-probability to evaluate.

        Returns
        -------
        Scalar tensor: log π(action | obs).  Requires grad if network params do.
        """
        feat_t, mask = self._obs_to_tensor(obs)
        logits, _    = self._network(feat_t, mask)
        dist         = Categorical(logits=logits)
        action_idx   = action_to_index(action)
        return dist.log_prob(torch.tensor(action_idx, device=self.device))

    def action_probs(self, obs: PlayerObservation) -> np.ndarray:
        """
        Return the full action probability vector as a numpy array of shape (5,).
        Useful for logging, visualisation, and IRL diagnostics.
        """
        with torch.no_grad():
            feat_t, mask = self._obs_to_tensor(obs)
            logits, _    = self._network(feat_t, mask)
            probs        = F.softmax(logits, dim=-1)
        return probs.cpu().numpy()

    def value_estimate(self, obs: PlayerObservation) -> float:
        """Return the critic's value estimate V(s) for the given observation."""
        with torch.no_grad():
            feat_t, _ = self._obs_to_tensor(obs)
            val       = self._network.value_estimate(feat_t)
        return float(val.item())

    # ------------------------------------------------------------------
    # Batch inference (used by RL trainers for efficiency)
    # ------------------------------------------------------------------

    def batch_forward(
        self,
        obs_list:   List[PlayerObservation],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run forward pass on a batch of observations.

        Parameters
        ----------
        obs_list : List of PlayerObservation objects.

        Returns
        -------
        logits : Tensor of shape (batch, NUM_ACTIONS) — masked.
        values : Tensor of shape (batch, 1).
        """
        feats = np.stack([self._encoder.encode(obs) for obs in obs_list], axis=0)
        masks = torch.stack([legal_action_mask(obs) for obs in obs_list], dim=0).to(self.device)
        feat_t = torch.from_numpy(feats).to(self.device)
        return self._network(feat_t, masks)

    def batch_log_probs(
        self,
        obs_list:     List[PlayerObservation],
        action_list:  List[Action],
    ) -> torch.Tensor:
        """
        Evaluate log π(a_t | s_t) for a batch of (obs, action) pairs.

        Returns
        -------
        Tensor of shape (batch,) — one log-prob per step.
        """
        logits, _ = self.batch_forward(obs_list)
        action_indices = torch.tensor(
            [action_to_index(a) for a in action_list],
            device=self.device,
        )
        dist      = Categorical(logits=logits)
        return dist.log_prob(action_indices)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the agent's network weights and reward params to a file."""
        state = {
            "seat":          self.seat,
            "reward_alpha":  self._reward_fn.params.alpha,
            "reward_beta":   self._reward_fn.params.beta,
            "network_state": self._network.state_dict(),
            "hidden_dim":    self._network.hidden_dim,
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(state, path)

    @classmethod
    def load(
        cls,
        path:        str,
        device:      str = "cpu",
        deterministic: bool = False,
    ) -> "PokerAgent":
        """Load an agent from a checkpoint file."""
        state  = torch.load(path, map_location=device)
        params = RewardParams(alpha=state["reward_alpha"], beta=state["reward_beta"])
        agent  = cls(
            seat=state["seat"],
            reward_params=params,
            hidden_dim=state.get("hidden_dim", 256),
            device=device,
            deterministic=deterministic,
        )
        agent._network.load_state_dict(state["network_state"])
        return agent

    def clone_network_weights_from(self, source: "PokerAgent") -> None:
        """
        Copy network weights from *source* into this agent.
        Used to initialise a perturbed agent from a trained base agent.
        """
        self._network.load_state_dict(copy.deepcopy(source._network.state_dict()))

    def set_deterministic(self, val: bool) -> None:
        self.deterministic = val

    def train_mode(self) -> None:
        """Switch network to training mode (enables dropout)."""
        self._network.train()

    def eval_mode(self) -> None:
        """Switch network to evaluation mode (disables dropout)."""
        self._network.eval()

    def parameters(self):
        """Expose network parameters for an optimiser."""
        return self._network.parameters()

    def __repr__(self) -> str:
        return (
            f"PokerAgent(seat={self.seat}, "
            f"reward={self._reward_fn.params}, "
            f"deterministic={self.deterministic})"
        )


# ---------------------------------------------------------------------------
# Factory: build a set of 4 agents with distinct (alpha, beta) values
# ---------------------------------------------------------------------------

def make_agent_set(
    reward_params_list: List[RewardParams],
    hidden_dim:         int = 256,
    device:             str = "cpu",
) -> List[PokerAgent]:
    """
    Create one PokerAgent per seat with the given reward parameters.

    Parameters
    ----------
    reward_params_list : List of exactly 4 RewardParams objects.
    hidden_dim         : Shared network hidden layer width.
    device             : Torch device.

    Returns
    -------
    List of 4 PokerAgent objects, one per seat.
    """
    from game_state import NUM_PLAYERS
    if len(reward_params_list) != NUM_PLAYERS:
        raise ValueError(
            f"Expected {NUM_PLAYERS} RewardParams, got {len(reward_params_list)}."
        )
    return [
        PokerAgent(seat=i, reward_params=rp, hidden_dim=hidden_dim, device=device)
        for i, rp in enumerate(reward_params_list)
    ]


def make_neutral_agents(
    n: int = 4,
    hidden_dim: int = 256,
    device: str = "cpu",
) -> List[PokerAgent]:
    """Create *n* agents with neutral (alpha=0, beta=0) reward functions."""
    from reward import NeutralRewardParams
    return [
        PokerAgent(seat=i, reward_params=NeutralRewardParams, hidden_dim=hidden_dim, device=device)
        for i in range(n)
    ]
