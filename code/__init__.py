"""
poker_irl/__init__.py
---------------------
Top-level package for the Multi-Agent Poker IRL project.

Public API surface:
  Cards:
    Card, Rank, Suit, Deck, make_deck

  Hand evaluation:
    evaluate_hand, compare_hands, HandResult, HandCategory

  Game state:
    GameState, PlayerObservation, PlayerHandState, Action, ActionType,
    Street, BetRecord, TrajectoryStep, HandTrajectory,
    NUM_PLAYERS, SMALL_BLIND, BIG_BLIND, FIXED_RAISE_SIZES

  Environment:
    PokerEnv, make_env_from_agents

  Feature encoding:
    FeatureEncoder, FEATURE_DIM, encode_batch

  Reward functions:
    RewardParams, RewardFunction, NeutralRewardParams,
    RegularisedRewardConfig, compute_reward_stateless,
    reward_gradient_wrt_params

  Agents:
    PokerAgent, ActorCriticNetwork, make_agent_set, make_neutral_agents,
    NUM_ACTIONS, ACTION_INDEX_TO_SPEC, action_to_index, index_to_action,
    legal_action_mask
"""

# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------
from cards import (
    Card,
    Deck,
    Rank,
    Suit,
    make_deck,
)

# ---------------------------------------------------------------------------
# Hand evaluation
# ---------------------------------------------------------------------------
from hand_evaluator import (
    HandCategory,
    HandResult,
    compare_hands,
    evaluate_hand,
    hand_rank_vector,
)

# ---------------------------------------------------------------------------
# Game state structures
# ---------------------------------------------------------------------------
from game_state import (
    Action,
    ActionType,
    BetRecord,
    BIG_BLIND,
    FIXED_RAISE_SIZES,
    GameState,
    HandTrajectory,
    NUM_PLAYERS,
    PlayerHandState,
    PlayerObservation,
    SMALL_BLIND,
    STARTING_STACK,
    Street,
    TrajectoryStep,
)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
from poker_env import (
    AgentCallback,
    PokerEnv,
    make_env_from_agents,
)

# ---------------------------------------------------------------------------
# Feature encoding
# ---------------------------------------------------------------------------
from feature_encoder import (
    FEATURE_DIM,
    FeatureEncoder,
    encode_batch,
)

# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------
from reward import (
    HandRewardComponents,
    NeutralRewardParams,
    RegularisedRewardConfig,
    RewardFunction,
    RewardParams,
    RollingVarianceTracker,
    compute_pot_involvement,
    compute_reward_stateless,
    reward_gradient_wrt_params,
)

# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
from agent import (
    ACTION_INDEX_TO_SPEC,
    NUM_ACTIONS,
    ActorCriticNetwork,
    PokerAgent,
    ResidualBlock,
    action_to_index,
    index_to_action,
    legal_action_mask,
    make_agent_set,
    make_neutral_agents,
)

__all__ = [
    # Cards
    "Card", "Deck", "Rank", "Suit", "make_deck",
    # Hand evaluation
    "HandCategory", "HandResult", "compare_hands", "evaluate_hand", "hand_rank_vector",
    # Game state
    "Action", "ActionType", "BetRecord", "BIG_BLIND", "FIXED_RAISE_SIZES",
    "GameState", "HandTrajectory", "NUM_PLAYERS", "PlayerHandState",
    "PlayerObservation", "SMALL_BLIND", "STARTING_STACK", "Street", "TrajectoryStep",
    # Environment
    "AgentCallback", "PokerEnv", "make_env_from_agents",
    # Feature encoding
    "FEATURE_DIM", "FeatureEncoder", "encode_batch",
    # Reward
    "HandRewardComponents", "NeutralRewardParams", "RegularisedRewardConfig",
    "RewardFunction", "RewardParams", "RollingVarianceTracker",
    "compute_pot_involvement", "compute_reward_stateless", "reward_gradient_wrt_params",
    # Agents
    "ACTION_INDEX_TO_SPEC", "NUM_ACTIONS", "ActorCriticNetwork", "PokerAgent",
    "ResidualBlock", "action_to_index", "index_to_action", "legal_action_mask",
    "make_agent_set", "make_neutral_agents",
]
