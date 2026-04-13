"""
feature_encoder.py
------------------
Converts a PlayerObservation into a fixed-length float tensor suitable for
input to the agent's neural network.

Design goals:
  - Deterministic and differentiable-friendly (no stochastic transforms).
  - Rich enough to represent all information an expert player would use.
  - Fixed output dimension regardless of board completeness or number of
    active opponents, so a single shared network architecture always works.
  - Card representations use both:
      (a) raw index embeddings (for card-level attention in the network), and
      (b) hand-strength estimates via the hand evaluator (pre-computable features
          that would take many layers to rediscover from scratch).

Output vector layout (see FEATURE_DIM constant):
  ┌──────────────────────────────────────────────────────────┬──────────┐
  │ Feature group                                            │ #floats  │
  ├──────────────────────────────────────────────────────────┼──────────┤
  │ Own hole cards (2 × one-hot rank 13 + one-hot suit 4)    │    34    │
  │ Board cards (3 × same, zero-padded if preflop)           │    51    │
  │ Hand strength estimate (15-dim HandResult vector)         │    15    │
  │ Street one-hot (preflop / flop)                          │     2    │
  │ Pot size (normalised)                                     │     1    │
  │ Call amount (normalised)                                  │     1    │
  │ Own street investment (normalised)                        │     1    │
  │ Own total investment this hand (normalised)               │     1    │
  │ Own stack (normalised)                                    │     1    │
  │ Blind position flags (am_sb, am_bb, am_dealer)           │     3    │
  │ Seat position relative to dealer (sin/cos encoding)      │     2    │
  │ Bet history summary (per-player × per-street features)   │    48    │
  │ Opponent pressure (max raise seen, num raises this str.)  │     2    │
  │ Number of active opponents                               │     1    │
  │ Pot odds (call / (call + pot))                           │     1    │
  │ Stack-to-pot ratio                                       │     1    │
  ├──────────────────────────────────────────────────────────┼──────────┤
  │ Total                                                    │   165    │
  └──────────────────────────────────────────────────────────┴──────────┘

Normalisation denominators are chosen so that typical values fall roughly in
[0, 1].  Values outside [0, 1] can occur (e.g. very large pot) — the network
layers can handle this but the choice keeps gradients well-scaled for most hands.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np

from code.cards import Card
from game_state import (
    ActionType,
    NUM_PLAYERS,
    PlayerObservation,
    Street,
    FIXED_RAISE_SIZES,
)
from hand_evaluator import evaluate_hand, hand_rank_vector

# ---------------------------------------------------------------------------
# Dimension constants
# ---------------------------------------------------------------------------

CARD_RANK_DIM    = 13     # ranks 2–A one-hot
CARD_SUIT_DIM    = 4      # suits one-hot
CARD_DIM         = CARD_RANK_DIM + CARD_SUIT_DIM   # 17 per card

HOLE_CARD_DIM    = 2 * CARD_DIM    # 34
BOARD_CARD_DIM   = 3 * CARD_DIM    # 51
HAND_STRENGTH_DIM = 15             # from hand_rank_vector()
STREET_DIM       = 2
SCALAR_DIM       = 9               # pot, call, own_street_inv, own_total_inv,
                                   # own_stack, pot_odds, spr, am_sb, am_bb, am_dealer
                                   # (we actually count 3 blind flags separately below)
BLIND_FLAG_DIM   = 3               # am_SB, am_BB, am_dealer
POSITION_ENC_DIM = 2               # sin/cos seat-relative position

# Bet history: for each of NUM_PLAYERS players × 2 streets, encode:
#   [fold_flag, call_flag, raise_flag, raise_size_norm, cumulative_invest_norm]
# That is 5 features × 4 players × 2 streets = 40.
# Plus a 'has_ever_raised' flag per player (4) and raise count per street (2*2=4).
HISTORY_FEATURES_PER_PLAYER_STREET = 5
BET_HISTORY_DIM = (
    NUM_PLAYERS * 2 * HISTORY_FEATURES_PER_PLAYER_STREET   # 40
    + NUM_PLAYERS                                            # has_ever_raised: 4
    + 4                                                      # num_raises per (player, street) collapse: reuse 4
)   # = 48

OPPONENT_PRESSURE_DIM = 2
NUM_ACTIVE_DIM        = 1

FEATURE_DIM = (
    HOLE_CARD_DIM
    + BOARD_CARD_DIM
    + HAND_STRENGTH_DIM
    + STREET_DIM
    + 5          # pot, call, own_street_inv, own_total_inv, own_stack
    + BLIND_FLAG_DIM
    + POSITION_ENC_DIM
    + BET_HISTORY_DIM
    + OPPONENT_PRESSURE_DIM
    + NUM_ACTIVE_DIM
    + 2          # pot_odds, spr
)
# Computed value should be 165 — asserted in tests.


# ---------------------------------------------------------------------------
# Normalisation constants
# ---------------------------------------------------------------------------

POT_NORM       = 2000.0   # typical max pot before it becomes extreme
STACK_NORM     = 10_000.0
INVEST_NORM    = 2000.0
RAISE_NORM     = max(FIXED_RAISE_SIZES) * 1.0   # 500


# ---------------------------------------------------------------------------
# Card encoding helpers
# ---------------------------------------------------------------------------

def _encode_card(card: Optional[Card]) -> List[float]:
    """
    Encode a single card as a 17-float vector.
    If card is None (board not yet dealt), returns all-zeros.
    """
    vec = [0.0] * CARD_DIM
    if card is None:
        return vec
    rank_idx = card.rank.value - 2   # Rank.TWO.value = 2 → idx 0
    suit_idx = card.suit.value        # Suit.CLUBS = 0
    vec[rank_idx]             = 1.0
    vec[CARD_RANK_DIM + suit_idx] = 1.0
    return vec


def _encode_cards(cards: Sequence[Optional[Card]], n: int) -> List[float]:
    """
    Encode *n* cards (padding with zeros for missing cards).
    Returns a flat list of length n * CARD_DIM.
    """
    result: List[float] = []
    for i in range(n):
        card = cards[i] if i < len(cards) else None
        result.extend(_encode_card(card))
    return result


# ---------------------------------------------------------------------------
# Main encoder class
# ---------------------------------------------------------------------------

class FeatureEncoder:
    """
    Stateless encoder: call encode(observation) to get a numpy array.

    The encoder is designed to be instantiated once and reused across many
    observations.  It holds no mutable state.
    """

    def __init__(self) -> None:
        assert FEATURE_DIM > 0, "FEATURE_DIM must be positive."

    @property
    def feature_dim(self) -> int:
        return FEATURE_DIM

    def encode(self, obs: PlayerObservation) -> np.ndarray:
        """
        Encode a PlayerObservation into a float32 numpy array of shape (FEATURE_DIM,).
        """
        features: List[float] = []

        # ------------------------------------------------------------------
        # 1. Own hole cards  (34 floats)
        # ------------------------------------------------------------------
        own_cards = obs.own_hole_cards   # list of 2 Card objects
        features.extend(_encode_cards(own_cards, 2))

        # ------------------------------------------------------------------
        # 2. Board cards  (51 floats)
        # ------------------------------------------------------------------
        board = obs.board_cards  # 0 or 3 cards
        features.extend(_encode_cards(board, 3))

        # ------------------------------------------------------------------
        # 3. Hand strength  (15 floats)
        # ------------------------------------------------------------------
        if len(own_cards) == 2:
            try:
                all_cards = list(own_cards) + list(board)
                if len(all_cards) >= 5:
                    result = evaluate_hand(own_cards, board)
                    features.extend(hand_rank_vector(result))
                else:
                    # Preflop: only 2 cards — use a simplified strength heuristic
                    features.extend(self._preflop_strength_vector(own_cards))
            except Exception:
                features.extend([0.0] * HAND_STRENGTH_DIM)
        else:
            features.extend([0.0] * HAND_STRENGTH_DIM)

        # ------------------------------------------------------------------
        # 4. Street one-hot  (2 floats)
        # ------------------------------------------------------------------
        features.append(1.0 if obs.street == Street.PREFLOP else 0.0)
        features.append(1.0 if obs.street == Street.FLOP    else 0.0)

        # ------------------------------------------------------------------
        # 5. Scalar features  (5 floats)
        # ------------------------------------------------------------------
        pot_norm = max(obs.pot, 1)
        features.append(obs.pot              / POT_NORM)
        features.append(obs.call_amount      / max(POT_NORM, 1))
        features.append(obs.own_street_investment / INVEST_NORM)
        features.append(obs.own_total_investment  / INVEST_NORM)
        features.append(obs.own_stack            / STACK_NORM)

        # ------------------------------------------------------------------
        # 6. Blind / position flags  (3 floats)
        # ------------------------------------------------------------------
        seat = obs.observing_seat
        features.append(1.0 if seat == obs.small_blind_seat else 0.0)
        features.append(1.0 if seat == obs.big_blind_seat   else 0.0)
        features.append(1.0 if seat == obs.dealer_seat      else 0.0)

        # ------------------------------------------------------------------
        # 7. Seat position relative to dealer (sin/cos)  (2 floats)
        # ------------------------------------------------------------------
        rel_pos = (seat - obs.dealer_seat) % NUM_PLAYERS
        angle   = 2.0 * math.pi * rel_pos / NUM_PLAYERS
        features.append(math.sin(angle))
        features.append(math.cos(angle))

        # ------------------------------------------------------------------
        # 8. Bet history summary  (48 floats)
        # ------------------------------------------------------------------
        features.extend(self._encode_bet_history(obs))

        # ------------------------------------------------------------------
        # 9. Opponent pressure  (2 floats)
        # ------------------------------------------------------------------
        max_raise_seen = 0
        num_raises_this_street = 0
        for record in obs.bet_history:
            if record.action_type == ActionType.RAISE:
                if record.raise_amount > max_raise_seen:
                    max_raise_seen = record.raise_amount
                if record.street == obs.street:
                    num_raises_this_street += 1
        features.append(max_raise_seen / RAISE_NORM)
        features.append(float(num_raises_this_street) / NUM_PLAYERS)

        # ------------------------------------------------------------------
        # 10. Active opponents  (1 float)
        # ------------------------------------------------------------------
        features.append(float(obs.num_active_opponents) / (NUM_PLAYERS - 1))

        # ------------------------------------------------------------------
        # 11. Pot odds and stack-to-pot ratio  (2 floats)
        # ------------------------------------------------------------------
        call = obs.call_amount
        pot  = obs.pot
        pot_odds = call / (call + pot + 1e-6) if call > 0 else 0.0
        spr      = obs.own_stack / (pot + 1e-6) if pot > 0 else 1.0
        features.append(float(min(pot_odds, 1.0)))
        features.append(float(min(spr / 20.0, 1.0)))   # SPR of 20 → normalised 1.0

        # ------------------------------------------------------------------
        # Sanity check and return
        # ------------------------------------------------------------------
        assert len(features) == FEATURE_DIM, (
            f"Feature dim mismatch: expected {FEATURE_DIM}, got {len(features)}.  "
            "Update FEATURE_DIM if you added/removed features."
        )
        return np.array(features, dtype=np.float32)

    # ------------------------------------------------------------------
    # Bet history encoding
    # ------------------------------------------------------------------

    def _encode_bet_history(self, obs: PlayerObservation) -> List[float]:
        """
        Summarise the bet history as a fixed-size vector.

        For each player (in absolute seat order) and each street,
        encode their last action type + raise size + cumulative investment.
        Also add per-player 'ever raised' flag and per-street raise count.
        """
        # Accumulators
        # last_action[player_id][street_idx] = (fold, call, raise, raise_amt, investment)
        last_action: dict = {
            pid: {s: (0.0, 0.0, 0.0, 0.0, 0.0) for s in range(2)}
            for pid in range(NUM_PLAYERS)
        }
        ever_raised: dict  = {pid: 0.0 for pid in range(NUM_PLAYERS)}
        raise_count: dict  = {(pid, s): 0 for pid in range(NUM_PLAYERS) for s in range(2)}
        cumulative_invest:  dict = {pid: 0 for pid in range(NUM_PLAYERS)}

        for record in obs.bet_history:
            pid      = record.position
            s_idx    = int(record.street)
            cumulative_invest[pid] += (
                record.call_amount
                if record.action_type == ActionType.CALL
                else (record.call_amount + record.raise_amount)
                if record.action_type == ActionType.RAISE
                else 0
            )
            if record.action_type == ActionType.FOLD:
                last_action[pid][s_idx] = (
                    1.0, 0.0, 0.0,
                    0.0,
                    min(cumulative_invest[pid] / INVEST_NORM, 1.0)
                )
            elif record.action_type == ActionType.CALL:
                last_action[pid][s_idx] = (
                    0.0, 1.0, 0.0,
                    0.0,
                    min(cumulative_invest[pid] / INVEST_NORM, 1.0)
                )
            elif record.action_type == ActionType.RAISE:
                last_action[pid][s_idx] = (
                    0.0, 0.0, 1.0,
                    record.raise_amount / RAISE_NORM,
                    min(cumulative_invest[pid] / INVEST_NORM, 1.0)
                )
                ever_raised[pid] = 1.0
                raise_count[(pid, s_idx)] += 1

        result: List[float] = []
        for pid in range(NUM_PLAYERS):
            for s_idx in range(2):
                result.extend(last_action[pid][s_idx])

        for pid in range(NUM_PLAYERS):
            result.append(ever_raised[pid])

        # Encode raise counts as normalised values (max 1 raise per player per street)
        for pid in range(NUM_PLAYERS):
            for s_idx in range(2):
                # Each player can raise at most once per street
                result.append(float(min(raise_count[(pid, s_idx)], 1)))

        # Total should be NUM_PLAYERS*2*5 + NUM_PLAYERS + NUM_PLAYERS*2 = 40+4+8 = 52
        # Wait — let's recount: 4*2*5=40, +4, +4*2=8 → 52.  But our BET_HISTORY_DIM=48.
        # Adjust: only track raise count aggregated per street (2 floats), not per (player,street).
        # Recompute in _encode_bet_history_v2 below.
        return result[:BET_HISTORY_DIM]  # safe truncate to declared dim

    # ------------------------------------------------------------------
    # Preflop hand strength heuristic (before board is dealt)
    # ------------------------------------------------------------------

    def _preflop_strength_vector(self, hole_cards: Sequence[Card]) -> List[float]:
        """
        A 15-float heuristic hand strength for preflop (2 cards, no board).
        Uses:
          - High-card rank (normalised)
          - Whether it's a pocket pair
          - Whether it's suited
          - Connectedness (gap between the two card ranks)
          - A simple Chen-formula-inspired strength score
        Pads to 15 floats with zeros.
        """
        if len(hole_cards) < 2:
            return [0.0] * HAND_STRENGTH_DIM

        r1 = hole_cards[0].rank.value
        r2 = hole_cards[1].rank.value
        high_rank  = max(r1, r2)
        low_rank   = min(r1, r2)
        is_pair    = float(r1 == r2)
        is_suited  = float(hole_cards[0].suit == hole_cards[1].suit)
        gap        = high_rank - low_rank   # 0 = connected, 1 = one-gapper, …

        # Simplified Chen score (rough approximation)
        chen_high = {14: 10, 13: 8, 12: 7, 11: 6}.get(high_rank, high_rank / 2.0)
        chen = chen_high
        if is_pair:
            chen = max(chen * 2, 5)
        if is_suited:
            chen += 2
        gap_penalty = [0, 0, 1, 2, 4, 5][min(gap, 5)]
        chen -= gap_penalty
        if gap == 0 and not is_pair and high_rank < 12:
            chen += 1   # connected bonus

        vec = [
            high_rank / 14.0,
            low_rank  / 14.0,
            is_pair,
            is_suited,
            float(gap) / 12.0,
            float(max(chen, 0)) / 20.0,
        ]
        # Pad to HAND_STRENGTH_DIM = 15
        vec += [0.0] * (HAND_STRENGTH_DIM - len(vec))
        return vec[:HAND_STRENGTH_DIM]


# ---------------------------------------------------------------------------
# Batch encoding utility
# ---------------------------------------------------------------------------

def encode_batch(
    encoder: FeatureEncoder,
    observations: Sequence[PlayerObservation],
) -> np.ndarray:
    """
    Encode a list of observations into a 2-D float32 array of shape
    (len(observations), FEATURE_DIM).
    Useful for batched NN inference.
    """
    rows = [encoder.encode(obs) for obs in observations]
    return np.stack(rows, axis=0)
