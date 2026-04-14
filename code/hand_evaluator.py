"""
hand_evaluator.py
-----------------
Seven-card (best-five-of-seven) poker hand evaluator.

Design goals:
  - Correct, readable, and self-contained — no external libraries.
  - Returns a *HandResult* that is totally ordered (>, <, ==) so comparing
    two results yields the winner unambiguously, including all kicker logic.
  - Fast enough for RL trajectory generation (pure Python; no bit-magic tables
    needed at this scale, but the structure makes it trivial to swap in a
    compiled lookup later if needed).

Hand ranking (highest → lowest):
  9  Royal Flush        (A K Q J T suited)
  8  Straight Flush
  7  Four of a Kind
  6  Full House
  5  Flush
  4  Straight
  3  Three of a Kind
  2  Two Pair
  1  One Pair
  0  High Card
"""

from __future__ import annotations

from collections import Counter
from enum import IntEnum
from itertools import combinations
from typing import List, NamedTuple, Sequence, Tuple

from cards import Card


# ---------------------------------------------------------------------------
# Hand category enum
# ---------------------------------------------------------------------------

class HandCategory(IntEnum):
    HIGH_CARD       = 0
    ONE_PAIR        = 1
    TWO_PAIR        = 2
    THREE_OF_A_KIND = 3
    STRAIGHT        = 4
    FLUSH           = 5
    FULL_HOUSE      = 6
    FOUR_OF_A_KIND  = 7
    STRAIGHT_FLUSH  = 8
    ROYAL_FLUSH     = 9  # treated as special case of STRAIGHT_FLUSH for clarity


# ---------------------------------------------------------------------------
# HandResult — a totally-ordered representation of a 5-card hand's strength
# ---------------------------------------------------------------------------

class HandResult(NamedTuple):
    """
    A fully comparable hand result.

    Fields (compared lexicographically, highest significance first):
      category  : HandCategory  — the primary hand rank
      tiebreakers: Tuple[int, ...]  — descending list of rank values used to
                   break ties within the same category.  Always exactly 5
                   integers (padded with zeros if needed), corresponding to the
                   ranks of the five cards making up the best hand, ordered by
                   relevance (e.g. quad rank first, then kicker for quads).

    Comparison: higher category wins; within a category, the tiebreaker tuple
    is compared element by element (standard Python tuple comparison).
    """
    category:    HandCategory
    tiebreakers: Tuple[int, ...]
    best_hand:   Tuple[Card, ...]   # the actual five cards (for display)

    # NamedTuple provides __eq__ and __lt__ from field comparison, but we
    # want to exclude `best_hand` from ordering.
    def __lt__(self, other: "HandResult") -> bool:  # type: ignore[override]
        return (self.category, self.tiebreakers) < (other.category, other.tiebreakers)

    def __le__(self, other: "HandResult") -> bool:
        return (self.category, self.tiebreakers) <= (other.category, other.tiebreakers)

    def __gt__(self, other: "HandResult") -> bool:
        return (self.category, self.tiebreakers) > (other.category, other.tiebreakers)

    def __ge__(self, other: "HandResult") -> bool:
        return (self.category, self.tiebreakers) >= (other.category, other.tiebreakers)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HandResult):
            return NotImplemented
        return (self.category, self.tiebreakers) == (other.category, other.tiebreakers)

    def __hash__(self) -> int:
        return hash((self.category, self.tiebreakers))

    def __str__(self) -> str:
        cards_str = " ".join(str(c) for c in self.best_hand)
        return f"{self.category.name.replace('_', ' ').title()} [{cards_str}]"


# ---------------------------------------------------------------------------
# Internal helpers for 5-card evaluation
# ---------------------------------------------------------------------------

def _ranks_sorted_desc(cards: Sequence[Card]) -> List[int]:
    """Return card ranks as ints sorted descending."""
    return sorted((c.rank.value for c in cards), reverse=True)


def _rank_counts(cards: Sequence[Card]) -> Counter:
    """Counter of rank values."""
    return Counter(c.rank.value for c in cards)


def _is_flush(cards: Sequence[Card]) -> bool:
    suits = [c.suit for c in cards]
    return len(set(suits)) == 1


def _straight_high_card(ranks_desc: List[int]) -> int:
    """
    Given 5 sorted-descending rank values, return the high-card rank if it
    forms a straight, else 0.  Handles the wheel (A-2-3-4-5 → high = 5).
    """
    # Normal straight: consecutive descending
    if ranks_desc[0] - ranks_desc[4] == 4 and len(set(ranks_desc)) == 5:
        return ranks_desc[0]
    # Wheel: A-2-3-4-5 — stored as [14, 5, 4, 3, 2]
    if ranks_desc == [14, 5, 4, 3, 2]:
        return 5
    return 0


def _evaluate_five(cards: Sequence[Card]) -> HandResult:
    """
    Evaluate exactly 5 cards and return a HandResult.
    This is the inner function; callers must pass exactly 5 cards.
    """
    assert len(cards) == 5, "evaluate_five requires exactly 5 cards."

    ranks_desc = _ranks_sorted_desc(cards)
    counts = _rank_counts(cards)
    flush = _is_flush(cards)
    straight_high = _straight_high_card(ranks_desc)

    # Sort cards by (count desc, rank desc) for tiebreaker construction
    # e.g. for full house [3,3,3,2,2] → three first, pair second
    groups = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    group_ranks = [r for r, _ in groups]
    group_counts_vals = [c for _, c in groups]

    cards_tuple = tuple(cards)

    # ------------------------------------------------------------------
    # Straight Flush / Royal Flush
    # ------------------------------------------------------------------
    if flush and straight_high:
        if straight_high == 14:
            return HandResult(HandCategory.ROYAL_FLUSH, (14, 13, 12, 11, 10), cards_tuple)
        tb = tuple(
            [14, 5, 4, 3, 2] if straight_high == 5 and 14 in ranks_desc else ranks_desc
        )
        return HandResult(HandCategory.STRAIGHT_FLUSH, tb, cards_tuple)

    # ------------------------------------------------------------------
    # Four of a Kind
    # ------------------------------------------------------------------
    if group_counts_vals[0] == 4:
        quad_rank  = group_ranks[0]
        kicker     = group_ranks[1]
        return HandResult(HandCategory.FOUR_OF_A_KIND, (quad_rank, kicker), cards_tuple)

    # ------------------------------------------------------------------
    # Full House
    # ------------------------------------------------------------------
    if group_counts_vals[0] == 3 and group_counts_vals[1] == 2:
        trip_rank = group_ranks[0]
        pair_rank = group_ranks[1]
        return HandResult(HandCategory.FULL_HOUSE, (trip_rank, pair_rank), cards_tuple)

    # ------------------------------------------------------------------
    # Flush
    # ------------------------------------------------------------------
    if flush:
        return HandResult(HandCategory.FLUSH, tuple(ranks_desc), cards_tuple)

    # ------------------------------------------------------------------
    # Straight
    # ------------------------------------------------------------------
    if straight_high:
        if straight_high == 5:
            return HandResult(HandCategory.STRAIGHT, (5, 4, 3, 2, 1), cards_tuple)
        return HandResult(HandCategory.STRAIGHT, tuple(ranks_desc), cards_tuple)

    # ------------------------------------------------------------------
    # Three of a Kind
    # ------------------------------------------------------------------
    if group_counts_vals[0] == 3:
        trip_rank = group_ranks[0]
        kickers   = tuple(group_ranks[1:])
        return HandResult(HandCategory.THREE_OF_A_KIND, (trip_rank,) + kickers, cards_tuple)

    # ------------------------------------------------------------------
    # Two Pair
    # ------------------------------------------------------------------
    if group_counts_vals[0] == 2 and group_counts_vals[1] == 2:
        high_pair = group_ranks[0]
        low_pair  = group_ranks[1]
        kicker    = group_ranks[2]
        return HandResult(HandCategory.TWO_PAIR, (high_pair, low_pair, kicker), cards_tuple)

    # ------------------------------------------------------------------
    # One Pair
    # ------------------------------------------------------------------
    if group_counts_vals[0] == 2:
        pair_rank = group_ranks[0]
        kickers   = tuple(group_ranks[1:])
        return HandResult(HandCategory.ONE_PAIR, (pair_rank,) + kickers, cards_tuple)

    # ------------------------------------------------------------------
    # High Card
    # ------------------------------------------------------------------
    return HandResult(HandCategory.HIGH_CARD, tuple(ranks_desc), cards_tuple)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_hand(hole_cards: Sequence[Card], board_cards: Sequence[Card]) -> HandResult:
    """
    Evaluate the best 5-card poker hand from up to 7 cards.

    Parameters
    ----------
    hole_cards  : The player's private 2 cards.
    board_cards : The community cards (0–5 cards depending on street).

    Returns
    -------
    HandResult for the best possible 5-card combination.
    """
    all_cards = list(hole_cards) + list(board_cards)
    if len(all_cards) < 5:
        raise ValueError(
            f"Need at least 5 cards to evaluate a hand, got {len(all_cards)}."
        )
    if len(all_cards) > 7:
        raise ValueError(
            f"Standard hand evaluation takes at most 7 cards, got {len(all_cards)}."
        )

    best: HandResult | None = None
    for combo in combinations(all_cards, 5):
        result = _evaluate_five(combo)
        if best is None or result > best:
            best = result

    assert best is not None
    return best


def compare_hands(
    hands: List[Tuple[int, Sequence[Card]]],
    board_cards: Sequence[Card],
) -> List[int]:
    """
    Determine the winner(s) among a set of players' hole cards.

    Parameters
    ----------
    hands       : List of (player_id, hole_cards) for players still in the hand.
    board_cards : Shared community cards.

    Returns
    -------
    List of player_ids who hold the best hand (multiple in case of a tie).
    """
    if not hands:
        return []

    results = [
        (pid, evaluate_hand(hole, board_cards))
        for pid, hole in hands
    ]

    best_result = max(r for _, r in results)
    return [pid for pid, r in results if r == best_result]


def hand_rank_vector(result: HandResult, num_categories: int = 10) -> List[float]:
    """
    Encode a HandResult as a float vector for use as a neural-net feature.

    Layout:
      - One-hot over the 10 hand categories (index 0–9).
      - Five normalised tiebreaker values in [0, 1]   (rank / 14.0).

    Total length: 10 + 5 = 15.
    """
    one_hot = [0.0] * num_categories
    one_hot[int(result.category)] = 1.0

    # Tiebreakers: pad or truncate to exactly 5 values, normalise by 14.
    tb = list(result.tiebreakers)
    tb = (tb + [0] * 5)[:5]
    normalised = [v / 14.0 for v in tb]

    return one_hot + normalised
