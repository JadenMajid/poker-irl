"""
cards.py
--------
Core card primitives: Suit, Rank, Card, and Deck.

Design notes:
  - Cards are immutable value objects represented as (rank, suit) enums.
  - A Deck is a mutable shuffle-and-deal interface over one standard 52-card deck.
  - All card objects are hashable so they can be stored in sets/dicts (useful
    for hand evaluators and board texture analysis).
"""

from __future__ import annotations

import random
from enum import IntEnum
from functools import total_ordering
from typing import List, Optional, Sequence


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Suit(IntEnum):
    """
    Standard suits ordered so that higher integer ≡ conventionally higher suit
    (used only as tie-breaker in some display contexts; suit order does not
    affect hand strength in standard poker).
    """
    CLUBS    = 0
    DIAMONDS = 1
    HEARTS   = 2
    SPADES   = 3

    def __str__(self) -> str:
        return {
            Suit.CLUBS:    "c",
            Suit.DIAMONDS: "d",
            Suit.HEARTS:   "h",
            Suit.SPADES:   "s",
        }[self]

    def symbol(self) -> str:
        return {
            Suit.CLUBS:    "♣",
            Suit.DIAMONDS: "♦",
            Suit.HEARTS:   "♥",
            Suit.SPADES:   "♠",
        }[self]


class Rank(IntEnum):
    """
    Card ranks.  TWO = 2, …, ACE = 14.
    The numeric values are intentional: they allow direct arithmetic comparisons
    and make it trivial to check straights (consecutive integers).
    """
    TWO   =  2
    THREE =  3
    FOUR  =  4
    FIVE  =  5
    SIX   =  6
    SEVEN =  7
    EIGHT =  8
    NINE  =  9
    TEN   = 10
    JACK  = 11
    QUEEN = 12
    KING  = 13
    ACE   = 14

    def __str__(self) -> str:
        special = {
            Rank.TEN:   "T",
            Rank.JACK:  "J",
            Rank.QUEEN: "Q",
            Rank.KING:  "K",
            Rank.ACE:   "A",
        }
        return special.get(self, str(self.value))


# ---------------------------------------------------------------------------
# Card
# ---------------------------------------------------------------------------

@total_ordering
class Card:
    """
    An immutable playing card.

    Cards are comparable by rank only (suit is irrelevant to hand strength).
    They are hashable and can be used as dict keys or stored in sets.
    """

    __slots__ = ("_rank", "_suit")

    def __init__(self, rank: Rank, suit: Suit) -> None:
        object.__setattr__(self, "_rank", rank)
        object.__setattr__(self, "_suit", suit)

    # prevent mutation after creation
    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Card objects are immutable.")

    @property
    def rank(self) -> Rank:
        return self._rank  # type: ignore[return-value]

    @property
    def suit(self) -> Suit:
        return self._suit  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Comparison (by rank; suit is a secondary tie-breaker for display sort)
    # ------------------------------------------------------------------
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self._rank == other._rank and self._suit == other._suit

    def __lt__(self, other: "Card") -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        if self._rank != other._rank:
            return self._rank < other._rank
        return self._suit < other._suit

    def __hash__(self) -> int:
        return hash((self._rank, self._suit))

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"Card({self._rank!r}, {self._suit!r})"

    def __str__(self) -> str:
        return f"{self._rank}{self._suit}"

    def to_int(self) -> int:
        """
        Encode the card as a single integer in [0, 51].
        Layout: rank_index * 4 + suit_index, where rank_index ∈ [0, 12]
        (TWO → 0, ACE → 12) and suit_index ∈ [0, 3].
        Useful for embedding lookups in neural networks.
        """
        rank_idx = self._rank.value - 2   # type: ignore[union-attr]
        suit_idx = self._suit.value        # type: ignore[union-attr]
        return rank_idx * 4 + suit_idx

    @staticmethod
    def from_int(idx: int) -> "Card":
        """Inverse of to_int()."""
        if not 0 <= idx <= 51:
            raise ValueError(f"Card index must be in [0, 51], got {idx}.")
        rank_idx = idx // 4
        suit_idx = idx % 4
        rank = Rank(rank_idx + 2)
        suit = Suit(suit_idx)
        return Card(rank, suit)

    @staticmethod
    def from_str(s: str) -> "Card":
        """
        Parse a short string like "Ah", "2c", "Td", "Ks".
        Rank part: 2-9, T, J, Q, K, A  (case-insensitive for rank).
        Suit part: c/d/h/s              (lowercase).
        """
        if len(s) != 2:
            raise ValueError(f"Cannot parse card from '{s}'.")
        rank_char = s[0].upper()
        suit_char = s[1].lower()
        rank_map = {
            "2": Rank.TWO,   "3": Rank.THREE, "4": Rank.FOUR,
            "5": Rank.FIVE,  "6": Rank.SIX,   "7": Rank.SEVEN,
            "8": Rank.EIGHT, "9": Rank.NINE,  "T": Rank.TEN,
            "J": Rank.JACK,  "Q": Rank.QUEEN, "K": Rank.KING,
            "A": Rank.ACE,
        }
        suit_map = {"c": Suit.CLUBS, "d": Suit.DIAMONDS, "h": Suit.HEARTS, "s": Suit.SPADES}
        if rank_char not in rank_map:
            raise ValueError(f"Unknown rank character '{rank_char}'.")
        if suit_char not in suit_map:
            raise ValueError(f"Unknown suit character '{suit_char}'.")
        return Card(rank_map[rank_char], suit_map[suit_char])


# ---------------------------------------------------------------------------
# Full deck
# ---------------------------------------------------------------------------

class Deck:
    """
    A standard 52-card deck.

    A new Deck is always created in sorted order; call shuffle() before
    dealing.  The deck is exhaustive: each card appears exactly once, so it
    is safe to assert no duplicate cards appear across hole cards and board.
    """

    def __init__(self) -> None:
        self._cards: List[Card] = [
            Card(rank, suit)
            for rank in Rank
            for suit in Suit
        ]
        self._index: int = 0   # pointer to the next card to be dealt

    def shuffle(self, seed: Optional[int] = None) -> None:
        """Shuffle the full deck and reset the deal pointer."""
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(self._cards)
        else:
            random.shuffle(self._cards)
        self._index = 0

    def deal(self, n: int = 1) -> List[Card]:
        """
        Deal the next *n* cards from the top of the deck.
        Raises RuntimeError if the deck is exhausted.
        """
        if self._index + n > len(self._cards):
            raise RuntimeError(
                f"Deck exhausted: tried to deal {n} card(s) but only "
                f"{len(self._cards) - self._index} remain."
            )
        cards = self._cards[self._index : self._index + n]
        self._index += n
        return cards

    def deal_one(self) -> Card:
        return self.deal(1)[0]

    @property
    def remaining(self) -> int:
        """Number of cards left in the deck."""
        return len(self._cards) - self._index

    def reset(self) -> None:
        """Return all cards to the deck without reshuffling."""
        self._index = 0

    def __len__(self) -> int:
        return self.remaining

    def __repr__(self) -> str:
        return f"Deck(remaining={self.remaining})"

    # ------------------------------------------------------------------
    # Helpers for testing / IRL trajectory generation
    # ------------------------------------------------------------------
    def peek_remaining(self) -> List[Card]:
        """Return a copy of the undealt cards (useful for simulation)."""
        return list(self._cards[self._index:])

    def remove(self, cards: Sequence[Card]) -> None:
        """
        Remove specific cards from the undealt portion.
        Useful when reconstructing a game state mid-hand.
        Raises ValueError if any card is already dealt or not in deck.
        """
        card_set = set(cards)
        remaining = self._cards[self._index:]
        remaining_set = set(remaining)
        missing = card_set - remaining_set
        if missing:
            raise ValueError(f"Cards not available in undealt deck: {missing}")
        self._cards = self._cards[:self._index] + [c for c in remaining if c not in card_set]


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def make_deck(seed: Optional[int] = None) -> Deck:
    """Create a shuffled deck ready to deal."""
    deck = Deck()
    deck.shuffle(seed=seed)
    return deck
