"""
game_state.py
-------------
Pure data structures that represent the full state of a poker hand at any point
in time.  These objects are deliberately free of game-logic side effects — they
are created and mutated only by PokerEnv (poker_env.py).

Design principles:
  - Immutable snapshots can be cloned cheaply (used for IRL trajectory logging).
  - All information that an agent could ever need is exposed through GameState.
  - Observations visible to each player are computed by observation_for_player()
    which enforces information hiding (other players' hole cards are hidden).
  - The BettingHistory structure is designed to be directly consumable by the
    feature encoder without additional pre-processing.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional

from code.cards import Card


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_PLAYERS        = 4
SMALL_BLIND        = 10
BIG_BLIND          = 20
FIXED_RAISE_SIZES  = (20, 100, 500)   # absolute raise *amounts* on top of call
STARTING_STACK     = 10_000           # large enough to simulate infinite cash


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

class ActionType(IntEnum):
    FOLD  = 0
    CALL  = 1   # includes check (call of 0)
    RAISE = 2   # requires specifying raise_amount


@dataclass(frozen=True)
class Action:
    """
    An action taken by a player during a betting round.

    Attributes
    ----------
    action_type   : FOLD, CALL, or RAISE.
    raise_amount  : The *additional* chips raised on top of the call amount.
                    Only meaningful when action_type == RAISE; must be one of
                    FIXED_RAISE_SIZES.  Zero otherwise.
    player_id     : The player who took this action (0-indexed).
    """
    action_type:  ActionType
    player_id:    int
    raise_amount: int = 0   # only set for RAISE actions

    def __post_init__(self) -> None:
        if self.action_type == ActionType.RAISE:
            if self.raise_amount not in FIXED_RAISE_SIZES:
                raise ValueError(
                    f"raise_amount must be one of {FIXED_RAISE_SIZES}, "
                    f"got {self.raise_amount}."
                )
        else:
            # freeze-safe way to zero the field for non-raise actions
            object.__setattr__(self, "raise_amount", 0)

    def __str__(self) -> str:
        if self.action_type == ActionType.RAISE:
            return f"P{self.player_id}:RAISE({self.raise_amount})"
        return f"P{self.player_id}:{self.action_type.name}"


class Street(IntEnum):
    """The two betting streets in our simplified 4-player game."""
    PREFLOP = 0
    FLOP    = 1


# ---------------------------------------------------------------------------
# Per-player state within a hand
# ---------------------------------------------------------------------------

@dataclass
class PlayerHandState:
    """
    Tracks the within-hand state of a single player.

    Attributes
    ----------
    player_id        : Global seat index [0, NUM_PLAYERS).
    stack            : Current chip stack (chips not yet committed to pot).
    hole_cards       : The player's two private cards.
    is_folded        : True once the player has folded this hand.
    has_acted        : Whether the player has acted at least once in the current
                       street (used to determine when a street ends).
    amount_in_pot    : Total chips committed to the pot across *all* streets.
    street_investment: Chips committed in the *current* street (reset each street).
    has_raised       : Whether this player has already raised in the current street.
                       Players may only raise once per street.
    """
    player_id:         int
    stack:             int
    hole_cards:        List[Card]  = field(default_factory=list)
    is_folded:         bool        = False
    has_acted:         bool        = False
    amount_in_pot:     int         = 0   # total across all streets
    street_investment: int         = 0   # current street only
    has_raised:        bool        = False

    @property
    def is_active(self) -> bool:
        """Player is still in the hand (not folded)."""
        return not self.is_folded

    def clone(self) -> "PlayerHandState":
        return copy.deepcopy(self)


# ---------------------------------------------------------------------------
# A single bet action recorded with context, for history summarisation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BetRecord:
    """
    One entry in the bet history within a street.  Used by the feature encoder
    to construct a compact, fixed-size summary of betting history.

    Attributes
    ----------
    street        : Which street this action occurred on.
    position      : Acting player's seat index.
    action_type   : FOLD / CALL / RAISE.
    raise_amount  : Zero unless RAISE.
    pot_before    : Size of the pot *before* this action.
    call_amount   : Amount the player needed to call *before* this action.
    """
    street:       Street
    position:     int
    action_type:  ActionType
    raise_amount: int
    pot_before:   int
    call_amount:  int


# ---------------------------------------------------------------------------
# Full game state at any moment during a hand
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    """
    Complete game state for one hand.

    This object is mutated in-place by PokerEnv during a hand, but can be
    cheaply cloned at any point to produce a trajectory snapshot.

    Attributes
    ----------
    hand_number      : Global hand counter (incremented each deal).
    dealer_seat      : The seat of the nominal dealer (button) this hand.
    small_blind_seat : Seat posting the small blind.
    big_blind_seat   : Seat posting the big blind.
    street           : Current street (PREFLOP / FLOP).
    board_cards      : Community cards dealt so far (0 preflop, 3 post-flop).
    pot              : Total chips in the pot.
    players          : List of PlayerHandState, indexed by seat.
    current_player   : Seat index of the player whose turn it is.
    street_bet_level : The highest total street_investment any active player has
                       committed on the current street (the "price to call").
    bet_history      : Chronological list of BetRecord for the current hand.
    is_terminal      : True once the hand has concluded.
    winners          : Seat indices of the hand winner(s); empty until terminal.
    pot_contributions: Final per-player chip contribution (set at hand end).
    player_stacks_start: Stack sizes at the start of the hand (for reward calc).
    """
    hand_number:          int
    dealer_seat:          int
    small_blind_seat:     int
    big_blind_seat:       int
    street:               Street
    board_cards:          List[Card]
    pot:                  int
    players:              List[PlayerHandState]
    current_player:       int
    street_bet_level:     int
    bet_history:          List[BetRecord]
    is_terminal:          bool
    winners:              List[int]
    pot_contributions:    Dict[int, int]   # player_id → total chips in pot
    player_stacks_start:  Dict[int, int]  # player_id → stack at hand start

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def active_players(self) -> List[PlayerHandState]:
        """Players who have not folded."""
        return [p for p in self.players if p.is_active]

    @property
    def active_player_ids(self) -> List[int]:
        return [p.player_id for p in self.active_players]

    @property
    def num_active(self) -> int:
        return len(self.active_players)

    def player(self, seat: int) -> PlayerHandState:
        return self.players[seat]

    def call_amount_for(self, seat: int) -> int:
        """
        The number of chips player at *seat* must put in to call (i.e. match
        the highest current street investment of any player).
        Already-committed chips in the current street reduce this.
        Returns 0 if they are already matching or ahead (should not happen
        in normal play but guarded for safety).
        """
        return max(0, self.street_bet_level - self.players[seat].street_investment)

    def legal_actions(self, seat: int) -> List[Action]:
        """
        Return the list of legal Action objects for the player at *seat*.

        Rules:
          - FOLD is always available.
          - CALL is always available (check if call amount == 0).
          - RAISE is available only if the player has not already raised this
            street.  All three fixed raise sizes are always offered.
        """
        actions: List[Action] = [
            Action(ActionType.FOLD, seat),
            Action(ActionType.CALL, seat),
        ]
        if not self.players[seat].has_raised:
            for size in FIXED_RAISE_SIZES:
                actions.append(Action(ActionType.RAISE, seat, raise_amount=size))
        return actions

    def clone(self) -> "GameState":
        """Deep-clone the game state (for trajectory snapshots)."""
        return copy.deepcopy(self)

    # ------------------------------------------------------------------
    # Information-asymmetric observation
    # ------------------------------------------------------------------

    def observation_for_player(self, seat: int) -> "PlayerObservation":
        """
        Build the observation that player *seat* is legally allowed to see:
          - Own hole cards (always visible).
          - Board cards (public).
          - All other public information (pot, bets, stacks, etc.).
          - Other players' hole cards are hidden (replaced with None).
        """
        visible_hole_cards: Dict[int, Optional[List[Card]]] = {}
        for p in self.players:
            if p.player_id == seat:
                visible_hole_cards[p.player_id] = list(p.hole_cards)
            elif self.is_terminal:
                # At showdown, reveal non-folded players' cards
                visible_hole_cards[p.player_id] = (
                    list(p.hole_cards) if p.is_active else None
                )
            else:
                visible_hole_cards[p.player_id] = None

        return PlayerObservation(
            observing_seat=seat,
            hand_number=self.hand_number,
            dealer_seat=self.dealer_seat,
            small_blind_seat=self.small_blind_seat,
            big_blind_seat=self.big_blind_seat,
            street=self.street,
            board_cards=list(self.board_cards),
            pot=self.pot,
            visible_hole_cards=visible_hole_cards,
            player_stacks={p.player_id: p.stack for p in self.players},
            player_street_investments={
                p.player_id: p.street_investment for p in self.players
            },
            player_total_investments={
                p.player_id: p.amount_in_pot for p in self.players
            },
            player_folded={p.player_id: p.is_folded for p in self.players},
            player_has_raised={p.player_id: p.has_raised for p in self.players},
            current_player=self.current_player,
            street_bet_level=self.street_bet_level,
            call_amount=self.call_amount_for(seat),
            legal_actions=self.legal_actions(seat),
            bet_history=list(self.bet_history),
            is_terminal=self.is_terminal,
            winners=list(self.winners),
        )


# ---------------------------------------------------------------------------
# Player-level observation (information-asymmetric view of GameState)
# ---------------------------------------------------------------------------

@dataclass
class PlayerObservation:
    """
    Everything a player is allowed to know at the moment they must act.

    This is the primary input to the agent's decision function and to the
    feature encoder.  It is self-contained — no further GameState access is
    required once this object is produced.
    """
    observing_seat:           int
    hand_number:              int
    dealer_seat:              int
    small_blind_seat:         int
    big_blind_seat:           int
    street:                   Street
    board_cards:              List[Card]
    pot:                      int
    visible_hole_cards:       Dict[int, Optional[List[Card]]]  # None = hidden
    player_stacks:            Dict[int, int]
    player_street_investments:Dict[int, int]
    player_total_investments: Dict[int, int]
    player_folded:            Dict[int, bool]
    player_has_raised:        Dict[int, bool]
    current_player:           int
    street_bet_level:         int
    call_amount:              int
    legal_actions:            List[Action]
    bet_history:              List[BetRecord]
    is_terminal:              bool
    winners:                  List[int]

    @property
    def own_hole_cards(self) -> List[Card]:
        cards = self.visible_hole_cards.get(self.observing_seat)
        return cards if cards is not None else []

    @property
    def is_preflop(self) -> bool:
        return self.street == Street.PREFLOP

    @property
    def is_postflop(self) -> bool:
        return self.street == Street.FLOP

    @property
    def own_street_investment(self) -> int:
        return self.player_street_investments.get(self.observing_seat, 0)

    @property
    def own_total_investment(self) -> int:
        return self.player_total_investments.get(self.observing_seat, 0)

    @property
    def own_stack(self) -> int:
        return self.player_stacks.get(self.observing_seat, 0)

    @property
    def num_active_opponents(self) -> int:
        return sum(
            1 for pid, folded in self.player_folded.items()
            if pid != self.observing_seat and not folded
        )


# ---------------------------------------------------------------------------
# Trajectory step — the atomic unit stored for IRL
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryStep:
    """
    One (state, action, next_state) transition for a single player.

    This is the core data structure consumed by the IRL module.

    Attributes
    ----------
    hand_number   : Which hand this step belongs to.
    street        : Street on which the action was taken.
    acting_seat   : The player who acted.
    observation   : The PlayerObservation visible to the acting player.
    action        : The action they took.
    reward        : The reward signal (only populated at end of hand; 0 mid-hand).
    next_obs      : Observation after the action (None if this was the last
                    action of the hand for this player).
    is_terminal   : True if this was the terminal action of the hand.
    """
    hand_number:  int
    street:       Street
    acting_seat:  int
    observation:  PlayerObservation
    action:       Action
    reward:       float           # filled in at hand end by the environment
    next_obs:     Optional[PlayerObservation]
    is_terminal:  bool


@dataclass
class HandTrajectory:
    """
    Complete trajectory for one hand — all steps across all players and streets.
    The IRL module processes trajectories at this granularity.
    """
    hand_number:        int
    steps:              List[TrajectoryStep]
    final_chip_deltas:  Dict[int, int]        # player_id → net chip change
    winner_ids:         List[int]

    def steps_for_player(self, seat: int) -> List[TrajectoryStep]:
        return [s for s in self.steps if s.acting_seat == seat]
