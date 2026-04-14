"""
poker_env.py
------------
The 4-player simplified Texas Hold'em environment.

Variant specification:
  - 4 players, fixed seats, infinite chip reserves (cash-game model).
  - 1 pre-flop betting round → deal 3 flop cards → 1 post-flop betting round.
  - Small blind = 10, big blind = 20.  Blind positions rotate clockwise each hand.
  - Betting order:
      Pre-flop : UTG (left of BB) → … → BB (last to act preflop, since they
                 paid the big blind and must be given the option to re-raise).
      Post-flop: SB → … (skipping folded players) in seat order.
  - Each player may raise at most once per street.
  - Raise sizes are fixed: +20, +100, +500 on top of the current call amount.
  - Hand ends when ≤1 player remains (others folded), or both betting rounds
    conclude with all-active-players having matched the highest bet.
  - Showdown: best 5-of-7 among non-folded players; pot split on exact ties.
  - One standard 52-card deck per hand.

The environment exposes a step() interface compatible with RL training loops
and also records full HandTrajectory objects for IRL consumption.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Sequence

from cards import Deck, make_deck
from game_state import (
    Action,
    ActionType,
    BetRecord,
    GameState,
    HandTrajectory,
    NUM_PLAYERS,
    PlayerHandState,
    PlayerObservation,
    SMALL_BLIND,
    BIG_BLIND,
    STARTING_STACK,
    Street,
    TrajectoryStep,
)
from hand_evaluator import compare_hands

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

AgentCallback = Callable[[PlayerObservation], Action]
"""
Signature expected of any callable that acts as an agent policy.
The environment calls callback(observation) → Action each time a player must act.
"""


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class PokerEnv:
    """
    Four-player simplified Hold'em environment.

    Usage (single-hand)
    -------------------
    >>> env = PokerEnv(agent_callbacks)
    >>> trajectory = env.play_hand()

    Usage (multi-hand RL loop)
    --------------------------
    >>> env = PokerEnv(agent_callbacks)
    >>> for _ in range(10_000):
    ...     trajectory = env.play_hand()
    ...     # feed trajectory to RL update ...

    Parameters
    ----------
    agent_callbacks : List of 4 callables (one per seat).  Each callable
                      receives a PlayerObservation and must return a legal
                      Action for that player.  If None is passed for a seat,
                      the environment raises an error when that seat must act.
    seed            : Optional RNG seed for reproducibility.
    record_trajectories: If True (default), full TrajectoryStep objects are
                      accumulated; set False for maximum training speed if you
                      only need final rewards.
    """

    def __init__(
        self,
        agent_callbacks: List[Optional[AgentCallback]],
        seed: Optional[int] = None,
        record_trajectories: bool = True,
    ) -> None:
        if len(agent_callbacks) != NUM_PLAYERS:
            raise ValueError(
                f"Exactly {NUM_PLAYERS} agent callbacks required, "
                f"got {len(agent_callbacks)}."
            )
        self._callbacks        = agent_callbacks
        self._seed             = seed
        self._record           = record_trajectories
        self._hand_number      = 0
        self._dealer_seat      = 0   # seat 0 is dealer / button for hand 1

        # Stacks persist across hands (cash-game model — always topped up).
        self._stacks: Dict[int, int] = {i: STARTING_STACK for i in range(NUM_PLAYERS)}

        # Cumulative statistics (useful for monitoring during training).
        self._total_chip_deltas: Dict[int, int] = {i: 0 for i in range(NUM_PLAYERS)}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def hand_number(self) -> int:
        return self._hand_number

    @property
    def cumulative_chip_deltas(self) -> Dict[int, int]:
        return dict(self._total_chip_deltas)

    def play_hand(self) -> HandTrajectory:
        """
        Execute one complete hand and return the trajectory.
        The agent callbacks are called synchronously for each decision point.
        """
        self._hand_number += 1
        state = self._initialize_hand()
        pending_steps: List[TrajectoryStep] = []  # steps awaiting reward fill-in

        while not state.is_terminal:
            seat = state.current_player
            obs  = state.observation_for_player(seat)

            # --- Query agent ---
            callback = self._callbacks[seat]
            if callback is None:
                raise RuntimeError(
                    f"No agent callback registered for seat {seat}."
                )
            action = callback(obs)

            # --- Validate action ---
            action = self._validate_and_coerce(action, state, seat)

            # --- Record step (pre-apply) ---
            if self._record:
                step = TrajectoryStep(
                    hand_number=self._hand_number,
                    street=state.street,
                    acting_seat=seat,
                    observation=obs,
                    action=action,
                    reward=0.0,         # filled in at hand end
                    next_obs=None,       # filled in after applying action
                    is_terminal=False,
                )
                pending_steps.append(step)

            # --- Apply action ---
            self._apply_action(state, action)

            # --- Fill next_obs for the previous step ---
            if self._record and pending_steps:
                last = pending_steps[-1]
                if not state.is_terminal:
                    next_obs = state.observation_for_player(state.current_player)
                else:
                    next_obs = None
                # Rebuild with updated fields (dataclass is mutable here)
                pending_steps[-1].next_obs    = next_obs
                pending_steps[-1].is_terminal = state.is_terminal

        # --- Resolve hand, compute chip deltas ---
        chip_deltas = self._resolve_terminal(state)

        # --- Update persistent stacks ---
        for pid, delta in chip_deltas.items():
            self._stacks[pid] += delta
            self._total_chip_deltas[pid] += delta

        # --- Back-fill rewards into trajectory steps ---
        if self._record:
            for step in pending_steps:
                step.reward = float(chip_deltas.get(step.acting_seat, 0))

        return HandTrajectory(
            hand_number=self._hand_number,
            steps=pending_steps,
            final_chip_deltas=chip_deltas,
            winner_ids=list(state.winners),
        )

    def set_agent_callback(self, seat: int, callback: AgentCallback) -> None:
        """Hot-swap an agent callback (e.g. to inject a fixed opponent during ablation)."""
        if not 0 <= seat < NUM_PLAYERS:
            raise ValueError(f"seat must be in [0, {NUM_PLAYERS}), got {seat}.")
        self._callbacks[seat] = callback

    def reset_stacks(self, stack: int = STARTING_STACK) -> None:
        """Reset all stacks to the given value (e.g. between experiment episodes)."""
        self._stacks = {i: stack for i in range(NUM_PLAYERS)}
        self._total_chip_deltas = {i: 0 for i in range(NUM_PLAYERS)}

    # ------------------------------------------------------------------
    # Internal — hand initialisation
    # ------------------------------------------------------------------

    def _initialize_hand(self) -> GameState:
        """Set up blinds, deal hole cards, return the initial GameState."""
        dealer   = self._dealer_seat
        sb_seat  = (dealer + 1) % NUM_PLAYERS
        bb_seat  = (dealer + 2) % NUM_PLAYERS

        deck = make_deck(seed=self._seed)

        # Build per-player state
        players: List[PlayerHandState] = []
        stacks_start: Dict[int, int] = {}
        for i in range(NUM_PLAYERS):
            stacks_start[i] = self._stacks[i]
            players.append(
                PlayerHandState(
                    player_id=i,
                    stack=self._stacks[i],
                    hole_cards=deck.deal(2),
                )
            )

        # Post blinds
        sb_player = players[sb_seat]
        bb_player = players[bb_seat]

        sb_amount = min(SMALL_BLIND, sb_player.stack)
        bb_amount = min(BIG_BLIND,   bb_player.stack)

        sb_player.stack             -= sb_amount
        sb_player.street_investment  = sb_amount
        sb_player.amount_in_pot      = sb_amount

        bb_player.stack             -= bb_amount
        bb_player.street_investment  = bb_amount
        bb_player.amount_in_pot      = bb_amount

        pot = sb_amount + bb_amount
        street_bet_level = bb_amount   # BB sets the opening bet level

        # Preflop acting order: UTG first (left of BB), then going clockwise,
        # with BB last (they get the option).
        utg = (bb_seat + 1) % NUM_PLAYERS
        current_player = utg

        state = GameState(
            hand_number=self._hand_number,
            dealer_seat=dealer,
            small_blind_seat=sb_seat,
            big_blind_seat=bb_seat,
            street=Street.PREFLOP,
            board_cards=[],
            pot=pot,
            players=players,
            current_player=current_player,
            street_bet_level=street_bet_level,
            bet_history=[],
            is_terminal=False,
            winners=[],
            pot_contributions={i: players[i].amount_in_pot for i in range(NUM_PLAYERS)},
            player_stacks_start=stacks_start,
        )
        self._dealer_seat = (dealer + 1) % NUM_PLAYERS   # advance button for next hand
        return state

    # ------------------------------------------------------------------
    # Internal — action application
    # ------------------------------------------------------------------

    def _validate_and_coerce(self, action: Action, state: GameState, seat: int) -> Action:
        """
        Ensure the action is legal.  If an agent returns an illegal action,
        we coerce to a safe fallback (CALL) and log a warning.
        """
        legal = state.legal_actions(seat)
        legal_types = {a.action_type for a in legal}

        if action.action_type == ActionType.RAISE:
            if ActionType.RAISE not in legal_types:
                logger.warning(
                    "Seat %d tried to RAISE but has already raised this street; "
                    "coercing to CALL.", seat
                )
                return Action(ActionType.CALL, seat)
            from game_state import FIXED_RAISE_SIZES
            if action.raise_amount not in FIXED_RAISE_SIZES:
                logger.warning(
                    "Seat %d raised by %d which is not a legal size; coercing to CALL.",
                    seat, action.raise_amount
                )
                return Action(ActionType.CALL, seat)

        if action.player_id != seat:
            action = Action(action.action_type, seat, action.raise_amount)

        return action

    def _apply_action(self, state: GameState, action: Action) -> None:
        """
        Mutate *state* in-place to reflect *action*.
        Advances current_player and transitions streets/terminal as needed.
        """
        seat   = action.player_id
        player = state.players[seat]

        # ----- Record bet history -----
        state.bet_history.append(
            BetRecord(
                street=state.street,
                position=seat,
                action_type=action.action_type,
                raise_amount=action.raise_amount,
                pot_before=state.pot,
                call_amount=state.call_amount_for(seat),
            )
        )

        # ----- Apply chips -----
        if action.action_type == ActionType.FOLD:
            player.is_folded = True
            player.has_acted = True

        elif action.action_type == ActionType.CALL:
            call_amt = state.call_amount_for(seat)
            call_amt = min(call_amt, player.stack)   # all-in cap
            player.stack             -= call_amt
            player.street_investment += call_amt
            player.amount_in_pot     += call_amt
            state.pot                += call_amt
            player.has_acted          = True

        elif action.action_type == ActionType.RAISE:
            # First call the current bet, then raise on top
            call_amt  = state.call_amount_for(seat)
            raise_amt = action.raise_amount
            total     = call_amt + raise_amt
            total     = min(total, player.stack)     # all-in cap

            player.stack             -= total
            player.street_investment += total
            player.amount_in_pot     += total
            state.pot                += total

            # Raise sets the new bet level to this player's total investment
            state.street_bet_level    = player.street_investment
            player.has_raised         = True
            player.has_acted          = True

            # Other active players who have acted now need to act again
            # (to give them a chance to call / re-raise — but since each
            # player can only raise once, they can only call or fold now).
            for p in state.players:
                if p.player_id != seat and p.is_active:
                    p.has_acted = False

        # ----- Update pot_contributions -----
        state.pot_contributions[seat] = state.players[seat].amount_in_pot

        # ----- Check for early termination (only one player left) -----
        if state.num_active == 1:
            state.is_terminal = True
            state.winners     = [state.active_player_ids[0]]
            return

        # ----- Advance to next player or next street -----
        if self._street_is_over(state):
            self._advance_street(state)
        else:
            state.current_player = self._next_active_player(state, seat)

    def _street_is_over(self, state: GameState) -> bool:
        """
        A street is over when every active player has acted at least once AND
        every active player's street_investment equals street_bet_level
        (i.e. the pot is right — all bets are matched).
        """
        for p in state.active_players:
            if not p.has_acted:
                return False
            if p.street_investment < state.street_bet_level and p.stack > 0:
                # Player has chips remaining but hasn't matched — street not over
                return False
        return True

    def _advance_street(self, state: GameState) -> None:
        """
        Move from preflop to flop, or from flop to terminal (showdown).
        Resets per-street tracking for all active players.
        """
        if state.street == Street.PREFLOP:
            # Deal flop — draw 3 board cards from the deck.
            # Reconstruct a deck minus all hole cards dealt.
            deck = Deck()
            dealt = {c for p in state.players for c in p.hole_cards}
            deck_cards = [c for c in deck._cards if c not in dealt]
            import random
            random.shuffle(deck_cards)
            state.board_cards = deck_cards[:3]

            state.street = Street.FLOP
            self._reset_street_state(state)

            # Post-flop action starts from SB (or next active player clockwise)
            state.current_player = self._first_postflop_actor(state)

        else:  # FLOP → terminal
            state.is_terminal = True
            self._determine_winners(state)

    def _reset_street_state(self, state: GameState) -> None:
        """Reset per-street player flags for the new street."""
        state.street_bet_level = 0
        for p in state.active_players:
            p.street_investment = 0
            p.has_acted         = False
            p.has_raised        = False

    def _first_postflop_actor(self, state: GameState) -> int:
        """
        Post-flop action starts with the first active player clockwise from
        the dealer/button (typically the small blind).
        """
        start = (state.dealer_seat + 1) % NUM_PLAYERS
        for i in range(NUM_PLAYERS):
            seat = (start + i) % NUM_PLAYERS
            if state.players[seat].is_active:
                return seat
        raise RuntimeError("No active players found for post-flop.")

    def _next_active_player(self, state: GameState, current_seat: int) -> int:
        """Return the next active player clockwise from *current_seat*."""
        for i in range(1, NUM_PLAYERS + 1):
            seat = (current_seat + i) % NUM_PLAYERS
            if state.players[seat].is_active:
                return seat
        raise RuntimeError("No active player found after seat %d." % current_seat)

    def _determine_winners(self, state: GameState) -> None:
        """
        Run the showdown: compare hands, set state.winners.
        """
        active = [(p.player_id, p.hole_cards) for p in state.active_players]
        state.winners = compare_hands(active, state.board_cards)

    # ------------------------------------------------------------------
    # Internal — terminal chip resolution
    # ------------------------------------------------------------------

    def _resolve_terminal(self, state: GameState) -> Dict[int, int]:
        """
        Distribute the pot to winner(s) and compute net chip delta for each player.

        Returns chip_deltas: {player_id: net chip change this hand}.
        """
        pot = state.pot

        # Split pot equally among winners (integer chips — odd chip goes to
        # the first winner in seat order, consistent with standard casino rules).
        num_winners = len(state.winners)
        base_share  = pot // num_winners
        remainder   = pot % num_winners

        chip_deltas: Dict[int, int] = {}
        for pid in range(NUM_PLAYERS):
            invested       = state.pot_contributions[pid]
            chip_deltas[pid] = -invested   # start negative by what they put in

        winners_sorted = sorted(state.winners)
        for idx, pid in enumerate(winners_sorted):
            bonus = 1 if idx < remainder else 0
            chip_deltas[pid] += base_share + bonus

        return chip_deltas


# ---------------------------------------------------------------------------
# Convenience factory: create an env from a list of agent objects
# ---------------------------------------------------------------------------

def make_env_from_agents(
    agents: Sequence,
    seed: Optional[int] = None,
    record_trajectories: bool = True,
) -> PokerEnv:
    """
    Wrap agent objects (must have a .act(observation) -> Action method)
    into an PokerEnv.
    """
    callbacks = [agent.act for agent in agents]
    return PokerEnv(callbacks, seed=seed, record_trajectories=record_trajectories)
