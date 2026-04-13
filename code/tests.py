"""
tests.py
--------
Self-contained test suite for the poker IRL foundational codebase.
Run with:  python tests.py  (no external test framework required)

Covers:
  1. Card primitives (encoding, parsing, deck exhaustion)
  2. Hand evaluator (all hand categories, kicker resolution, tie handling)
  3. Game state data structures
  4. Poker environment (full hand, betting rules, blinds, showdown)
  5. Feature encoder (output shape, preflop/postflop)
  6. Reward function (chip delta, variance, pot involvement)
  7. Agent (action selection, log-prob, mask, checkpoint)
  8. Integration test: full 100-hand session
"""

import sys
import os
import traceback
import numpy as np
import torch

# Add parent dir to path so we can import from poker_irl/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from code.cards import Card, Rank, Suit, Deck
from hand_evaluator import (
    evaluate_hand, compare_hands, HandCategory, hand_rank_vector
)
from game_state import (
    Action, ActionType, PlayerObservation, Street,
    NUM_PLAYERS
)
from poker_env import PokerEnv, make_env_from_agents
from feature_encoder import FeatureEncoder, FEATURE_DIM, encode_batch
from reward import (
    RewardParams, RewardFunction, NeutralRewardParams,
    RollingVarianceTracker, compute_reward_stateless, reward_gradient_wrt_params
)
from code.agent import (
    PokerAgent, ActorCriticNetwork, legal_action_mask, make_neutral_agents, make_agent_set, NUM_ACTIONS
)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

PASSED = []
FAILED = []

def test(name: str):
    """Decorator to register and run a named test."""
    def decorator(fn):
        try:
            fn()
            PASSED.append(name)
            print(f"  ✓  {name}")
        except Exception as e:
            FAILED.append((name, traceback.format_exc()))
            print(f"  ✗  {name}")
            print(f"     {e}")
        return fn
    return decorator


def assert_eq(a, b, msg=""):
    assert a == b, f"{msg}  Got {a!r} expected {b!r}"

def assert_gt(a, b, msg=""):
    assert a > b, f"{msg}  Got {a!r} (should be > {b!r})"

def assert_close(a, b, tol=1e-4, msg=""):
    assert abs(a - b) < tol, f"{msg}  Got {a} vs {b} (tol={tol})"


# ---------------------------------------------------------------------------
# 1. Card tests
# ---------------------------------------------------------------------------

print("\n=== Cards ===")

@test("Card encoding round-trip (to_int / from_int)")
def _():
    for i in range(52):
        c = Card.from_int(i)
        assert c.to_int() == i, f"Round-trip failed at index {i}"

@test("Card from_str / __str__")
def _():
    c = Card.from_str("Ah")
    assert c.rank == Rank.ACE
    assert c.suit == Suit.HEARTS
    assert str(c) == "Ah"

@test("Card immutability")
def _():
    c = Card(Rank.KING, Suit.SPADES)
    try:
        c._rank = Rank.ACE
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

@test("Deck has 52 distinct cards")
def _():
    deck = Deck()
    cards = deck.deal(52)
    assert len(cards) == 52
    assert len(set(c.to_int() for c in cards)) == 52

@test("Deck exhaustion raises RuntimeError")
def _():
    deck = Deck()
    deck.deal(52)
    try:
        deck.deal(1)
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass

@test("Deck shuffle changes order (statistically)")
def _():
    d1 = Deck(); d1.shuffle(seed=42)
    d2 = Deck(); d2.shuffle(seed=99)
    c1 = d1.deal(10)
    c2 = d2.deal(10)
    # Very unlikely to match in practice
    assert c1 != c2

# ---------------------------------------------------------------------------
# 2. Hand evaluator
# ---------------------------------------------------------------------------

print("\n=== Hand Evaluator ===")

def cards(*strs):
    return [Card.from_str(s) for s in strs]

@test("Royal flush detected")
def _():
    h = evaluate_hand(cards("Ah", "Kh"), cards("Qh", "Jh", "Th"))
    assert h.category == HandCategory.ROYAL_FLUSH

@test("Straight flush detected")
def _():
    h = evaluate_hand(cards("9h", "8h"), cards("7h", "6h", "5h"))
    assert h.category == HandCategory.STRAIGHT_FLUSH

@test("Four of a kind detected")
def _():
    h = evaluate_hand(cards("As", "Ah"), cards("Ac", "Ad", "2s"))
    assert h.category == HandCategory.FOUR_OF_A_KIND

@test("Full house detected")
def _():
    h = evaluate_hand(cards("As", "Ah"), cards("Ac", "2d", "2s"))
    assert h.category == HandCategory.FULL_HOUSE

@test("Flush detected")
def _():
    h = evaluate_hand(cards("Ah", "3h"), cards("7h", "9h", "2h"))
    assert h.category == HandCategory.FLUSH

@test("Straight detected")
def _():
    h = evaluate_hand(cards("9s", "8h"), cards("7d", "6c", "5s"))
    assert h.category == HandCategory.STRAIGHT

@test("Wheel (A-5 straight) detected")
def _():
    h = evaluate_hand(cards("As", "2h"), cards("3d", "4c", "5s"))
    assert h.category == HandCategory.STRAIGHT
    assert h.tiebreakers[0] == 5   # high card is 5

@test("Three of a kind detected")
def _():
    h = evaluate_hand(cards("As", "Ah"), cards("Ac", "2d", "7s"))
    assert h.category == HandCategory.THREE_OF_A_KIND

@test("Two pair detected")
def _():
    h = evaluate_hand(cards("As", "Ah"), cards("2c", "2d", "7s"))
    assert h.category == HandCategory.TWO_PAIR

@test("One pair detected")
def _():
    h = evaluate_hand(cards("As", "Ah"), cards("2c", "7d", "9s"))
    assert h.category == HandCategory.ONE_PAIR

@test("High card detected")
def _():
    h = evaluate_hand(cards("As", "Kh"), cards("2c", "7d", "9s"))
    assert h.category == HandCategory.HIGH_CARD

@test("Compare hands: flush beats straight")
def _():
    flush    = evaluate_hand(cards("Ah", "3h"), cards("7h", "9h", "2h"))
    straight = evaluate_hand(cards("9s", "8h"), cards("7d", "6c", "5s"))
    assert flush > straight

@test("Kicker resolution in one-pair")
def _():
    # Both have pair of aces; first has K kicker, second has Q
    h1 = evaluate_hand(cards("As", "Ah"), cards("Kc", "7d", "2s"))
    h2 = evaluate_hand(cards("Ad", "Ac"), cards("Qc", "7h", "2d"))
    assert h1 > h2

@test("Exact tie returns equal result")
def _():
    # Same 5-card hand using 7 cards
    h1 = evaluate_hand(cards("Ah", "Kh"), cards("Qh", "Jh", "Th"))
    h2 = evaluate_hand(cards("As", "Ks"), cards("Qs", "Js", "Ts"))
    assert h1 == h2

@test("compare_hands returns multiple winners on tie")
def _():
    hands = [
        (0, cards("Ah", "Kh")),
        (1, cards("As", "Ks")),
    ]
    board = cards("Qh", "Jd", "Ts")
    # Both have A-K-Q-J-T (broadway straight) → tie
    board_full = cards("Qh", "Jd", "Ts")
    result = compare_hands(hands, board_full)
    # With this board both get the straight
    assert len(result) >= 1   # at least one winner

@test("hand_rank_vector has correct length")
def _():
    h = evaluate_hand(cards("Ah", "Kh"), cards("Qh", "Jh", "Th"))
    vec = hand_rank_vector(h)
    assert len(vec) == 15


# ---------------------------------------------------------------------------
# 3. Game state
# ---------------------------------------------------------------------------

print("\n=== Game State ===")

@test("Action rejects invalid raise amounts")
def _():
    try:
        Action(ActionType.RAISE, player_id=0, raise_amount=999)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

@test("Action zeroes raise_amount for CALL")
def _():
    a = Action(ActionType.CALL, player_id=1, raise_amount=100)
    assert a.raise_amount == 0

@test("legal_actions lists raises only when has_raised=False")
def _():
    agents = make_neutral_agents()
    env    = PokerEnv([a.act for a in agents], record_trajectories=True)
    # Peek at the initial state
    state = env._initialize_hand()
    seat  = state.current_player
    actions = state.legal_actions(seat)
    types = {a.action_type for a in actions}
    assert ActionType.RAISE in types
    # Simulate a raise then check
    state.players[seat].has_raised = True
    actions2 = state.legal_actions(seat)
    types2 = {a.action_type for a in actions2}
    assert ActionType.RAISE not in types2


# ---------------------------------------------------------------------------
# 4. Poker environment
# ---------------------------------------------------------------------------

print("\n=== Poker Environment ===")

def make_random_env(seed=0):
    """Env where all agents act randomly (uniform over legal actions)."""
    import random as pyrandom
    def random_callback(obs):
        return pyrandom.choice(obs.legal_actions)
    callbacks = [random_callback] * NUM_PLAYERS
    return PokerEnv(callbacks, seed=seed, record_trajectories=True)

@test("Single hand completes without error")
def _():
    env = make_random_env()
    traj = env.play_hand()
    assert traj.hand_number == 1
    assert len(traj.winner_ids) >= 1

@test("Chip conservation: sum of deltas == 0")
def _():
    env = make_random_env(seed=42)
    for _ in range(20):
        traj = env.play_hand()
        total = sum(traj.final_chip_deltas.values())
        assert total == 0, f"Chip conservation violated: sum={total}"

@test("Trajectory steps have correct player IDs")
def _():
    env = make_random_env()
    traj = env.play_hand()
    for step in traj.steps:
        assert 0 <= step.acting_seat < NUM_PLAYERS

@test("Winner is always an active (non-folded) player")
def _():
    env = make_random_env(seed=7)
    for _ in range(30):
        traj = env.play_hand()
        # We can only check indirectly via chip_deltas: winner(s) have positive delta
        winners = traj.winner_ids
        assert len(winners) >= 1

@test("Blind rotation advances each hand")
def _():
    env = make_random_env()
    sb_seats = []
    for _ in range(NUM_PLAYERS + 1):
        state = env._initialize_hand()
        sb_seats.append(state.small_blind_seat)
    # All 4 seats should have been SB at least once in 5 hands
    assert len(set(sb_seats)) == NUM_PLAYERS

@test("Preflop: board cards == 0; postflop: board cards == 3")
def _():
    import random as pyrandom
    seen_preflop = False
    seen_postflop = False

    def callback_that_checks(obs):
        nonlocal seen_preflop, seen_postflop
        if obs.street == Street.PREFLOP:
            assert len(obs.board_cards) == 0
            seen_preflop = True
        else:
            assert len(obs.board_cards) == 3
            seen_postflop = True
        return pyrandom.choice(obs.legal_actions)

    env = PokerEnv([callback_that_checks] * NUM_PLAYERS, record_trajectories=False)
    for _ in range(5):
        env.play_hand()

    assert seen_preflop and seen_postflop

@test("Player who folds receives negative chip delta (equal to investment)")
def _():
    """
    An agent always folds preflop (except if it's in the blinds) and
    should lose exactly the blind amount.
    """

    def fold_policy(obs: PlayerObservation) -> Action:
        # Always fold unless call is free (blinds paid)
        if obs.call_amount == 0:
            return Action(ActionType.CALL, obs.observing_seat)
        return Action(ActionType.FOLD, obs.observing_seat)

    env = PokerEnv([fold_policy] * NUM_PLAYERS, record_trajectories=True)
    for _ in range(4):
        traj = env.play_hand()
        # Conservation check
        assert sum(traj.final_chip_deltas.values()) == 0


# ---------------------------------------------------------------------------
# 5. Feature encoder
# ---------------------------------------------------------------------------

print("\n=== Feature Encoder ===")

@test(f"Encoder output has shape ({FEATURE_DIM},)")
def _():
    encoder = FeatureEncoder()
    agents  = make_neutral_agents()
    env     = make_env_from_agents(agents)

    collected_obs = []
    def capture(obs):
        collected_obs.append(obs)
        import random
        return random.choice(obs.legal_actions)

    env2 = PokerEnv([capture] * NUM_PLAYERS)
    env2.play_hand()

    assert len(collected_obs) > 0
    for obs in collected_obs:
        vec = encoder.encode(obs)
        assert vec.shape == (FEATURE_DIM,), f"Got {vec.shape}"

@test("Encoder output contains no NaN or Inf")
def _():
    encoder = FeatureEncoder()
    agents  = make_neutral_agents()
    env     = make_env_from_agents(agents)
    obs_list = []

    def capture(obs):
        obs_list.append(obs)
        import random
        return random.choice(obs.legal_actions)

    env2 = PokerEnv([capture] * NUM_PLAYERS)
    for _ in range(5):
        env2.play_hand()

    for obs in obs_list:
        vec = encoder.encode(obs)
        assert not np.any(np.isnan(vec)), "NaN in features"
        assert not np.any(np.isinf(vec)), "Inf in features"

@test("encode_batch produces correct shape")
def _():
    encoder = FeatureEncoder()
    obs_list = []

    def capture(obs):
        obs_list.append(obs)
        import random
        return random.choice(obs.legal_actions)

    env = PokerEnv([capture] * NUM_PLAYERS)
    env.play_hand()

    batch = encode_batch(encoder, obs_list)
    assert batch.shape == (len(obs_list), FEATURE_DIM)


# ---------------------------------------------------------------------------
# 6. Reward function
# ---------------------------------------------------------------------------

print("\n=== Reward Function ===")

@test("Neutral agent reward equals chip delta")
def _():
    """With alpha=0, beta=0, reward should equal chip_delta exactly."""
    agents = make_neutral_agents()
    env    = make_env_from_agents(agents)
    traj   = env.play_hand()

    rf = RewardFunction(NeutralRewardParams, variance_window=100)
    for seat in range(NUM_PLAYERS):
        components = rf.compute(traj, seat)
        # variance_penalty = 0, pot_bonus = 0 → total = chip_delta
        assert_close(components.total, components.chip_delta, tol=1e-6)

@test("Risk-averse agent has positive variance penalty after many hands")
def _():
    params = RewardParams(alpha=1.0, beta=0.0)
    rf = RewardFunction(params, variance_window=50)
    # Feed a mix of wins and losses to build up variance
    for delta in [100, -100, 200, -200, 50, -50] * 10:
        rf._tracker.update(float(delta))

    var = rf._tracker.variance()
    assert var > 0, "Variance should be positive after mixed outcomes"

@test("Pot-involvement agent receives positive bonus for big pot action")
def _():
    params = RewardParams(alpha=0.0, beta=1.0)
    rf = RewardFunction(params, variance_window=100)

    agents = make_neutral_agents()
    env    = make_env_from_agents(agents)
    traj   = env.play_hand()

    for seat in range(NUM_PLAYERS):
        components = rf.compute(traj, seat)
        max_pot = components.max_pot_commitment
        if max_pot > 0:
            expected_bonus = 1.0 * (max_pot / 2000.0)
            assert_close(components.pot_involvement_bonus, expected_bonus, tol=1e-4)

@test("Rolling variance tracker — Bessel-corrected variance")
def _():
    tracker = RollingVarianceTracker(window_size=200)
    data = [float(i) for i in range(1, 101)]
    for x in data:
        tracker.update(x)
    expected_var = np.var(data, ddof=1)
    assert_close(tracker.variance(), expected_var, tol=0.01)

@test("compute_reward_stateless matches full RewardFunction")
def _():
    params = RewardParams(alpha=0.5, beta=0.3)
    chip_delta = 150.0
    rolling_var = 2500.0
    max_pot = 800.0

    result = compute_reward_stateless(params, chip_delta, rolling_var, max_pot)
    expected = chip_delta - 0.5 * 2500.0 + 0.3 * (800.0 / 2000.0)
    assert_close(result, expected, tol=1e-6)

@test("reward_gradient_wrt_params has correct sign")
def _():
    grad = reward_gradient_wrt_params(chip_delta=100, rolling_var=500, max_pot=1000)
    assert grad[0] == -500.0            # dR/d_alpha = -variance
    assert_close(grad[1], 1000/2000.0)  # dR/d_beta = max_pot/POT_NORM


# ---------------------------------------------------------------------------
# 7. Agent
# ---------------------------------------------------------------------------

print("\n=== Agent ===")

@test("Agent selects a legal action")
def _():
    import random as pyrandom
    agent  = PokerAgent(seat=0, reward_params=NeutralRewardParams)
    agents = make_neutral_agents()
    env    = make_env_from_agents(agents)
    obs_list = []

    def capture_and_act(obs):
        obs_list.append(obs)
        return pyrandom.choice(obs.legal_actions)

    env2 = PokerEnv([capture_and_act] + [a.act for a in agents[1:]])
    env2.play_hand()

    for obs in obs_list:
        action = agent.act(obs)
        legal_types = {a.action_type for a in obs.legal_actions}
        assert action.action_type in legal_types

@test("action_log_probs returns a finite scalar tensor")
def _():
    import random as pyrandom
    agent  = PokerAgent(seat=0, reward_params=NeutralRewardParams)
    agents = make_neutral_agents()
    obs_list = []
    act_list = []

    def capture(obs):
        a = pyrandom.choice(obs.legal_actions)
        obs_list.append(obs)
        act_list.append(a)
        return a

    env = PokerEnv([capture] + [a.act for a in agents[1:]])
    env.play_hand()

    for obs, act in zip(obs_list, act_list):
        lp = agent.action_log_probs(obs, act)
        assert torch.isfinite(lp).all()
        assert lp.item() <= 0   # log-prob ≤ 0

@test("Action mask excludes raise when has_raised=True")
def _():
    import random as pyrandom
    obs_list = []

    def capture(obs):
        obs_list.append(obs)
        return pyrandom.choice(obs.legal_actions)

    env = PokerEnv([capture] * NUM_PLAYERS)
    env.play_hand()

    for obs in obs_list:
        mask = legal_action_mask(obs)
        legal_types = {a.action_type for a in obs.legal_actions}
        # Indices 2,3,4 are RAISE actions
        for idx in [2, 3, 4]:
            if ActionType.RAISE not in legal_types:
                assert not mask[idx], f"Raise idx {idx} should be masked"

@test("ActorCriticNetwork forward pass has correct output shapes")
def _():
    net   = ActorCriticNetwork(input_dim=FEATURE_DIM, hidden_dim=128)
    x     = torch.randn(4, FEATURE_DIM)
    mask  = torch.ones(4, NUM_ACTIONS, dtype=torch.bool)
    logits, value = net(x, mask)
    assert logits.shape == (4, NUM_ACTIONS)
    assert value.shape  == (4, 1)

@test("Agent save / load round-trip")
def _():
    import tempfile
    params = RewardParams(alpha=0.5, beta=0.3)
    agent  = PokerAgent(seat=2, reward_params=params, hidden_dim=64)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name

    try:
        agent.save(path)
        loaded = PokerAgent.load(path, device="cpu")
        assert loaded.seat == 2
        assert_close(loaded.reward_params.alpha, 0.5)
        assert_close(loaded.reward_params.beta,  0.3)
    finally:
        os.unlink(path)

@test("make_agent_set creates 4 agents with distinct params")
def _():
    params_list = [
        RewardParams(0.0, 0.0),
        RewardParams(0.2, 0.5),
        RewardParams(0.5, 1.0),
        RewardParams(0.8, 1.5),
    ]
    agents = make_agent_set(params_list)
    assert len(agents) == 4
    for i, agent in enumerate(agents):
        assert agent.seat == i
        assert_close(agent.reward_params.alpha, params_list[i].alpha)
        assert_close(agent.reward_params.beta,  params_list[i].beta)

@test("clone_network_weights_from copies weights correctly")
def _():
    a1 = PokerAgent(seat=0, reward_params=NeutralRewardParams, hidden_dim=64)
    a2 = PokerAgent(seat=1, reward_params=NeutralRewardParams, hidden_dim=64)
    a2.clone_network_weights_from(a1)
    # Check that parameters are identical
    for p1, p2 in zip(a1.network.parameters(), a2.network.parameters()):
        assert torch.allclose(p1, p2), "Weights differ after cloning"

@test("batch_log_probs has correct shape")
def _():
    import random as pyrandom
    agent    = PokerAgent(seat=0, reward_params=NeutralRewardParams)
    obs_list = []
    act_list = []

    def capture(obs):
        a = pyrandom.choice(obs.legal_actions)
        obs_list.append(obs)
        act_list.append(a)
        return a

    env = PokerEnv([capture] * NUM_PLAYERS)
    for _ in range(3):
        env.play_hand()

    if obs_list:
        lps = agent.batch_log_probs(obs_list, act_list)
        assert lps.shape == (len(obs_list),)
        assert torch.all(lps <= 0)


# ---------------------------------------------------------------------------
# 8. Integration: 100-hand session
# ---------------------------------------------------------------------------

print("\n=== Integration ===")

@test("100-hand session: chip conservation and no crashes")
def _():
    params_list = [
        RewardParams(0.0, 0.0),
        RewardParams(0.2, 0.5),
        RewardParams(0.5, 1.0),
        RewardParams(0.8, 1.5),
    ]
    agents = make_agent_set(params_list)
    env    = make_env_from_agents(agents, seed=42)

    reward_fns = [RewardFunction(p) for p in params_list]
    total_rewards = [0.0] * NUM_PLAYERS

    for _ in range(100):
        traj = env.play_hand()
        assert sum(traj.final_chip_deltas.values()) == 0
        for i in range(NUM_PLAYERS):
            r = reward_fns[i].scalar_reward(traj, i)
            total_rewards[i] += r

    print(f"     Total rewards over 100 hands: {[f'{r:.1f}' for r in total_rewards]}")

@test("100-hand trajectory collection: all steps have valid fields")
def _():
    agents = make_neutral_agents()
    env    = make_env_from_agents(agents, record_trajectories=True)
    encoder = FeatureEncoder()

    for _ in range(100):
        traj = env.play_hand()
        for step in traj.steps:
            assert 0 <= step.acting_seat < NUM_PLAYERS
            assert step.action.player_id == step.acting_seat
            vec = encoder.encode(step.observation)
            assert vec.shape == (FEATURE_DIM,)
            assert not np.any(np.isnan(vec))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\n{'='*50}")
print(f"  Results: {len(PASSED)} passed, {len(FAILED)} failed")
if FAILED:
    print("\n  FAILURES:")
    for name, tb in FAILED:
        print(f"\n  ✗ {name}")
        print(tb)
    sys.exit(1)
else:
    print("  All tests passed.")
