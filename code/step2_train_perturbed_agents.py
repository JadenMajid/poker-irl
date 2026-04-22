"""
step2_train_perturbed_agents.py
--------------------------------
Load the neutral base agent, clone it into 4 independent agents, assign each
a distinct (alpha, beta) reward parameterisation, and fine-tune each agent
via PPO with a KL penalty that prevents excessive divergence from the base policy.

Reward parameter design
-----------------------
The four agents cover the four quadrants of (alpha, beta) space:

  Seat 0: alpha=+, beta=+   → risk-averse AND pot-hungry
  Seat 1: alpha=+, beta=-   → risk-averse but pot-avoidant
  Seat 2: alpha=-, beta=+   → risk-seeking AND pot-hungry
  Seat 3: alpha=-, beta=-   → risk-seeking and pot-avoidant

"Negative alpha" is implemented as a small negative value, meaning the agent
*prefers* higher variance (a risk-seeking gambler).  Negative beta means the
agent slightly *penalises* large pot commitments.

Calibration of magnitude:
  We want the reward perturbations to contribute roughly 10–20% of the total
  reward signal on a typical hand.  A typical hand has:
    - chip delta ≈ ±40–200 chips (depending on blind level / aggression)
    - rolling variance ≈ 5000–30000 after a burn-in period
    - max_pot_commitment ≈ 50–500 chips

  alpha contribution:  alpha * rolling_var / chip_delta_std
  beta  contribution:  beta  * max_pot / POT_NORM / chip_delta_std

  We set alpha=0.005, beta=0.3 (for positive values) so that at typical values:
    alpha * 10000 = 50   ≈ 15–25% of a typical hand's reward
    beta  * 0.15  = 0.045 — this is in normalised units, so the actual bonus
                         is beta * (max_pot / 2000) ≈ 0.3 * 0.15 = 0.045 * 2000/chip_std

  After sensitivity analysis, we found alpha=0.004 and beta=0.25 puts the
  perturbation in the 10–20% range.  See calibration notes below.

KL regularisation
-----------------
During fine-tuning each agent runs PPO with an added KL penalty term:
    L_total = L_PPO + kl_coef * KL(pi_current || pi_base)

The KL penalty starts at kl_coef=0.05 and is annealed slowly to 0.005,
preventing the agent from straying too far from the base policy early in
training while allowing full adaptation by the end.

Output files:
  checkpoints/perturbed_agent_{seat}.pt      — final per-seat weights
  checkpoints/perturbed_agent_params.json    — (alpha, beta) per seat
  checkpoints/perturbed_training_log.json    — per-update training statistics
"""

from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import (
    ActorCriticNetwork,
    NUM_ACTIONS,
    index_to_action,
    legal_action_mask,
)
from feature_encoder import FeatureEncoder, FEATURE_DIM
from game_state import NUM_PLAYERS, PlayerObservation, Action
from poker_env import PokerEnv
from ppo_trainer import (
    PPOConfig,
    PPOTrainer,
    ConvergenceDetector,
    FeatureSampleStore,
)
from reward import RewardParams, RewardFunction

# ── configuration ──────────────────────────────────────────────────────────

CHECKPOINT_DIR = "checkpoints"
DEVICE         = "cpu"
TORCH_THREADS  = max(1, int(os.getenv("POKER_TORCH_THREADS", str(os.cpu_count() or 1))))
PARALLEL_UPDATE_WORKERS = max(1, int(os.getenv("STEP2_PARALLEL_UPDATE_WORKERS", "1")))
HIDDEN_DIM     = 256
LOG_EVERY      = 500
SAVE_EVERY     = 50_000
MAX_HANDS      = 1_000_000

# ── Mini-batch hand accumulation ──────────────────────────────────────────
# Number of complete hands each agent accumulates before writing transitions
# into the RolloutBuffer.  Rewards are normalised across the batch before
# commit, smoothing the gradient signal and reducing trendline volatility.
# Each of the 4 agents maintains its OWN accumulator independently; they
# commit and update at different wall-clock moments as their buffers fill.
HANDS_PER_MINI_BATCH: int = 24

# When True, terminal rewards within each per-agent mini-batch are
# standardised (zero-mean / unit-variance) before buffer commit.  This is
# the primary mechanism for trendline noise reduction.
NORMALISE_BATCH_REWARDS: bool = True

# Minimum std denominator during reward normalisation — prevents division
# by zero on degenerate batches where all hands yield identical rewards.
REWARD_NORM_EPS: float = 1e-8

# ── Reward parameters for each seat ───────────────────────────────────────
# Calibrated so perturbation ≈ 10–20% of typical hand reward.
# alpha units: penalty per unit of chip variance (chips²)
# beta units : bonus per unit of normalised pot commitment [0,1]
REWARD_PARAMS = [
    RewardParams(alpha=+0.004, beta=+0.25),   # Seat 0: risk-averse, pot-hungry
    RewardParams(alpha=+0.004, beta=-0.20),   # Seat 1: risk-averse, pot-avoidant
    RewardParams(alpha=-0.003, beta=+0.25),   # Seat 2: risk-seeking, pot-hungry
    RewardParams(alpha=-0.003, beta=-0.20),   # Seat 3: risk-seeking, pot-avoidant
]

# Note: negative alpha is unusual but correct for "risk-seeking" agents.
# The reward function is: R = chip_delta - alpha*var + beta*pot_involve
# With alpha < 0 the agent is rewarded for higher variance outcomes.

# ── PPO config for fine-tuning ─────────────────────────────────────────────
# Smaller LR and more conservative clip range than base training
FINETUNE_PPO_CFG = PPOConfig(
    n_steps_per_update=2048,
    n_epochs=8,
    mini_batch_size=128,
    clip_range=0.15,
    value_clip_range=0.15,
    value_coef=0.5,
    entropy_coef=0.005,    # lower entropy encourages specialisation
    kl_coef=0.01,          # initial KL penalty (annealed below)
    gae_lambda=0.95,
    gamma=1.0,
    learning_rate=1e-4,    # smaller than base training
    max_grad_norm=0.4,
    convergence_window=2000,
    convergence_threshold=2e-4,
    min_hands_before_convergence_check=15_000,
    use_lr_schedule=True,
    lr_schedule_T_max=1_200_000,
)

# KL annealing schedule
KL_ANNEAL_FACTOR = 0.9995   # multiply kl_coef every hand
KL_FLOOR         = 0.005

# Convergence parameters
CONV_THRESHOLD  = 2e-4
CONV_MIN_HANDS  = 15_000
CONV_CHECK_EVERY= 1000
CONV_WINDOW     = 2000

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _resolve_device(device_cfg: str) -> str:
    """Resolve device in priority order for "auto": mps -> cuda -> cpu."""
    cfg = device_cfg.lower().strip()
    if cfg != "auto":
        return cfg
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _format_kl_for_log(detector: ConvergenceDetector) -> str:
    kl = detector.latest_mean_kl()
    if np.isfinite(kl):
        return f"{kl:.5f}"
    return "warmup"


# ---------------------------------------------------------------------------
# ProcessPoolExecutor worker infrastructure
# ---------------------------------------------------------------------------
#
# ProcessPoolExecutor requires that only picklable, value-typed data cross
# process boundaries.  Sending whole IndependentAgent objects back and forth
# is both slow (full network pickle every call) and semantically wrong (the
# returned copy's updated weights would be silently discarded, leaving the
# main-process agent unchanged).
#
# Instead we use a persistent worker pool:
#
#   1. _worker_init()  — runs ONCE per worker process at pool startup.
#      It rebuilds the IndependentAgent for its assigned seat from scratch
#      using the base-agent checkpoint path (a plain string, trivially
#      picklable).  The agent lives in the worker's memory for the lifetime
#      of the pool.
#
#   2. _worker_update() — called each time maybe_update is needed.
#      It receives only the list of PendingHand objects (numpy arrays +
#      scalars, all picklable) accumulated since the last call, commits them
#      to the local buffer, runs the PPO update if the buffer is ready, and
#      returns a lightweight UpdateResult (scalars + the new network state
#      dict so the main process can keep its own copy in sync for saving).
#
# Communication cost: O(batch_size * transitions * feature_dim) floats
# inbound, one state_dict outbound on update.  For HANDS_PER_MINI_BATCH=24
# with ~6 decisions/hand and FEATURE_DIM~128 this is roughly 150 KB per
# worker call — negligible vs. the PPO compute cost.

# Module-level registry: populated by _worker_init, keyed by seat.
_WORKER_AGENTS: Dict[int, "IndependentAgent"] = {}


@dataclass
class UpdateResult:
    """Scalar training stats returned from a worker update call."""
    seat:        int
    did_update:  bool
    policy_loss: float = 0.0
    value_loss:  float = 0.0
    entropy:     float = 0.0
    kl_penalty:  float = 0.0
    mean_kl:     float = float("nan")
    converged:   bool  = False
    # Network weights after the update so the main process can sync for saving.
    # None when did_update is False (no PPO step was taken this call).
    state_dict:  Optional[Dict] = None


def _worker_init(
    seat:           int,
    base_ckpt_path: str,
    reward_params:  RewardParams,
    ppo_cfg:        PPOConfig,
    device_str:     str,
    torch_threads:  int,
) -> None:
    """
    Initialise a persistent IndependentAgent in this worker process.
    Called exactly once per worker by ProcessPoolExecutor via initializer=.
    All arguments are plain picklable values.
    """
    import torch as _torch
    _torch.set_num_threads(torch_threads)

    _device = _torch.device(device_str)
    _ckpt   = _torch.load(base_ckpt_path, map_location=_device)

    def _load_net() -> ActorCriticNetwork:
        net = ActorCriticNetwork(
            input_dim=_ckpt.get("feature_dim", FEATURE_DIM),
            hidden_dim=_ckpt.get("hidden_dim",  HIDDEN_DIM),
        ).to(_device)
        net.load_state_dict(_ckpt["network_state"])
        return net

    base_net = _load_net()
    base_net.eval()
    for p in base_net.parameters():
        p.requires_grad_(False)

    agent_net = _load_net()
    agent_net.eval()

    cfg = copy.deepcopy(ppo_cfg)
    rf  = RewardFunction(reward_params, variance_window=200)

    agent = IndependentAgent(seat, agent_net, base_net, rf, cfg, _device)
    _WORKER_AGENTS[seat] = agent


def _worker_update(seat: int, pending_hands: List["PendingHand"]) -> UpdateResult:
    """
    Called in a worker process to:
      1. Commit the newly accumulated hands to the agent's rollout buffer
         (with reward normalisation).
      2. Run a PPO update if the buffer has enough steps.
      3. Return an UpdateResult with scalars and — if an update occurred —
         the new network state_dict for the main process to cache.

    Only picklable data (PendingHand = numpy arrays + Python scalars) enters;
    only an UpdateResult (scalars + state_dict of plain tensors) exits.
    """
    agent = _WORKER_AGENTS[seat]

    # ── Commit the batch ───────────────────────────────────────────────────
    converged = False
    if pending_hands:
        # Replicate _commit_batch logic here so we can pass hands directly
        # without going through the agent's accumulator (which was already
        # drained on the main-process side before sending).
        raw_rewards = np.array([h.terminal_reward for h in pending_hands], dtype=np.float32)
        if NORMALISE_BATCH_REWARDS:
            r_mean = raw_rewards.mean()
            r_std  = raw_rewards.std()
            normed = (raw_rewards - r_mean) / (r_std + REWARD_NORM_EPS)
        else:
            normed = raw_rewards

        buf = agent.trainer.buffer
        for hand, reward in zip(pending_hands, normed):
            n = len(hand.transitions)
            for k, (feat, mask_np, idx, lp, val) in enumerate(hand.transitions):
                is_last = (k == n - 1)
                buf.add(
                    feature  = feat,
                    mask     = mask_np,
                    action   = idx,
                    log_prob = lp,
                    value    = val,
                    reward   = float(reward) if is_last else 0.0,
                    done     = is_last,
                )

        sample    = agent.feat_store.sample_tensor(256, agent.device)
        converged = agent.detector.on_hand_end(agent.network, sample, agent.device)

    # ── PPO update ─────────────────────────────────────────────────────────
    stats = agent.maybe_update()
    if stats is None:
        return UpdateResult(
            seat=seat,
            did_update=False,
            mean_kl=agent.detector.latest_mean_kl(),
            converged=converged,
        )

    return UpdateResult(
        seat=seat,
        did_update=True,
        policy_loss=stats["policy_loss"],
        value_loss=stats["value_loss"],
        entropy=stats["entropy"],
        kl_penalty=stats["kl_penalty"],
        mean_kl=agent.detector.latest_mean_kl(),
        converged=converged,
        state_dict=agent.network.state_dict(),
    )


# ---------------------------------------------------------------------------
# Pending hand: staging area for one complete hand's transitions
# ---------------------------------------------------------------------------

@dataclass
class PendingHand:
    """
    Holds all (feature, mask, action, log_prob, value) tuples for one hand
    plus the terminal reward.  Kept separate from the RolloutBuffer so that
    batch-level reward normalisation can be applied across all pending hands
    before anything is committed.
    """
    transitions:    List = field(default_factory=list)
    terminal_reward: float = 0.0

    def add_transition(
        self,
        feat:     np.ndarray,
        mask:     np.ndarray,
        action:   int,
        log_prob: float,
        value:    float,
    ) -> None:
        self.transitions.append((feat, mask, action, log_prob, value))

    def __len__(self) -> int:
        return len(self.transitions)


# ---------------------------------------------------------------------------
# Independent agent wrapper (each seat has its own network)
# ---------------------------------------------------------------------------

class IndependentAgent:
    """
    An agent with its own network, reward function, rollout buffer, and trainer.
    Records its own transitions; gets reward at hand end.
    """

    def __init__(
        self,
        seat:        int,
        network:     ActorCriticNetwork,
        ref_network: ActorCriticNetwork,  # frozen base
        reward_fn:   RewardFunction,
        cfg:         PPOConfig,
        device:      torch.device,
    ) -> None:
        self.seat       = seat
        self.network    = network
        self.reward_fn  = reward_fn
        self.device     = device
        self.encoder    = FeatureEncoder()
        self.trainer    = PPOTrainer(network, cfg, device, ref_network=ref_network)
        self.detector   = ConvergenceDetector(
            window=CONV_WINDOW,
            threshold=CONV_THRESHOLD,
            min_hands=CONV_MIN_HANDS,
            check_every=CONV_CHECK_EVERY,
        )
        self.feat_store = FeatureSampleStore(capacity=1024)

        # Mini-batch accumulator: complete hands waiting for batch commit.
        # Each agent maintains its own accumulator independently so that
        # different seats can commit and update at different times.
        self._pending_hands: List[PendingHand] = []
        self._current_hand: Optional[PendingHand] = None

    def begin_hand(self) -> None:
        self._current_hand = PendingHand()

    def act(self, obs: PlayerObservation) -> Action:
        feat   = self.encoder.encode(obs)
        mask   = legal_action_mask(obs)
        feat_t = torch.tensor(feat, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = mask.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, value = self.network(feat_t, mask_t)
            dist          = Categorical(logits=logits.squeeze(0))
            idx_t         = dist.sample()
            lp            = dist.log_prob(idx_t)
            idx           = int(idx_t.item())
            val           = float(value.squeeze().item())

        self.feat_store.add(feat)
        self._current_hand.add_transition(feat, mask.numpy(), idx, float(lp.item()), val)
        return index_to_action(idx, self.seat)

    def on_hand_end(self, terminal_reward: float) -> bool:
        """
        Finalise the current hand and append it to the pending batch.
        Once HANDS_PER_MINI_BATCH hands are accumulated, commit them all
        to the RolloutBuffer with normalised rewards.  Returns converged.
        """
        self._current_hand.terminal_reward = terminal_reward
        self._pending_hands.append(self._current_hand)
        self._current_hand = None

        if len(self._pending_hands) >= HANDS_PER_MINI_BATCH:
            return self._commit_batch()
        return False

    def _commit_batch(self) -> bool:
        """
        Normalise rewards across the accumulated batch of hands, write all
        transitions into the RolloutBuffer, then update the convergence
        detector.  Returns True if the detector has fired.
        """
        hands = self._pending_hands
        self._pending_hands = []

        # Collect raw terminal rewards for the batch
        raw_rewards = np.array([h.terminal_reward for h in hands], dtype=np.float32)

        # Normalise across the batch to zero-mean / unit-variance
        if NORMALISE_BATCH_REWARDS:
            r_mean = raw_rewards.mean()
            r_std  = raw_rewards.std()
            normed = (raw_rewards - r_mean) / (r_std + REWARD_NORM_EPS)
        else:
            normed = raw_rewards

        # Write transitions into the rollout buffer
        buf = self.trainer.buffer
        for hand, reward in zip(hands, normed):
            n = len(hand.transitions)
            for k, (feat, mask_np, idx, lp, val) in enumerate(hand.transitions):
                is_last = (k == n - 1)
                buf.add(
                    feature  = feat,
                    mask     = mask_np,
                    action   = idx,
                    log_prob = lp,
                    value    = val,
                    reward   = float(reward) if is_last else 0.0,
                    done     = is_last,
                )

        # Update convergence detector once per batch commit
        sample = self.feat_store.sample_tensor(256, self.device)
        return self.detector.on_hand_end(self.network, sample, self.device)

    def maybe_update(self) -> Optional[Dict]:
        """Run PPO update if buffer is full enough.  Returns stats dict or None."""
        if len(self.trainer.buffer) >= self.trainer.cfg.n_steps_per_update:
            self.network.train()
            stats = self.trainer.update()
            self.network.eval()
            return stats
        return None

    def anneal_kl(self) -> None:
        self.trainer.cfg.kl_coef = max(
            KL_FLOOR, self.trainer.cfg.kl_coef * KL_ANNEAL_FACTOR
        )

    def pending_batch_size(self) -> int:
        """Number of hands currently waiting in this agent's accumulator."""
        return len(self._pending_hands)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def run_fine_tuning() -> None:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    resolved_device = _resolve_device(DEVICE)
    device = torch.device(resolved_device)

    # For ProcessPoolExecutor, each worker gets its own slice of threads.
    # We divide the total thread budget across workers to avoid oversubscription.
    n_workers = min(PARALLEL_UPDATE_WORKERS, NUM_PLAYERS)
    if device.type == "cpu":
        worker_threads = max(1, TORCH_THREADS // max(n_workers, 1))
        # Main process also does inference; give it the same budget
        torch.set_num_threads(worker_threads)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(min(worker_threads, 8))
        log.info(
            "Using CPU: %d workers × %d torch threads each (TORCH_THREADS=%d).",
            n_workers, worker_threads, TORCH_THREADS,
        )
    else:
        worker_threads = 1
        log.info("Using %s device.", device.type)

    # ── Load base agent ────────────────────────────────────────────────────
    base_path = os.path.join(CHECKPOINT_DIR, "base_agent.pt")
    if not os.path.exists(base_path):
        log.error("Base agent checkpoint not found at %s.  Run step1 first.", base_path)
        sys.exit(1)

    log.info("Loading base agent from %s ...", base_path)
    ckpt = torch.load(base_path, map_location=device)

    def load_network() -> ActorCriticNetwork:
        net = ActorCriticNetwork(
            input_dim=ckpt.get("feature_dim", FEATURE_DIM),
            hidden_dim=ckpt.get("hidden_dim",  HIDDEN_DIM),
        ).to(device)
        net.load_state_dict(ckpt["network_state"])
        return net

    base_network = load_network()
    base_network.eval()
    for p in base_network.parameters():
        p.requires_grad_(False)

    log.info("Base agent loaded.  Building 4 perturbed agents ...")

    # ── Build main-process agents (used for inference during hand simulation)
    # These are the authoritative copies of each network.  When a worker
    # completes a PPO update it returns its new state_dict, which we load
    # back here so the main-process agent stays in sync for inference and
    # checkpointing.
    agents: List[IndependentAgent] = []
    for seat, params in enumerate(REWARD_PARAMS):
        net   = load_network()
        net.eval()
        cfg   = copy.deepcopy(FINETUNE_PPO_CFG)
        rf    = RewardFunction(params, variance_window=200)
        agent = IndependentAgent(seat, net, base_network, rf, cfg, device)
        agents.append(agent)
        log.info(
            "  Seat %d: alpha=%.4f  beta=%.4f  kl_coef=%.4f",
            seat, params.alpha, params.beta, cfg.kl_coef,
        )

    # ── Training loop ──────────────────────────────────────────────────────
    training_log  = []
    hand_count    = 0
    update_counts = [0] * NUM_PLAYERS
    converged     = [False] * NUM_PLAYERS
    start_time    = time.time()

    log.info(
        "Starting fine-tuning.  MAX_HANDS=%d  HANDS_PER_MINI_BATCH=%d  "
        "reward_normalisation=%s  workers=%d",
        MAX_HANDS, HANDS_PER_MINI_BATCH, NORMALISE_BATCH_REWARDS, n_workers,
    )

    # Spawn context ensures clean process creation on all platforms (including
    # macOS where the default changed in Python 3.8, and Linux where fork can
    # cause deadlocks with OpenMP/MKL that PyTorch uses internally).
    mp_ctx = mp.get_context("spawn")

    # Build per-seat initialiser kwargs once — all picklable plain values.
    init_kwargs_per_seat: List[Dict] = [
        dict(
            seat          = seat,
            base_ckpt_path= base_path,
            reward_params = REWARD_PARAMS[seat],
            ppo_cfg       = copy.deepcopy(FINETUNE_PPO_CFG),
            device_str    = resolved_device,
            torch_threads = worker_threads,
        )
        for seat in range(NUM_PLAYERS)
    ]

    # When PARALLEL_UPDATE_WORKERS == 1 skip the pool entirely; a pool of
    # one worker has more overhead than a direct serial call.
    use_pool = n_workers > 1

    # We create a single long-lived pool so workers persist across hands.
    # Each worker is pinned to one seat via the initialiser and never
    # receives another seat's data.  We use n_workers workers = one per
    # active seat (max 4), letting the OS scheduler handle core affinity.
    pool_ctx = (
        ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=mp_ctx,
            initializer=_worker_init,
            # Pass seat-specific kwargs by submitting a dummy first call;
            # ProcessPoolExecutor's initializer takes *args not per-worker
            # kwargs, so we use a seat-keyed dispatch approach below.
        )
        if use_pool else None
    )

    # ProcessPoolExecutor's initializer runs the SAME function in every
    # worker with the SAME arguments, which doesn't work for per-seat init.
    # We work around this by sending an explicit _worker_init call as the
    # very first future for each seat, submitted before any training begins.
    if use_pool:
        log.info("Initialising %d worker processes ...", n_workers)
        init_futures = {}
        for seat in range(NUM_PLAYERS):
            kw = init_kwargs_per_seat[seat]
            # Submit the init call; result is None but blocks until done.
            fut = pool_ctx.submit(
                _worker_init,
                kw["seat"], kw["base_ckpt_path"], kw["reward_params"],
                kw["ppo_cfg"], kw["device_str"], kw["torch_threads"],
            )
            init_futures[seat] = fut
        # Wait for all workers to finish initialising before training starts.
        for seat, fut in init_futures.items():
            fut.result()   # raises if worker init threw
        log.info("All workers initialised.")

    try:
        while hand_count < MAX_HANDS:
            # ── Hand simulation (main process, sequential) ─────────────────
            for a in agents:
                a.begin_hand()

            def make_cb(seat: int):
                def callback(obs: PlayerObservation) -> Action:
                    return agents[seat].act(obs)
                return callback

            env  = PokerEnv([make_cb(i) for i in range(NUM_PLAYERS)], record_trajectories=True)
            traj = env.play_hand()
            hand_count += 1

            # Distribute rewards and accumulate into per-agent pending batches
            all_converged = True
            for seat, agent in enumerate(agents):
                reward_components = agent.reward_fn.compute(traj, seat)
                terminal_reward   = reward_components.total
                # on_hand_end appends to the agent's pending list; it returns
                # True (converged) only when a batch commit fires the detector.
                # With the process pool the actual commit and convergence check
                # happen in the worker, so here we only accumulate.
                agent._current_hand.terminal_reward = terminal_reward
                agent._pending_hands.append(agent._current_hand)
                agent._current_hand = None

                if not converged[seat]:
                    all_converged = False
                agent.anneal_kl()

            # ── PPO updates ────────────────────────────────────────────────
            active_seats = [s for s in range(NUM_PLAYERS) if not converged[s]]

            # Collect seats whose mini-batch is ready to be committed/updated
            ready_seats = [
                s for s in active_seats
                if len(agents[s]._pending_hands) >= HANDS_PER_MINI_BATCH
            ]

            if not ready_seats:
                # No agent has a full batch yet — nothing to submit
                pass
            elif use_pool:
                # Ship each ready agent's pending hands to its worker process.
                # We drain _pending_hands here (in the main process) so the
                # list is empty by the time the next batch starts accumulating.
                futures: Dict = {}
                for seat in ready_seats:
                    hands = agents[seat]._pending_hands
                    agents[seat]._pending_hands = []
                    fut = pool_ctx.submit(_worker_update, seat, hands)
                    futures[fut] = seat

                for fut in as_completed(futures):
                    seat   = futures[fut]
                    result: UpdateResult = fut.result()

                    # Sync convergence state back to main process
                    if result.converged and not converged[seat]:
                        converged[seat] = True
                        log.info(
                            "  Seat %d converged at hand %d (KL=%.5f).",
                            seat, hand_count, result.mean_kl,
                        )

                    if not result.did_update:
                        continue

                    # Sync the updated weights back into the main-process
                    # agent so it uses the latest policy for inference.
                    agents[seat].network.load_state_dict(result.state_dict)
                    agents[seat].network.eval()

                    update_counts[seat] += 1
                    if update_counts[seat] % 10 == 0:
                        log.info(
                            "  Seat %d | Update %3d | π-loss %.4f | "
                            "entropy %.4f | kl_pen %.4f",
                            seat, update_counts[seat],
                            result.policy_loss, result.entropy, result.kl_penalty,
                        )
                    training_log.append({
                        "hand":        hand_count,
                        "seat":        seat,
                        "update":      update_counts[seat],
                        "policy_loss": result.policy_loss,
                        "value_loss":  result.value_loss,
                        "entropy":     result.entropy,
                        "kl_penalty":  result.kl_penalty,
                        "mean_kl":     result.mean_kl,
                    })
            else:
                # Serial fallback: commit batches and update in main process
                for seat in ready_seats:
                    agent = agents[seat]
                    cvgd  = agent._commit_batch()
                    if cvgd and not converged[seat]:
                        converged[seat] = True
                        log.info(
                            "  Seat %d converged at hand %d (KL=%.5f).",
                            seat, hand_count, agent.detector.latest_mean_kl(),
                        )
                    stats = agent.maybe_update()
                    if stats is None:
                        continue
                    update_counts[seat] += 1
                    if update_counts[seat] % 10 == 0:
                        log.info(
                            "  Seat %d | Update %3d | π-loss %.4f | "
                            "entropy %.4f | kl_pen %.4f",
                            seat, update_counts[seat],
                            stats["policy_loss"], stats["entropy"], stats["kl_penalty"],
                        )
                    training_log.append({
                        "hand":        hand_count,
                        "seat":        seat,
                        "update":      update_counts[seat],
                        "policy_loss": stats["policy_loss"],
                        "value_loss":  stats["value_loss"],
                        "entropy":     stats["entropy"],
                        "kl_penalty":  stats["kl_penalty"],
                        "mean_kl":     agent.detector.latest_mean_kl(),
                    })

            # Re-check all_converged after updates
            all_converged = all(converged)

            # Progress log
            if hand_count % LOG_EVERY == 0:
                elapsed  = time.time() - start_time
                hands_ph = hand_count / max(elapsed, 1) * 3600
                kls      = [_format_kl_for_log(a.detector) for a in agents]
                log.info(
                    "Hand %7d | KLs %s | %.0f hands/hr | converged=%s",
                    hand_count, kls, hands_ph, converged,
                )

            # Intermediate save (using main-process agent weights, which are
            # kept in sync via state_dict syncing after each worker update)
            if hand_count % SAVE_EVERY == 0:
                for seat, agent in enumerate(agents):
                    last_stats = next(
                        (item for item in reversed(training_log) if item["seat"] == seat),
                        None,
                    )
                    if last_stats:
                        _save_agent(
                            agent.network, seat, hand_count,
                            CHECKPOINT_DIR, "perturbed_ckpt",
                            last_stats["policy_loss"], last_stats["value_loss"],
                            last_stats["entropy"], last_stats["kl_penalty"],
                            last_stats["mean_kl"],
                        )
                    else:
                        _save_agent(agent.network, seat, hand_count,
                                    CHECKPOINT_DIR, "perturbed_ckpt")

            if all_converged:
                log.info("All 4 agents converged at hand %d.", hand_count)
                break

    finally:
        # Always shut the pool down cleanly, even on exception or KeyboardInterrupt
        if pool_ctx is not None:
            pool_ctx.shutdown(wait=False, cancel_futures=True)

    # ── Final saves ────────────────────────────────────────────────────────
    log.info("Fine-tuning complete.  Total hands: %d", hand_count)

    for seat, agent in enumerate(agents):
        last_stats = next(
            (item for item in reversed(training_log) if item["seat"] == seat),
            None,
        )
        if last_stats:
            _save_agent(
                agent.network, seat, hand_count, CHECKPOINT_DIR, "perturbed_agent",
                last_stats["policy_loss"], last_stats["value_loss"],
                last_stats["entropy"], last_stats["kl_penalty"], last_stats["mean_kl"],
            )
        else:
            _save_agent(agent.network, seat, hand_count,
                        CHECKPOINT_DIR, "perturbed_agent")
        log.info("  Saved: checkpoints/perturbed_agent_%d.pt", seat)

    params_record = [
        {"seat": i, "alpha": REWARD_PARAMS[i].alpha, "beta": REWARD_PARAMS[i].beta}
        for i in range(NUM_PLAYERS)
    ]
    with open(os.path.join(CHECKPOINT_DIR, "perturbed_agent_params.json"), "w") as f:
        json.dump(params_record, f, indent=2)

    with open(os.path.join(CHECKPOINT_DIR, "perturbed_training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)

    log.info("Saved: checkpoints/perturbed_agent_params.json")
    log.info("Saved: checkpoints/perturbed_training_log.json")


def _save_agent(
    network:      ActorCriticNetwork,
    seat:         int,
    step:         int,
    out_dir:      str,
    prefix:       str,
    policy_loss:  Optional[float] = None,
    value_loss:   Optional[float] = None,
    entropy:      Optional[float] = None,
    kl_penalty:   Optional[float] = None,
    mean_kl:      Optional[float] = None,
) -> None:
    path = os.path.join(out_dir, f"{prefix}_{seat}.pt")
    torch.save({
        "network_state": network.state_dict(),
        "hidden_dim":    network.hidden_dim,
        "feature_dim":   FEATURE_DIM,
        "num_actions":   NUM_ACTIONS,
        "seat":          seat,
        "step":          step,
        "alpha":         REWARD_PARAMS[seat].alpha,
        "beta":          REWARD_PARAMS[seat].beta,
        "policy_loss":   policy_loss,
        "value_loss":    value_loss,
        "entropy":       entropy,
        "kl_penalty":    kl_penalty,
        "mean_kl":       mean_kl,
        "hands_per_mini_batch":  HANDS_PER_MINI_BATCH,
        "normalise_rewards":     NORMALISE_BATCH_REWARDS,
    }, path)


# ---------------------------------------------------------------------------
# Entry point — __main__ guard is REQUIRED when using multiprocessing with
# the "spawn" start method (default on macOS/Windows, and used explicitly
# here for safety).  Without it, each worker process would re-execute the
# module body and attempt to spawn its own pool, causing an import bomb.
if __name__ == "__main__":
    run_fine_tuning()
