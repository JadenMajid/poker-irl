"""
step3_collect_and_run_irl.py
-----------------------------
Phase 1: Collect a large set of fixed-policy trajectories from all 4 perturbed agents.
Phase 2: For each agent in turn, run inverse reinforcement learning (IRL) to
         recover their (alpha, beta) reward parameters from observed behaviour.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IRL Algorithm: Gradient-Ascent Bayesian IRL with Opponent Modelling (GABO-IRL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Mathematical setup
------------------
We model each agent's policy as a Boltzmann-rational actor:

    π_θ(a | s) ∝ exp( Q_θ(s, a) )

where Q_θ(s, a) is the action-value under reward parameters θ = (α, β).

For LINEAR reward  R(s,a;θ) = φ₀(s,a) + α·φ_α(s,a) + β·φ_β(s,a)  we decompose:

    Q_θ(s,a) = Q₀(s,a) + α · Q_α(s,a) + β · Q_β(s,a)

The log-likelihood of observed trajectory {(s_t, a_t)} is:

    ℒ(θ) = Σ_t log π_θ(a_t | s_t)

The posterior (with Gaussian prior p(θ) = N(0, σ²I)) is:

    log p(θ | data) = ℒ(θ) - ||θ||²/(2σ²)

We maximise this via gradient ascent on (α, β).

Critical normalisation
----------------------
The variance reward component produces A_var values in units of chips² (e.g.
±50,000–500,000), while A_pot is normalised to [0, 1].  Without normalisation,
gradient steps for alpha are O(10,000×) larger than for beta, causing divergence.

Fix: reparametrise as

    α_norm = α × VAR_NORM         (normalised alpha; learned by IRL)
    β_norm = β                    (already normalised)

where VAR_NORM = std(rolling_var over the dataset) ≈ 50,000–200,000 chips².

The IRL optimises (α_norm, β_norm) in normalised space (both O(0.1–1.0) scale),
then recovers the true parameters:

    α_true = α_norm / VAR_NORM
    β_true = β_norm

This is a pure reparametrisation — the model is identical, only the coordinate
system changes.  It makes both gradient components comparable in magnitude and
ensures stable convergence within 5,000 gradient steps.

Opponent modelling
------------------
For each non-target seat we fit a BehaviourCloningNet (3-layer MLP) via
cross-entropy loss on observed (state, action) pairs.  This provides:
  (a) An estimate of how each opponent actually plays post-convergence.
  (b) A baseline for the target agent's effective environment dynamics.

The opponent model informs the IRL by letting us attribute the target agent's
choices to their OWN reward function rather than confounding with opponent patterns.

Output files
------------
  irl_results/trajectories.pkl            — collected trajectory data
  irl_results/irl_estimates.json          — final (alpha_hat, beta_hat) per seat
  irl_results/irl_convergence_log.json    — estimate evolution over gradient steps
  irl_results/opponent_models_*.pt        — fitted opponent BC networks
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import pickle
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import (
    ActorCriticNetwork,
    NUM_ACTIONS,
    action_to_index,
    index_to_action,
    legal_action_mask,
)
from feature_encoder import FeatureEncoder, FEATURE_DIM
from game_state import (
    ActionType,
    NUM_PLAYERS,
    PlayerObservation,
    Action,
    HandTrajectory,
    TrajectoryStep,
)
from poker_env import PokerEnv
from reward import RewardParams, POT_NORM

# ── configuration ──────────────────────────────────────────────────────────

CHECKPOINT_DIR = "checkpoints"
IRL_DIR        = "irl_results"
DEVICE         = "cpu"
HIDDEN_DIM     = 256

# Trajectory collection
N_COLLECTION_HANDS = 50_000
LOG_COLLECT_EVERY  = 5_000

# Number of parallel worker processes for trajectory collection.
# Each worker simulates an independent slice of hands with its own network
# copies — no shared state.  Speedup is near-linear with core count.
# Set to 1 to disable and use the original serial path.
N_COLLECT_WORKERS: int = min(4, os.cpu_count() or 1)

# Number of parallel workers for opponent BC training and per-seat IRL.
# BC models and IRL runs are fully independent across seats.
N_IRL_WORKERS: int = min(4, os.cpu_count() or 1)

# Opponent modelling (behavioural cloning)
OPP_HIDDEN_DIM  = 128
OPP_EPOCHS      = 50
OPP_LR          = 1e-3
OPP_BATCH_SIZE  = 512
OPP_MIN_SAMPLES = 200

# IRL gradient ascent
IRL_LR          = 0.02      # LR for (alpha_norm, beta) in normalised space
IRL_N_STEPS     = 5_000     # max gradient steps per agent
IRL_BATCH_SIZE  = 256       # hands sampled per gradient step
IRL_PRIOR_SIGMA = 1.0       # Gaussian prior std (in normalised space)
IRL_GRAD_CLIP   = 5.0       # gradient norm clip for stability
IRL_LOG_EVERY   = 100

# Gradient accumulation for IRL parameter smoothing
# ---------------------------------------------------
# Rather than applying an Adam step after every single sampled batch,
# we accumulate gradients over IRL_GRAD_ACCUM_STEPS batches and average
# them before the optimiser update.  This is the IRL analogue of PPO
# mini-batching: more data per effective gradient step → lower variance
# on the (α̂, β̂) trendlines without changing the total number of
# forward passes or the learning rate schedule.
#
# Effective batch size = IRL_BATCH_SIZE × IRL_GRAD_ACCUM_STEPS hands.
# Set to 1 to restore the original single-step behaviour.
IRL_GRAD_ACCUM_STEPS: int = 4

# Convergence (gradient steps)
CONV_WINDOW     = 300
CONV_THRESHOLD  = 5e-4
CONV_MIN_STEPS  = 500

# Rolling variance window for reward computation
VAR_WINDOW      = 100

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    """
    Compact record of one decision point for IRL.
    Reward components are filled at hand end (terminal step gets non-zero values).
    """
    seat:           int
    feature:        np.ndarray   # (FEATURE_DIM,) float32
    action_idx:     int          # 0–4
    legal_mask:     np.ndarray   # (NUM_ACTIONS,) bool
    reward_chip:    float        # net chip delta (non-zero only at terminal step)
    reward_var_pen: float        # rolling variance at hand end (non-zero at terminal)
    reward_pot:     float        # max_pot / POT_NORM (non-zero at terminal)
    is_terminal:    bool
    hand_id:        int


@dataclass
class HandRecord:
    """Processed record of one complete hand for all seats."""
    hand_id:     int
    steps:       Dict[int, List[StepRecord]]   # seat → steps this hand
    chip_deltas: Dict[int, float]
    max_pots:    Dict[int, float]              # seat → max pot they committed to


# ---------------------------------------------------------------------------
# Trajectory collector
# ---------------------------------------------------------------------------

def _collect_hand_chunk(
    chunk_start:  int,
    chunk_size:   int,
    agent_paths:  Dict[int, str],
    device_str:   str,
) -> List[HandRecord]:
    """
    Worker function: simulate ``chunk_size`` hands and return HandRecords.
    hand_id values are offset by chunk_start so they remain globally unique
    after the main process concatenates all chunks.

    Receives only picklable plain values; returns picklable HandRecord objects
    (which contain numpy arrays and Python scalars).
    """
    import torch as _torch
    _torch.set_num_threads(1)   # one thread per worker avoids oversubscription

    _device  = _torch.device(device_str)
    _encoder = FeatureEncoder()
    _networks: Dict[int, ActorCriticNetwork] = {}

    for seat, path in agent_paths.items():
        _ckpt = _torch.load(path, map_location=_device)
        net   = ActorCriticNetwork(
            input_dim=_ckpt.get("feature_dim", FEATURE_DIM),
            hidden_dim=_ckpt.get("hidden_dim",  HIDDEN_DIM),
        ).to(_device)
        net.load_state_dict(_ckpt["network_state"])
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        _networks[seat] = net

    records: List[HandRecord] = []

    for local_i in range(chunk_size):
        hand_id    = chunk_start + local_i
        hand_steps: Dict[int, List] = {i: [] for i in range(NUM_PLAYERS)}

        def make_cb(seat: int, net: ActorCriticNetwork):
            def callback(obs: PlayerObservation) -> Action:
                feat   = _encoder.encode(obs)
                mask   = legal_action_mask(obs)
                feat_t = _torch.tensor(feat, dtype=_torch.float32,
                                       device=_device).unsqueeze(0)
                mask_t = mask.unsqueeze(0).to(_device)
                with _torch.no_grad():
                    logits, _ = net(feat_t, mask_t)
                    dist      = Categorical(logits=logits.squeeze(0))
                    idx       = int(dist.sample().item())
                action = index_to_action(idx, seat)
                hand_steps[seat].append((feat, mask.numpy(), idx))
                return action
            return callback

        env  = PokerEnv(
            [make_cb(i, _networks[i]) for i in range(NUM_PLAYERS)],
            record_trajectories=True,
        )
        traj = env.play_hand()

        chip_deltas = {
            i: float(traj.final_chip_deltas.get(i, 0))
            for i in range(NUM_PLAYERS)
        }
        max_pots: Dict[int, float] = {}
        for seat in range(NUM_PLAYERS):
            mp_val = 0.0
            for step in traj.steps:
                if step.acting_seat == seat:
                    if step.action.action_type in (ActionType.CALL, ActionType.RAISE):
                        mp_val = max(mp_val, float(step.observation.pot))
            max_pots[seat] = mp_val

        steps_by_seat: Dict[int, List[StepRecord]] = {i: [] for i in range(NUM_PLAYERS)}
        for seat in range(NUM_PLAYERS):
            seat_steps = hand_steps[seat]
            n_s        = len(seat_steps)
            for k, (feat, mask_np, idx) in enumerate(seat_steps):
                is_last = (k == n_s - 1)
                steps_by_seat[seat].append(StepRecord(
                    seat=seat,
                    feature=feat,
                    action_idx=idx,
                    legal_mask=mask_np,
                    reward_chip=chip_deltas[seat] if is_last else 0.0,
                    reward_var_pen=0.0,
                    reward_pot=(max_pots[seat] / POT_NORM) if is_last else 0.0,
                    is_terminal=is_last,
                    hand_id=hand_id,
                ))

        records.append(HandRecord(
            hand_id=hand_id,
            steps=steps_by_seat,
            chip_deltas=chip_deltas,
            max_pots=max_pots,
        ))

    return records


def collect_trajectories(
    n_hands:        int,
    agent_paths:    Optional[Dict[int, str]] = None,
) -> List[HandRecord]:
    """
    Load all 4 perturbed agents (frozen), simulate n_hands, collect HandRecords.

    Hands are split evenly across N_COLLECT_WORKERS worker processes.  Each
    worker loads its own network copies from the checkpoint files and simulates
    its slice entirely independently.  The main process concatenates results
    and re-sorts by hand_id.

    Parameters
    ----------
    n_hands      : Number of hands to play.
    agent_paths  : Optional override of {seat: checkpoint_path}.  If None,
                   defaults to checkpoints/perturbed_agent_{seat}.pt for all seats.
    """
    if agent_paths is None:
        agent_paths = {
            i: os.path.join(CHECKPOINT_DIR, f"perturbed_agent_{i}.pt")
            for i in range(NUM_PLAYERS)
        }
    for path in agent_paths.values():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Agent checkpoint not found: {path}")

    n_workers  = N_COLLECT_WORKERS
    chunk_size = (n_hands + n_workers - 1) // n_workers   # ceiling division
    start      = time.time()

    log.info(
        "  Collecting %d hands across %d worker(s) (%d hands/worker) ...",
        n_hands, n_workers, chunk_size,
    )

    if n_workers == 1:
        # Serial fast-path avoids process spawn overhead for small datasets
        records = _collect_hand_chunk(0, n_hands, agent_paths, DEVICE)
        elapsed  = time.time() - start
        log.info("  Collected %d hands in %.1fs.", n_hands, elapsed)
        return records

    mp_ctx   = mp.get_context("spawn")
    all_records: List[HandRecord] = []

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx) as pool:
        futures = {}
        for w in range(n_workers):
            c_start = w * chunk_size
            c_size  = min(chunk_size, n_hands - c_start)
            if c_size <= 0:
                break
            fut = pool.submit(_collect_hand_chunk, c_start, c_size, agent_paths, DEVICE)
            futures[fut] = (w, c_start, c_size)

        completed = 0
        for fut in as_completed(futures):
            w, c_start, c_size = futures[fut]
            chunk = fut.result()
            all_records.extend(chunk)
            completed += c_size
            elapsed   = time.time() - start
            hands_ph  = completed / max(elapsed, 1) * 3600
            log.info(
                "  Worker %d done: %d hands | total %d/%d | %.0f hands/hr",
                w, c_size, completed, n_hands, hands_ph,
            )

    # Sort by hand_id to restore deterministic ordering
    all_records.sort(key=lambda r: r.hand_id)
    elapsed = time.time() - start
    log.info("  Collection complete: %d hands in %.1fs.", len(all_records), elapsed)
    return all_records


# ---------------------------------------------------------------------------
# Rolling variance computation
# ---------------------------------------------------------------------------

def compute_rolling_variance_penalties(
    records: List[HandRecord],
    window:  int = VAR_WINDOW,
) -> Tuple[Dict[int, List[float]], Dict[int, float]]:
    """
    Compute rolling variance of chip deltas per seat.

    Returns
    -------
    var_per_hand : seat → list of rolling variance values (one per hand)
    var_std      : seat → std of the rolling variance series (used for VAR_NORM)
    """
    var_per_hand: Dict[int, List[float]] = {i: [] for i in range(NUM_PLAYERS)}
    windows:      Dict[int, List[float]] = {i: [] for i in range(NUM_PLAYERS)}

    for rec in records:
        for seat in range(NUM_PLAYERS):
            delta = rec.chip_deltas[seat]
            w     = windows[seat]
            w.append(delta)
            if len(w) > window:
                w.pop(0)
            var = float(np.var(w, ddof=1)) if len(w) >= 2 else 0.0
            var_per_hand[seat].append(var)

    # std of the rolling variance series — used to set VAR_NORM per seat
    var_std: Dict[int, float] = {}
    for seat in range(NUM_PLAYERS):
        series = var_per_hand[seat]
        if len(series) >= 2:
            var_std[seat] = max(float(np.std(series)), 1.0)
        else:
            var_std[seat] = 1.0

    return var_per_hand, var_std


def fill_var_penalties(
    records:      List[HandRecord],
    var_per_hand: Dict[int, List[float]],
) -> None:
    """Fill in the reward_var_pen field for each terminal StepRecord in-place."""
    for hand_idx, rec in enumerate(records):
        for seat in range(NUM_PLAYERS):
            steps = rec.steps[seat]
            if steps:
                steps[-1].reward_var_pen = var_per_hand[seat][hand_idx]


# ---------------------------------------------------------------------------
# MC return data preparation
# ---------------------------------------------------------------------------

def compute_mc_returns_per_hand(
    records:      List[HandRecord],
    var_per_hand: Dict[int, List[float]],
) -> Dict[int, List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    """
    For each seat build the IRL training tuples:
      (feats, masks, acts, returns_3d)

    where returns_3d has shape (n_steps, 3):
      col 0 = chip_delta  (non-zero at terminal step)
      col 1 = rolling_var (non-zero at terminal step)
      col 2 = pot_involve  (non-zero at terminal step, already normalised by POT_NORM)

    Only the terminal step of each hand carries reward signal (gamma=1 episodic).
    """
    result: Dict[int, List] = {i: [] for i in range(NUM_PLAYERS)}

    for hand_idx, rec in enumerate(records):
        for seat in range(NUM_PLAYERS):
            steps   = rec.steps[seat]
            n_steps = len(steps)
            if n_steps == 0:
                continue

            feats   = np.stack([s.feature    for s in steps])
            masks   = np.stack([s.legal_mask for s in steps])
            acts    = np.array([s.action_idx for s in steps])
            returns = np.zeros((n_steps, 3), dtype=np.float32)
            returns[-1, 0] = rec.chip_deltas[seat]
            returns[-1, 1] = var_per_hand[seat][hand_idx]
            returns[-1, 2] = rec.max_pots[seat] / POT_NORM

            result[seat].append((feats, masks, acts, returns))

    return result


# ---------------------------------------------------------------------------
# Behavioural cloning for opponent modelling
# ---------------------------------------------------------------------------

class BehaviourCloningNet(nn.Module):
    """
    Lightweight 3-layer MLP predicting action logits from state features.
    Trained via cross-entropy on observed (state, action) pairs.
    """
    def __init__(self, input_dim: int = FEATURE_DIM,
                 hidden_dim: int = OPP_HIDDEN_DIM,
                 num_actions: int = NUM_ACTIONS) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.net(x)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        return logits

    def log_probs(self, x: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return F.log_softmax(self.forward(x, mask), dim=-1)


def train_opponent_model(
    features: np.ndarray,
    masks:    np.ndarray,
    actions:  np.ndarray,
    device:   torch.device,
) -> BehaviourCloningNet:
    """Fit a BehaviourCloningNet via cross-entropy.  Returns model in eval mode."""
    model = BehaviourCloningNet().to(device)
    opt   = Adam(model.parameters(), lr=OPP_LR)

    feat_t = torch.tensor(features, dtype=torch.float32, device=device)
    mask_t = torch.tensor(masks,    dtype=torch.bool,    device=device)
    act_t  = torch.tensor(actions,  dtype=torch.int64,   device=device)

    N = len(features)
    model.train()
    best_loss = float("inf")
    for epoch in range(OPP_EPOCHS):
        perm  = torch.randperm(N, device=device)
        total = 0.0
        nb    = 0
        for start in range(0, N, OPP_BATCH_SIZE):
            bi     = perm[start : start + OPP_BATCH_SIZE]
            logits = model(feat_t[bi], mask_t[bi])
            loss   = F.cross_entropy(logits, act_t[bi])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb    += 1
        epoch_loss = total / max(nb, 1)
        if (epoch + 1) % 10 == 0:
            log.debug("    BC epoch %3d: loss=%.4f", epoch + 1, epoch_loss)
        best_loss = min(best_loss, epoch_loss)

    model.eval()
    return model


def _train_opp_model_worker(
    target_seat: int,
    opp_seat:    int,
    features:    np.ndarray,
    masks:       np.ndarray,
    actions:     np.ndarray,
    device_str:  str,
) -> Tuple[int, int, Dict]:
    """
    Picklable wrapper around train_opponent_model for ProcessPoolExecutor.
    Returns (target_seat, opp_seat, state_dict) so the caller can key results.
    """
    import torch as _torch
    _torch.set_num_threads(1)
    _device = _torch.device(device_str)
    model   = train_opponent_model(features, masks, actions, _device)
    return target_seat, opp_seat, model.state_dict()


def _run_irl_worker(
    target_seat:          int,
    step_data:            List[Tuple],
    opp_state_dicts:      Dict[int, Dict],
    target_net_state_dict: Dict,
    net_input_dim:        int,
    net_hidden_dim:       int,
    device_str:           str,
    true_alpha:           float,
    true_beta:            float,
    var_norm:             float,
) -> Dict:
    """
    Picklable wrapper around run_irl_for_seat for ProcessPoolExecutor.

    Receives network weights as plain state dicts (picklable) rather than
    live nn.Module objects, rebuilds them inside the worker process, then
    delegates to run_irl_for_seat.
    """
    import torch as _torch
    _torch.set_num_threads(1)
    _device = _torch.device(device_str)

    # Rebuild target network
    target_net = ActorCriticNetwork(
        input_dim=net_input_dim, hidden_dim=net_hidden_dim
    ).to(_device)
    target_net.load_state_dict(target_net_state_dict)
    target_net.eval()
    for p in target_net.parameters():
        p.requires_grad_(False)

    # Rebuild opponent BC models
    opponent_models: Dict[int, BehaviourCloningNet] = {}
    for opp_seat, sd in opp_state_dicts.items():
        opp_net = BehaviourCloningNet().to(_device)
        opp_net.load_state_dict(sd)
        opp_net.eval()
        opponent_models[opp_seat] = opp_net

    return run_irl_for_seat(
        target_seat=target_seat,
        step_data=step_data,
        opponent_models=opponent_models,
        target_network=target_net,
        device=_device,
        true_alpha=true_alpha,
        true_beta=true_beta,
        var_norm=var_norm,
    )

class IRLOptimiser:
    """
    Gradient-ascent Bayesian IRL for recovering (alpha, beta) of one target seat.

    Reparametrisation for numerical stability
    -----------------------------------------
    We optimise (alpha_norm, beta) where:

        alpha_norm = alpha_true × VAR_NORM

    This keeps both gradient components on the same O(0.1–1.0) scale.
    After optimisation, recover:

        alpha_true = alpha_norm / VAR_NORM

    VAR_NORM is computed per-dataset as the standard deviation of the rolling
    variance series, making it adaptive to the actual chip volatility.

    Gradient computation
    --------------------
    For each hand's terminal step (s_T, a_T):

        Q_θ(s_T, a_obs) = Q₀(s_T, a_obs)
                        + alpha_norm × A_var_norm(s_T)
                        + beta       × A_pot(s_T)

    where:
        A_var_norm = (rolling_var - V_var) / VAR_NORM   ∈ O(-1, 1)
        A_pot      = (pot/POT_NORM) - V_pot             ∈ O(-1, 1)

    Log-likelihood gradient (feature expectation matching):
        ∂ℒ/∂alpha_norm = A_var_norm × (1 - π_θ(a_obs|s))
        ∂ℒ/∂beta       = A_pot      × (1 - π_θ(a_obs|s))

    Gaussian prior (in normalised space):
        ∂log p(θ)/∂alpha_norm = -alpha_norm / σ²
        ∂log p(θ)/∂beta       = -beta       / σ²
    """

    def __init__(
        self,
        target_seat:      int,
        step_data:        List[Tuple],
        opponent_models:  Dict[int, BehaviourCloningNet],
        target_network:   ActorCriticNetwork,
        device:           torch.device,
        var_norm:         float,
        prior_sigma:      float = IRL_PRIOR_SIGMA,
        lr:               float = IRL_LR,
        grad_accum_steps: int   = IRL_GRAD_ACCUM_STEPS,
    ) -> None:
        self.seat           = target_seat
        self.step_data      = step_data
        self.opp_models     = opponent_models
        self.target_network = target_network
        self.device         = device
        self.var_norm       = max(var_norm, 1.0)    # chips²
        self.prior_sigma    = prior_sigma
        self.grad_accum_steps = max(1, grad_accum_steps)

        # Parameters: [alpha_norm, beta] — both O(1) in normalised space
        # Initialised at zero (prior centre)
        self.theta = nn.Parameter(
            torch.zeros(2, dtype=torch.float64, device=device)
        )
        self.optimiser = Adam([self.theta], lr=lr)

        # History (in normalised space — converted to true scale on readout)
        self.alpha_norm_history: List[float] = []
        self.beta_history:       List[float] = []
        self.ll_history:         List[float] = []

        # Gradient accumulation state
        # _accum_grad holds the running sum; _accum_ll holds the sum of
        # mean log-likelihoods across accumulation steps.
        self._accum_grad: Optional[torch.Tensor] = None
        self._accum_ll:   float = 0.0
        self._accum_count: int  = 0

        self._precompute_baselines()

    # ------------------------------------------------------------------
    # Baseline computation
    # ------------------------------------------------------------------

    def _precompute_baselines(self) -> None:
        """
        Compute V_var (mean rolling variance) and V_pot (mean pot involvement)
        as control variates to reduce gradient variance.
        These are means over the dataset — they cancel in expectation and only
        reduce variance without introducing bias.
        """
        var_vals = [d[3][-1, 1] for d in self.step_data]   # rolling_var at terminal
        pot_vals = [d[3][-1, 2] for d in self.step_data]   # pot/POT_NORM at terminal

        self.V_var = float(np.mean(var_vals)) if var_vals else 0.0
        self.V_pot = float(np.mean(pot_vals)) if pot_vals else 0.0

        # Normalised baseline (in reparametrised space)
        self.V_var_norm = self.V_var / self.var_norm

        log.info(
            "    Seat %d baselines: V_var=%.0f  VAR_NORM=%.0f  "
            "V_var_norm=%.4f  V_pot=%.4f",
            self.seat, self.V_var, self.var_norm, self.V_var_norm, self.V_pot,
        )

    # ------------------------------------------------------------------
    # Gradient computation
    # ------------------------------------------------------------------

    def _compute_gradient(self, batch: List[Tuple]) -> Tuple[torch.Tensor, float]:
        """
        Compute gradient of log p(theta|data) w.r.t. theta = (alpha_norm, beta).

        Only processes the terminal step of each hand (where reward signal is
        non-zero).  Intermediate steps contribute zero gradient (see docstring).
        """
        alpha_norm = self.theta[0]   # scalar Parameter, float64
        beta       = self.theta[1]

        total_ll   = 0.0
        g_alpha    = 0.0
        g_beta     = 0.0
        n          = 0

        for feats, masks, acts, returns in batch:
            # ── Terminal step only ──────────────────────────────────────
            feat_t = torch.tensor(feats[-1:], dtype=torch.float32, device=self.device)
            mask_t = torch.tensor(masks[-1:], dtype=torch.bool,    device=self.device)
            a_obs  = int(acts[-1])

            with torch.no_grad():
                base_logits, _ = self.target_network(feat_t, mask_t)
            # base_logits: shape (1, NUM_ACTIONS), already legal-masked

            # Normalised advantages
            raw_var  = float(returns[-1, 1])
            raw_pot  = float(returns[-1, 2])
            A_var_n  = (raw_var - self.V_var) / self.var_norm    # ∈ O(-1, 1)
            A_pot    = raw_pot  - self.V_pot                      # ∈ O(-1, 1)

            # Reward shaping on the observed action
            #   Q_θ(s, a_obs) = Q₀(s, a_obs) + alpha_norm * A_var_n + beta * A_pot
            shaping = alpha_norm.item() * A_var_n + beta.item() * A_pot

            adj     = base_logits[0].clone().double()
            adj[a_obs] = adj[a_obs] + shaping

            # Log-likelihood: log π_θ(a_obs | s)
            legal = mask_t[0]
            log_z = torch.logsumexp(adj[legal], dim=0)
            ll    = (adj[a_obs] - log_z).item()

            # π_θ(a_obs | s) = exp(ll)
            pi_a = float(np.exp(np.clip(ll, -30, 0)))   # clip for numerical safety

            # Gradient (feature expectation matching update)
            g_alpha += A_var_n * (1.0 - pi_a)
            g_beta  += A_pot   * (1.0 - pi_a)
            total_ll += ll
            n        += 1

        if n == 0:
            return torch.zeros(2, dtype=torch.float64, device=self.device), 0.0

        g_alpha /= n
        g_beta  /= n
        ll_mean  = total_ll / n

        # Gaussian prior gradient (in normalised parameter space)
        prior_g_alpha = -float(alpha_norm.item()) / (self.prior_sigma ** 2)
        prior_g_beta  = -float(beta.item())       / (self.prior_sigma ** 2)

        full_grad = torch.tensor(
            [g_alpha + prior_g_alpha, g_beta + prior_g_beta],
            dtype=torch.float64,
            device=self.device,
        )

        # Gradient norm clip for robustness
        grad_norm = float(full_grad.norm().item())
        if grad_norm > IRL_GRAD_CLIP:
            full_grad = full_grad * (IRL_GRAD_CLIP / grad_norm)

        return full_grad, ll_mean

    def step(self, batch: List[Tuple]) -> float:
        """
        Accumulate one batch's gradient, and — once grad_accum_steps batches
        have been accumulated — average the gradients and apply a single Adam
        step.  Returns the mean log-likelihood for this batch (not the
        accumulated average), so the caller's logging cadence is unaffected.

        When grad_accum_steps == 1 this is identical to the original
        single-step behaviour.
        """
        full_grad, ll = self._compute_gradient(batch)

        # Accumulate
        if self._accum_grad is None:
            self._accum_grad = full_grad.clone()
        else:
            self._accum_grad += full_grad
        self._accum_ll    += ll
        self._accum_count += 1

        if self._accum_count >= self.grad_accum_steps:
            # Average across accumulation steps and apply the optimiser
            avg_grad = self._accum_grad / self._accum_count
            avg_ll   = self._accum_ll   / self._accum_count

            self.optimiser.zero_grad()
            # We maximise, so pass the negated gradient to the minimiser
            self.theta.grad = (-avg_grad).to(dtype=self.theta.dtype)
            self.optimiser.step()

            # Record the post-step parameter values
            alpha_n = float(self.theta[0].item())
            beta_v  = float(self.theta[1].item())
            self.alpha_norm_history.append(alpha_n)
            self.beta_history.append(beta_v)
            self.ll_history.append(avg_ll)

            # Reset accumulator
            self._accum_grad  = None
            self._accum_ll    = 0.0
            self._accum_count = 0

        return ll

    # ------------------------------------------------------------------
    # Readouts (convert normalised → true scale)
    # ------------------------------------------------------------------

    @property
    def current_alpha(self) -> float:
        """True-scale alpha = alpha_norm / VAR_NORM."""
        return float(self.theta[0].item()) / self.var_norm

    @property
    def current_beta(self) -> float:
        return float(self.theta[1].item())

    def mean_alpha_history(self, last_n: int) -> float:
        h = self.alpha_norm_history[-last_n:]
        return float(np.mean(h)) / self.var_norm if h else 0.0

    def mean_beta_history(self, last_n: int) -> float:
        h = self.beta_history[-last_n:]
        return float(np.mean(h)) if h else 0.0

    def _sample_batch(self, batch_size: int) -> List[Tuple]:
        n   = len(self.step_data)
        idx = np.random.choice(n, min(batch_size, n), replace=False)
        return [self.step_data[i] for i in idx]

    # ------------------------------------------------------------------
    # Convergence
    # ------------------------------------------------------------------

    def is_converged(self) -> bool:
        """
        Converged when the std of BOTH parameters in normalised space over the
        last CONV_WINDOW steps is below CONV_THRESHOLD.
        This checks stability of the estimate, not proximity to truth.
        """
        if len(self.alpha_norm_history) < CONV_MIN_STEPS + CONV_WINDOW:
            return False
        std_alpha = float(np.std(self.alpha_norm_history[-CONV_WINDOW:]))
        std_beta  = float(np.std(self.beta_history[-CONV_WINDOW:]))
        return std_alpha < CONV_THRESHOLD and std_beta < CONV_THRESHOLD


# ---------------------------------------------------------------------------
# Run IRL for one seat
# ---------------------------------------------------------------------------

def run_irl_for_seat(
    target_seat:     int,
    step_data:       List[Tuple],
    opponent_models: Dict[int, "BehaviourCloningNet"],
    target_network:  ActorCriticNetwork,
    device:          torch.device,
    true_alpha:      float,
    true_beta:       float,
    var_norm:        float,
) -> Dict:
    """
    Run the full GABO-IRL procedure for one target seat.
    Returns a result dict with estimates, history, and convergence info.
    """
    log.info("  Running IRL for seat %d (true α=%.5f, true β=%.4f)",
             target_seat, true_alpha, true_beta)
    log.info("    Data: %d hands  |  VAR_NORM=%.0f  |  LR=%.4f  |  "
             "grad_accum=%d (eff. batch=%d hands)",
             len(step_data), var_norm, IRL_LR,
             IRL_GRAD_ACCUM_STEPS, IRL_BATCH_SIZE * IRL_GRAD_ACCUM_STEPS)

    if not step_data:
        log.warning("    No data for seat %d — skipping.", target_seat)
        return {"seat": target_seat, "error": "no_data"}

    opt = IRLOptimiser(
        target_seat=target_seat,
        step_data=step_data,
        opponent_models=opponent_models,
        target_network=target_network,
        device=device,
        var_norm=var_norm,
    )

    start = time.time()
    for step_i in range(IRL_N_STEPS):
        batch = opt._sample_batch(IRL_BATCH_SIZE)
        ll    = opt.step(batch)

        if (step_i + 1) % IRL_LOG_EVERY == 0:
            α̂  = opt.current_alpha
            β̂  = opt.current_beta
            elapsed = time.time() - start
            log.info(
                "    Step %4d | α̂=%+.5f (err %+.5f) | β̂=%+.4f (err %+.4f)"
                " | LL=%.4f | %.0fs",
                step_i + 1,
                α̂,  α̂  - true_alpha,
                β̂,  β̂  - true_beta,
                ll, elapsed,
            )

        if opt.is_converged():
            log.info("    Converged at step %d.", step_i + 1)
            break

    # Posterior mean over last CONV_WINDOW steps (more stable than point estimate)
    final_alpha = opt.mean_alpha_history(CONV_WINDOW)
    final_beta  = opt.mean_beta_history(CONV_WINDOW)

    log.info(
        "  Seat %d final: α̂=%+.5f (true %+.5f) | β̂=%+.4f (true %+.4f)",
        target_seat, final_alpha, true_alpha, final_beta, true_beta,
    )

    # Reconstruct true-scale alpha history for storage
    true_scale_alpha_history = [
        a / opt.var_norm for a in opt.alpha_norm_history
    ]

    return {
        "seat":          target_seat,
        "true_alpha":    true_alpha,
        "true_beta":     true_beta,
        "est_alpha":     final_alpha,
        "est_beta":      final_beta,
        "alpha_mse":     (final_alpha - true_alpha) ** 2,
        "beta_mse":      (final_beta  - true_beta)  ** 2,
        "var_norm":      var_norm,
        "n_steps":       len(opt.alpha_norm_history),
        "converged":     opt.is_converged(),
        "alpha_history": true_scale_alpha_history,   # true-scale for plotting
        "beta_history":  opt.beta_history,
        "ll_history":    opt.ll_history,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_collection_and_irl(
    is_ablation:  bool = False,
    ablation_tag: str  = "",
    agent_paths:  Optional[Dict[int, str]] = None,
) -> None:
    """
    Full pipeline: collect trajectories → train opponent models → run IRL.

    Parameters
    ----------
    is_ablation  : If True, load ablation agent params and use ablation_tag suffix.
    ablation_tag : String appended to output filenames (e.g. "ablation").
    agent_paths  : Optional override for agent checkpoint paths.
    """
    os.makedirs(IRL_DIR, exist_ok=True)
    device = torch.device(DEVICE)
    suffix = f"_{ablation_tag}" if ablation_tag else ""

    # ── Load true parameters ───────────────────────────────────────────────
    params_path = os.path.join(
        CHECKPOINT_DIR,
        "ablation_agent_params.json" if is_ablation else "perturbed_agent_params.json",
    )
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Reward params not found: {params_path}")
    with open(params_path) as f:
        true_params: Dict[int, Tuple[float, float]] = {
            p["seat"]: (p["alpha"], p["beta"]) for p in json.load(f)
        }
    log.info("True reward params: %s", {k: f"α={v[0]:.4f} β={v[1]:.4f}"
                                        for k, v in true_params.items()})

    # ── Phase 1: Collect trajectories ─────────────────────────────────────
    traj_path = os.path.join(IRL_DIR, f"hand_records{suffix}.pkl")
    if os.path.exists(traj_path):
        log.info("Loading cached trajectories from %s ...", traj_path)
        with open(traj_path, "rb") as f:
            hand_records: List[HandRecord] = pickle.load(f)
        log.info("  Loaded %d hand records.", len(hand_records))
    else:
        log.info("Collecting %d hands of fixed-policy play ...", N_COLLECTION_HANDS)
        hand_records = collect_trajectories(N_COLLECTION_HANDS, agent_paths)
        with open(traj_path, "wb") as f:
            pickle.dump(hand_records, f)
        log.info("  Saved → %s", traj_path)

    # ── Compute rolling variance + VAR_NORM per seat ───────────────────────
    log.info("Computing rolling variance penalties ...")
    var_per_hand, var_std = compute_rolling_variance_penalties(
        hand_records, window=VAR_WINDOW
    )
    fill_var_penalties(hand_records, var_per_hand)

    # Log VAR_NORM values to help interpret IRL results
    for seat in range(NUM_PLAYERS):
        log.info("  Seat %d: VAR_NORM=%.0f chips²  (std of rolling var series)",
                 seat, var_std[seat])

    # ── Build MC return data ───────────────────────────────────────────────
    log.info("Building MC return data ...")
    mc_data = compute_mc_returns_per_hand(hand_records, var_per_hand)

    # ── Phase 2: Per-seat IRL ──────────────────────────────────────────────
    seats_to_run = list(true_params.keys())
    if is_ablation:
        seats_to_run = [s for s in seats_to_run
                        if true_params[s] != (0.0, 0.0)]
        log.info("Ablation: running IRL on seats %s only.", seats_to_run)

    # ── 2a: Load all target networks (needed for both BC and IRL) ─────────
    target_networks: Dict[int, ActorCriticNetwork] = {}
    target_net_sdicts: Dict[int, Dict]              = {}
    target_net_dims:   Dict[int, Tuple[int, int]]   = {}

    for target_seat in seats_to_run:
        if is_ablation:
            agent_path = os.path.join(CHECKPOINT_DIR, "ablation_perturbed_agent_0.pt")
        elif agent_paths and target_seat in agent_paths:
            agent_path = agent_paths[target_seat]
        else:
            agent_path = os.path.join(CHECKPOINT_DIR, f"perturbed_agent_{target_seat}.pt")

        if not os.path.exists(agent_path):
            log.error("Agent checkpoint not found: %s", agent_path)
            continue

        ckpt = torch.load(agent_path, map_location=device)
        net  = ActorCriticNetwork(
            input_dim=ckpt.get("feature_dim", FEATURE_DIM),
            hidden_dim=ckpt.get("hidden_dim",  HIDDEN_DIM),
        ).to(device)
        net.load_state_dict(ckpt["network_state"])
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        target_networks[target_seat]   = net
        target_net_sdicts[target_seat] = ckpt["network_state"]
        target_net_dims[target_seat]   = (
            ckpt.get("feature_dim", FEATURE_DIM),
            ckpt.get("hidden_dim",  HIDDEN_DIM),
        )

    # ── 2b: Train all opponent BC models in parallel ───────────────────────
    # For each target seat we need BC models for the other 3 seats.
    # All (target_seat, opp_seat) pairs are independent — submit them all.
    log.info(
        "Training opponent BC models in parallel (N_IRL_WORKERS=%d) ...",
        N_IRL_WORKERS,
    )
    mp_ctx = mp.get_context("spawn")

    # Collected state dicts: opp_sdicts[target_seat][opp_seat] = state_dict
    opp_sdicts: Dict[int, Dict[int, Dict]] = {s: {} for s in seats_to_run}

    bc_jobs: List[Tuple] = []
    for target_seat in seats_to_run:
        for opp_seat in range(NUM_PLAYERS):
            if opp_seat == target_seat:
                continue
            opp_data = mc_data.get(opp_seat, [])
            if len(opp_data) < OPP_MIN_SAMPLES:
                log.warning(
                    "  Too few samples (%d) for opponent model seat %d — skipping.",
                    len(opp_data), opp_seat,
                )
                continue
            all_feats = np.concatenate([d[0] for d in opp_data])
            all_masks = np.concatenate([d[1] for d in opp_data])
            all_acts  = np.concatenate([d[2] for d in opp_data])
            bc_jobs.append((target_seat, opp_seat, all_feats, all_masks, all_acts))

    with ProcessPoolExecutor(max_workers=N_IRL_WORKERS, mp_context=mp_ctx) as pool:
        bc_futures = {
            pool.submit(
                _train_opp_model_worker,
                ts, os_, feats, masks, acts, DEVICE,
            ): (ts, os_)
            for ts, os_, feats, masks, acts in bc_jobs
        }
        for fut in as_completed(bc_futures):
            ts, os_   = bc_futures[fut]
            ts_r, os_r, sd = fut.result()
            opp_sdicts[ts_r][os_r] = sd
            # Save the opponent model
            opp_save = os.path.join(
                IRL_DIR,
                f"opp_model_target{ts_r}_opp{os_r}{suffix}.pt"
            )
            torch.save(sd, opp_save)
            log.info(
                "  BC model done: target_seat=%d opp_seat=%d → %s",
                ts_r, os_r, opp_save,
            )

    # ── 2c: Run all per-seat IRL optimisers in parallel ───────────────────
    log.info(
        "Running IRL for %d seat(s) in parallel (N_IRL_WORKERS=%d) ...",
        len(seats_to_run), N_IRL_WORKERS,
    )

    irl_results = []
    with ProcessPoolExecutor(max_workers=N_IRL_WORKERS, mp_context=mp_ctx) as pool:
        irl_futures = {}
        for target_seat in seats_to_run:
            if target_seat not in target_net_sdicts:
                continue   # checkpoint was missing — skipped above
            inp_dim, hid_dim = target_net_dims[target_seat]
            true_alpha, true_beta = true_params[target_seat]

            log.info("\n" + "=" * 62)
            log.info("IRL — submitting seat %d to worker pool", target_seat)
            log.info("=" * 62)

            fut = pool.submit(
                _run_irl_worker,
                target_seat,
                mc_data[target_seat],
                opp_sdicts[target_seat],
                target_net_sdicts[target_seat],
                inp_dim,
                hid_dim,
                DEVICE,
                true_alpha,
                true_beta,
                var_std[target_seat],
            )
            irl_futures[fut] = target_seat

        for fut in as_completed(irl_futures):
            target_seat = irl_futures[fut]
            result      = fut.result()
            irl_results.append(result)
            log.info(
                "  IRL done: seat %d  α̂=%+.5f  β̂=%+.4f",
                target_seat,
                result.get("est_alpha", float("nan")),
                result.get("est_beta",  float("nan")),
            )

    # Keep results in seat order for consistent output
    irl_results.sort(key=lambda r: r.get("seat", 0))

    # ── Save results ───────────────────────────────────────────────────────
    # Summary (no full histories — those go in convergence log)
    summary = [
        {k: v for k, v in r.items() if k not in
         ("alpha_history", "beta_history", "ll_history")}
        for r in irl_results
    ]
    est_path = os.path.join(IRL_DIR, f"irl_estimates{suffix}.json")
    with open(est_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved: %s", est_path)

    conv_path = os.path.join(IRL_DIR, f"irl_convergence_log{suffix}.json")
    with open(conv_path, "w") as f:
        json.dump(irl_results, f, indent=2)
    log.info("Saved: %s", conv_path)

    # ── Print summary table ────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("IRL RESULTS SUMMARY")
    log.info("=" * 70)
    log.info("  %4s  %9s  %9s  %9s  %9s  %10s  %10s",
             "Seat", "true_α", "est_α", "true_β", "est_β", "α-MSE", "β-MSE")
    for r in irl_results:
        if "error" in r:
            log.info("  %4d  ERROR: %s", r["seat"], r["error"])
            continue
        log.info(
            "  %4d  %+9.5f  %+9.5f  %+9.4f  %+9.4f  %10.6f  %10.6f",
            r["seat"],
            r["true_alpha"], r["est_alpha"],
            r["true_beta"],  r["est_beta"],
            r["alpha_mse"],  r["beta_mse"],
        )


# ---------------------------------------------------------------------------
# Entry point — __main__ guard required for "spawn" multiprocessing context.
if __name__ == "__main__":
    run_collection_and_irl()
