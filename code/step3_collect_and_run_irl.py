"""
step3_collect_and_run_irl.py
-----------------------------
Phase 1: Collect a large set of fixed-policy trajectories from all 4 perturbed agents.
Phase 2: For each agent in turn, run inverse reinforcement learning (IRL) to
         recover their (alpha, beta) reward parameters from observed behaviour.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IRL Algorithm: Bayesian Full-Action Policy-Shift IRL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Why this formulation
--------------------
The previous version in this file adjusted only the observed action logit at
terminal steps, which creates a known identifiability failure: likelihood can
increase monotonically in one parameter direction (especially beta), yielding
biased signs and magnitudes.

This version uses a proper conditional action likelihood over ALL legal actions
at every decision point.  We treat the neutral base policy as the reference
logit function and model reward-parameter effects as an additive policy shift:

    logit_θ(s, a) = logit_ref(s, a)
                    + alpha * φ_alpha(s, a)
                    + beta  * φ_beta(s, a)

where:
  - φ_beta is a pot-involvement proxy derived from post-action pot size.
  - φ_alpha is a risk proxy based on squared immediate commitment.

We then maximise a Gaussian-prior posterior objective:

    log p(θ | D) = Σ_t log π_θ(a_t | s_t)
                   - 1/2 * [(alpha/σ_α)^2 + (beta/σ_β)^2]

This remains an approximation to full RL fixed-point IRL, but is internally
consistent and avoids one-action shaping artefacts.

Output files
------------
  irl_results/trajectories.pkl            — collected trajectory data
  irl_results/irl_estimates.json          — final (alpha_hat, beta_hat) per seat
  irl_results/irl_convergence_log.json    — estimate evolution over gradient steps
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
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
    index_to_action,
    legal_action_mask,
)
from feature_encoder import FeatureEncoder, FEATURE_DIM
from game_state import (
    ActionType,
    FIXED_RAISE_SIZES,
    NUM_PLAYERS,
    PlayerObservation,
    Action,
)
from poker_env import PokerEnv
from reward import POT_NORM

# ── configuration ──────────────────────────────────────────────────────────

CHECKPOINT_DIR = "checkpoints"
IRL_DIR        = "irl_results"
DEVICE         = "cpu"
HIDDEN_DIM     = 256

# Trajectory collection
N_COLLECTION_HANDS = 100_000
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
IRL_LR          = 0.001        # LR for (alpha, beta)
IRL_N_STEPS     = 50_000     # max gradient steps per agent
IRL_BATCH_SIZE  = 1024*2**6      # decision-states sampled per gradient step
IRL_PRIOR_SIGMA_ALPHA = 0.02
IRL_PRIOR_SIGMA_BETA  = 0.60
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

# Feature indices from feature_encoder.py layout.
# We only need pot and call amount for action-feature construction.
FEAT_IDX_POT  = 102
FEAT_IDX_CALL = 103

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
    reward_pot:     float        # max_pot (non-zero at terminal)
    is_terminal:    bool
    hand_id:        int
    p_max:          float = 0.0


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
            state = {"p_max": 0.0}
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
                current_p_max = state["p_max"]
                hand_steps[seat].append((feat, mask.numpy(), idx, current_p_max))
                if idx > 0:
                    state["p_max"] = max(state["p_max"], float(obs.pot))
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
            for k, (feat, mask_np, idx, p_max) in enumerate(seat_steps):
                is_last = (k == n_s - 1)
                steps_by_seat[seat].append(StepRecord(
                    seat=seat,
                    feature=feat,
                    action_idx=idx,
                    legal_mask=mask_np,
                    reward_chip=chip_deltas[seat] if is_last else 0.0,
                    reward_var_pen=0.0,
                    reward_pot=float(max_pots[seat]) if is_last else 0.0,
                    is_terminal=is_last,
                    hand_id=hand_id,
                    p_max=p_max,
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
      col 2 = pot_involve  (non-zero at terminal step, max_pot chips)

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
            p_maxes = np.array([s.p_max for s in steps], dtype=np.float32)
            returns = np.zeros((n_steps, 4), dtype=np.float32)
            returns[-1, 0] = rec.chip_deltas[seat]
            returns[-1, 1] = var_per_hand[seat][hand_idx]
            returns[-1, 2] = rec.max_pots[seat]
            returns[:, 3] = p_maxes

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
    target_net_state_dict: Dict,
    net_input_dim:        int,
    net_hidden_dim:       int,
    device_str:           str,
    true_alpha:           float,
    true_beta:            float,
    var_norm:             float,
    S:                    float,
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

    return run_irl_for_seat(
        target_seat=target_seat,
        step_data=step_data,
        opponent_models={},
        target_network=target_net,
        device=_device,
        true_alpha=true_alpha,
        true_beta=true_beta,
        var_norm=var_norm,
        S=S,
    )

class IRLOptimiser:
    """
    Bayesian posterior optimiser over (alpha, beta) using full-action likelihood.

    Key fixes versus the previous implementation:
      1) Uses a neutral reference policy (Q0) instead of the target policy itself.
      2) Applies reward shaping to ALL legal actions, not only the observed action.
      3) Fits on all decision points, not just terminal actions.

    We model policy logits as:
        logit_θ(s, a) = logit_ref(s, a)
                        + alpha * φ_alpha(s, a)
                        + beta  * φ_beta(s, a)

    where action features are constructed from observable commitment proxies:
      - φ_beta: pot-involvement proxy via post-action pot size.
      - φ_alpha: risk proxy via squared immediate commitment (negative sign so
                 alpha>0 is risk-averse).

    This is an approximation to the full RL fixed-point, but unlike the old
    estimator it is a proper conditional likelihood model and avoids monotone
    one-action shaping artefacts.
    """

    def __init__(
        self,
        target_seat:      int,
        step_data:        List[Tuple],
        opponent_models:  Dict[int, BehaviourCloningNet],
        target_network:   ActorCriticNetwork,
        device:           torch.device,
        var_norm:         float,
        S:                float,
        lr:               float = IRL_LR,
        grad_accum_steps: int   = IRL_GRAD_ACCUM_STEPS,
        init_theta:       Optional[Tuple[float, float]] = None,
    ) -> None:
        self.seat           = target_seat
        self.step_data      = step_data
        self.opp_models     = opponent_models
        self.reference_network = target_network
        self.device         = device
        self.var_norm       = max(var_norm, 1.0)
        self.S              = max(S, 1.0)
        # We scale features so the optimal theta is roughly 1.0.
        # This prevents Adam from just accumulating constant steps (if too small) or exploding (if too big).
        self.prior_sigma_alpha = 10.0
        self.prior_sigma_beta  = 10.0

        init_alpha = 0.0 if init_theta is None else float(init_theta[0])
        init_beta  = 0.0 if init_theta is None else float(init_theta[1])
        self.theta = nn.Parameter(
            torch.tensor([init_alpha, init_beta], dtype=torch.float64, device=device)
        )
        self.optimiser = Adam([self.theta], lr=lr)

        # History in true parameter space.
        self.alpha_history: List[float] = []
        self.beta_history:  List[float] = []
        self.ll_history:    List[float] = []

        self._prepare_state_tensors()

    def _prepare_state_tensors(self) -> None:
        """Flatten hand-level tuples into per-decision tensors once."""
        feats_list: List[np.ndarray] = []
        masks_list: List[np.ndarray] = []
        acts_list:  List[np.ndarray] = []
        p_max_list: List[np.ndarray] = []

        for feats, masks, acts, returns in self.step_data:
            if len(acts) == 0:
                continue
            feats_list.append(np.asarray(feats, dtype=np.float32))
            masks_list.append(np.asarray(masks, dtype=bool))
            acts_list.append(np.asarray(acts, dtype=np.int64))
            p_max_list.append(np.asarray(returns[:, 3], dtype=np.float64))

        if not feats_list:
            self.n_states = 0
            self.features = torch.empty((0, FEATURE_DIM), dtype=torch.float32, device=self.device)
            self.masks = torch.empty((0, NUM_ACTIONS), dtype=torch.bool, device=self.device)
            self.actions = torch.empty((0,), dtype=torch.int64, device=self.device)
            self.p_max = torch.empty((0,), dtype=torch.float64, device=self.device)
            self.base_logits = torch.empty((0, NUM_ACTIONS), dtype=torch.float64, device=self.device)
            self.phi_alpha = torch.empty((0, NUM_ACTIONS), dtype=torch.float64, device=self.device)
            self.phi_beta = torch.empty((0, NUM_ACTIONS), dtype=torch.float64, device=self.device)
            self.alpha_feature_contrast = 0.0
            self.beta_feature_contrast = 0.0
            self.eval_idx = torch.empty((0,), dtype=torch.int64, device=self.device)
            return

        feats_np = np.concatenate(feats_list, axis=0)
        masks_np = np.concatenate(masks_list, axis=0)
        acts_np  = np.concatenate(acts_list, axis=0)
        p_max_np = np.concatenate(p_max_list, axis=0)

        self.n_states = int(len(acts_np))
        self.features = torch.tensor(feats_np, dtype=torch.float32, device=self.device)
        self.masks    = torch.tensor(masks_np, dtype=torch.bool, device=self.device)
        self.actions  = torch.tensor(acts_np, dtype=torch.int64, device=self.device)
        self.p_max    = torch.tensor(p_max_np, dtype=torch.float64, device=self.device)

        with torch.no_grad():
            base_logits, _ = self.reference_network(self.features, self.masks)
        self.base_logits = base_logits.to(dtype=torch.float64)

        if self.features.shape[1] <= FEAT_IDX_CALL:
            raise ValueError(
                f"Feature dimension {self.features.shape[1]} is too small for pot/call indices."
            )

        pot_now  = self.features[:, FEAT_IDX_POT].to(dtype=torch.float64) * POT_NORM
        call_amt = self.features[:, FEAT_IDX_CALL].to(dtype=torch.float64) * POT_NORM

        zero = torch.zeros_like(call_amt)
        r20  = float(FIXED_RAISE_SIZES[0])
        r100 = float(FIXED_RAISE_SIZES[1])
        r500 = float(FIXED_RAISE_SIZES[2])
        commit = torch.stack(
            [
                zero,
                call_amt,
                call_amt + r20,
                call_amt + r100,
                call_amt + r500,
            ],
            dim=1,
        )

        # Action-conditioned reward proxies.
        max_pot_expected = torch.empty((self.n_states, NUM_ACTIONS), dtype=torch.float64, device=self.device)
        max_pot_expected[:, 0] = self.p_max
        max_pot_expected[:, 1] = pot_now
        max_pot_expected[:, 2] = pot_now
        max_pot_expected[:, 3] = pot_now
        max_pot_expected[:, 4] = pot_now

        phi_beta_raw = max_pot_expected

        var_proxy = torch.empty((self.n_states, NUM_ACTIONS), dtype=torch.float64, device=self.device)
        var_proxy[:, 0] = 0.0
        var_proxy[:, 1] = ((pot_now + commit[:, 1]) ** 2) / 4.0
        var_proxy[:, 2] = ((pot_now + commit[:, 2]) ** 2) / 4.0
        var_proxy[:, 3] = ((pot_now + commit[:, 3]) ** 2) / 4.0
        var_proxy[:, 4] = ((pot_now + commit[:, 4]) ** 2) / 4.0

        # Negative sign because logit = base - alpha * phi_alpha + beta * phi_beta
        # So we want -alpha * var_proxy, meaning phi_alpha should be var_proxy
        # But wait, objective uses + theta[0] * phi_alpha, so phi_alpha must be -var_proxy
        phi_alpha_raw = -var_proxy

        # De-mean across legal actions per state to keep only relative preferences.
        legal = self.masks.to(dtype=torch.float64)
        denom = legal.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_alpha = (phi_alpha_raw * legal).sum(dim=1, keepdim=True) / denom
        mean_beta  = (phi_beta_raw  * legal).sum(dim=1, keepdim=True) / denom

        centered_alpha = phi_alpha_raw - mean_alpha
        centered_beta  = phi_beta_raw  - mean_beta

        var_alpha = ((centered_alpha * centered_alpha) * legal).sum(dim=1) / denom.squeeze(1)
        var_beta  = ((centered_beta  * centered_beta)  * legal).sum(dim=1) / denom.squeeze(1)
        self.alpha_feature_contrast = float(torch.sqrt(var_alpha).mean().item())
        self.beta_feature_contrast  = float(torch.sqrt(var_beta).mean().item())

        self.phi_alpha = centered_alpha
        self.phi_beta  = centered_beta

        n_eval = min(50_000, self.n_states)
        if n_eval == self.n_states:
            self.eval_idx = torch.arange(self.n_states, dtype=torch.int64, device=self.device)
        else:
            self.eval_idx = torch.randperm(self.n_states, device=self.device)[:n_eval]

    def _sample_batch_indices(self, batch_size: int) -> torch.Tensor:
        n = min(batch_size, self.n_states)
        return torch.randint(0, self.n_states, (n,), device=self.device)

    def _posterior_objective(
        self,
        idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = (
            self.base_logits[idx]
            + self.theta[0] * self.phi_alpha[idx]
            + self.theta[1] * self.phi_beta[idx]
        )
        logits = logits.masked_fill(~self.masks[idx], float("-inf"))

        dist = Categorical(logits=logits)
        ll = dist.log_prob(self.actions[idx]).mean()
        prior = -0.5 * (
            (self.theta[0] / self.prior_sigma_alpha) ** 2
            + (self.theta[1] / self.prior_sigma_beta) ** 2
        )
        return ll + prior, ll

    def step(self, batch: List[Tuple]) -> float:
        del batch  # kept for API compatibility with caller
        if self.n_states == 0:
            return 0.0

        idx = self._sample_batch_indices(IRL_BATCH_SIZE)
        objective, ll = self._posterior_objective(idx)
        loss = -objective

        self.optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([self.theta], IRL_GRAD_CLIP)
        self.optimiser.step()

        self.alpha_history.append(float(self.theta[0].item()))
        self.beta_history.append(float(self.theta[1].item()))
        self.ll_history.append(float(ll.item()))
        return float(ll.item())

    # ------------------------------------------------------------------
    # Readouts (convert normalised → true scale)
    # ------------------------------------------------------------------

    @property
    def current_alpha(self) -> float:
        return float(self.theta[0].item())

    @property
    def current_beta(self) -> float:
        return float(self.theta[1].item())

    def mean_alpha_history(self, last_n: int) -> float:
        h = self.alpha_history[-last_n:]
        return float(np.mean(h)) if h else 0.0

    def mean_beta_history(self, last_n: int) -> float:
        h = self.beta_history[-last_n:]
        return float(np.mean(h)) if h else 0.0

    def _sample_batch(self, batch_size: int) -> List[Tuple]:
        del batch_size
        return []

    def posterior_on_eval(self) -> Tuple[float, float]:
        """Return (posterior, mean_ll) on a fixed evaluation subset."""
        if self.eval_idx.numel() == 0:
            return 0.0, 0.0
        with torch.no_grad():
            objective, ll = self._posterior_objective(self.eval_idx)
        return float(objective.item()), float(ll.item())

    def ll_on_eval_for(self, alpha: float, beta: float) -> float:
        """Mean action log-likelihood on eval subset for explicit parameters."""
        if self.eval_idx.numel() == 0:
            return 0.0
        with torch.no_grad():
            idx = self.eval_idx
            logits = (
                self.base_logits[idx]
                + float(alpha) * self.phi_alpha[idx]
                + float(beta) * self.phi_beta[idx]
            )
            logits = logits.masked_fill(~self.masks[idx], float("-inf"))
            ll = Categorical(logits=logits).log_prob(self.actions[idx]).mean()
        return float(ll.item())

    # ------------------------------------------------------------------
    # Convergence
    # ------------------------------------------------------------------

    def is_converged(self) -> bool:
        """
        Converged when the std of BOTH parameters over the
        last CONV_WINDOW steps is below CONV_THRESHOLD.
        This checks stability of the estimate, not proximity to truth.
        """
        if len(self.alpha_history) < CONV_MIN_STEPS + CONV_WINDOW:
            return False
        std_alpha = float(np.std(self.alpha_history[-CONV_WINDOW:]))
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
    S:               float,
) -> Dict:
    """
    Run the full GABO-IRL procedure for one target seat.
    Returns a result dict with estimates, history, and convergence info.
    """
    log.info("  Running IRL for seat %d (true α=%.5f, true β=%.4f)",
             target_seat, true_alpha, true_beta)
    log.info("    Data: %d hands  |  VAR_NORM=%.0f  |  LR=%.4f",
             len(step_data), var_norm, IRL_LR,
             )

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
        S=S,
    )

    if opt.n_states == 0:
        log.warning("    No decision states for seat %d — skipping.", target_seat)
        return {"seat": target_seat, "error": "no_states"}

    log.info(
        "    Flattened %d decision states | feature contrast α=%.5f β=%.5f",
        opt.n_states,
        opt.alpha_feature_contrast,
        opt.beta_feature_contrast,
    )

    start = time.time()
    for step_i in range(IRL_N_STEPS):
        ll = opt.step([])

        if (step_i + 1) % IRL_LOG_EVERY == 0:
            α̂  = opt.current_alpha
            β̂  = opt.current_beta
            post, _ = opt.posterior_on_eval()
            elapsed = time.time() - start
            log.info(
                "    Step %4d | α̂=%+.5f (err %+.5f) | β̂=%+.4f (err %+.4f)"
                " | LL=%.4f | post=%.4f | %.0fs",
                step_i + 1,
                α̂,  α̂  - true_alpha,
                β̂,  β̂  - true_beta,
                ll, post, elapsed,
            )

        if opt.is_converged():
            log.info("    Converged at step %d.", step_i + 1)
            break

    # Posterior mean over last CONV_WINDOW steps (more stable than point estimate)
    final_alpha = opt.mean_alpha_history(CONV_WINDOW)
    final_beta  = opt.mean_beta_history(CONV_WINDOW)
    final_post, final_ll = opt.posterior_on_eval()
    baseline_ll = opt.ll_on_eval_for(0.0, 0.0)
    ll_gain = final_ll - baseline_ll

    log.info(
        "  Seat %d final: α̂=%+.5f (true %+.5f) | β̂=%+.4f (true %+.4f)"
        " | eval-LL=%.4f | ΔLL(vs neutral)=%.6f",
        target_seat, final_alpha, true_alpha, final_beta, true_beta, final_ll, ll_gain,
    )

    identifiable = abs(ll_gain) > 5e-4
    if not identifiable:
        log.warning(
            "  Seat %d appears weakly identifiable (very small LL gain over neutral).",
            target_seat,
        )

    return {
        "seat":          target_seat,
        "true_alpha":    true_alpha,
        "true_beta":     true_beta,
        "est_alpha":     final_alpha,
        "est_beta":      final_beta,
        "alpha_mse":     (final_alpha - true_alpha) ** 2,
        "beta_mse":      (final_beta  - true_beta)  ** 2,
        "var_norm":      var_norm,
        "n_states":      opt.n_states,
        "n_steps":       len(opt.alpha_history),
        "converged":     opt.is_converged(),
        "alpha_feature_contrast": opt.alpha_feature_contrast,
        "beta_feature_contrast":  opt.beta_feature_contrast,
        "posterior_eval": final_post,
        "ll_eval":       final_ll,
        "ll_neutral":    baseline_ll,
        "ll_gain_vs_neutral": ll_gain,
        "weak_identifiability": (not identifiable),
        "alpha_history": opt.alpha_history,
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

    # ── Compute standard deviation of chip deltas (S) per seat ─────────────
    log.info("Computing chip delta standard deviations ...")
    chip_std: Dict[int, float] = {}
    for seat in range(NUM_PLAYERS):
        deltas = [r.chip_deltas[seat] for r in hand_records]
        std = float(np.std(deltas)) if len(deltas) > 1 else 1.0
        chip_std[seat] = max(std, 1.0)
        log.info("  Seat %d: S=%.2f", seat, chip_std[seat])

    # ── Build MC return data ───────────────────────────────────────────────
    log.info("Building MC return data ...")
    mc_data = compute_mc_returns_per_hand(hand_records, var_per_hand)

    # ── Phase 2: Per-seat IRL ──────────────────────────────────────────────
    seats_to_run = list(true_params.keys())
    if is_ablation:
        seats_to_run = [s for s in seats_to_run
                        if true_params[s] != (0.0, 0.0)]
        log.info("Ablation: running IRL on seats %s only.", seats_to_run)

    # ── 2a: Load neutral reference policy (Q0) for IRL ────────────────────
    target_net_sdicts: Dict[int, Dict]            = {}
    target_net_dims:   Dict[int, Tuple[int, int]] = {}

    ref_path = os.path.join(CHECKPOINT_DIR, "base_agent.pt")
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference base checkpoint not found: {ref_path}")
    ref_ckpt = torch.load(ref_path, map_location=device)
    ref_state = ref_ckpt["network_state"]
    ref_dims = (
        ref_ckpt.get("feature_dim", FEATURE_DIM),
        ref_ckpt.get("hidden_dim", HIDDEN_DIM),
    )

    for target_seat in seats_to_run:
        target_net_sdicts[target_seat] = ref_state
        target_net_dims[target_seat] = ref_dims

    log.info(
        "Using neutral reference policy for all seats: %s",
        ref_path,
    )

    # Opponent models are not used by the current posterior objective.
    mp_ctx = mp.get_context("spawn")

    # ── 2c: Run all per-seat IRL optimisers in parallel ───────────────────
    log.info(
        "Running IRL for %d seat(s) in parallel (N_IRL_WORKERS=%d) ...",
        len(seats_to_run), N_IRL_WORKERS,
    )

    irl_results = []
    if N_IRL_WORKERS <= 1:
        for target_seat in seats_to_run:
            if target_seat not in target_net_sdicts:
                continue
            inp_dim, hid_dim = target_net_dims[target_seat]
            true_alpha, true_beta = true_params[target_seat]

            log.info("\n" + "=" * 62)
            log.info("IRL — running seat %d in serial", target_seat)
            log.info("=" * 62)

            result = _run_irl_worker(
                target_seat,
                mc_data[target_seat],
                target_net_sdicts[target_seat],
                inp_dim,
                hid_dim,
                DEVICE,
                true_alpha,
                true_beta,
                var_std[target_seat],
                chip_std[target_seat],
            )
            irl_results.append(result)
            log.info(
                "  IRL done: seat %d  α̂=%+.5f  β̂=%+.4f",
                target_seat,
                result.get("est_alpha", float("nan")),
                result.get("est_beta",  float("nan")),
            )
    else:
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
                    target_net_sdicts[target_seat],
                    inp_dim,
                    hid_dim,
                    DEVICE,
                    true_alpha,
                    true_beta,
                    var_std[target_seat],
                    chip_std[target_seat],
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
