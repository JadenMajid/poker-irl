import re

with open("code/step3_collect_and_run_irl.py", "r") as f:
    content = f.read()

# 1. Replace the target network loading
old_loading = """    # ── 2a: Load neutral reference policy (Q0) for IRL ────────────────────
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
    )"""

new_loading = """    # ── 2a: Load target agent policies (Q0) for IRL ────────────────────
    target_net_sdicts: Dict[int, Dict]            = {}
    target_net_dims:   Dict[int, Tuple[int, int]] = {}

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
        target_net_sdicts[target_seat] = ckpt["network_state"]
        target_net_dims[target_seat] = (
            ckpt.get("feature_dim", FEATURE_DIM),
            ckpt.get("hidden_dim",  HIDDEN_DIM),
        )

    log.info("Using target agent policies for IRL base logits.")"""

content = content.replace(old_loading, new_loading)

# 2. Replace IRLOptimiser class
import sys
# Find start of IRLOptimiser
start_idx = content.find("class IRLOptimiser:")
# Find start of run_irl_for_seat
end_idx = content.find("def run_irl_for_seat(")

if start_idx == -1 or end_idx == -1:
    print("Could not find IRLOptimiser or run_irl_for_seat")
    sys.exit(1)

new_irl = """class IRLOptimiser:
    \"\"\"
    Gradient-ascent Bayesian IRL for recovering (alpha, beta) of one target seat.
    Matches the analytical feature expectation matching gradient from doc.tex.
    \"\"\"

    def __init__(
        self,
        target_seat:      int,
        step_data:        List[Tuple],
        opponent_models:  Dict[int, BehaviourCloningNet],
        target_network:   ActorCriticNetwork,
        device:           torch.device,
        var_norm:         float,
        S:                float,
        prior_sigma:      float = 10.0,
        lr:               float = IRL_LR,
        grad_accum_steps: int   = IRL_GRAD_ACCUM_STEPS,
    ) -> None:
        self.seat           = target_seat
        self.step_data      = step_data
        self.opp_models     = opponent_models
        self.target_network = target_network
        self.device         = device
        self.var_norm       = max(var_norm, 1.0)
        self.S              = max(S, 1.0)
        self.prior_sigma    = prior_sigma
        self.grad_accum_steps = max(1, grad_accum_steps)

        self.theta = nn.Parameter(
            torch.zeros(2, dtype=torch.float64, device=device)
        )
        self.optimiser = Adam([self.theta], lr=lr)

        self.alpha_history: List[float] = []
        self.beta_history:  List[float] = []
        self.ll_history:    List[float] = []

        self._accum_grad: Optional[torch.Tensor] = None
        self._accum_ll:   float = 0.0
        self._accum_count: int  = 0
        
        self.alpha_feature_contrast = 0.0
        self.beta_feature_contrast  = 0.0
        
        self.n_states = len([d for d in self.step_data if len(d[2]) > 0])
        n_eval = min(50_000, self.n_states)
        if n_eval == self.n_states and self.n_states > 0:
            self.eval_idx = np.arange(self.n_states)
        elif self.n_states > 0:
            self.eval_idx = np.random.choice(self.n_states, n_eval, replace=False)
        else:
            self.eval_idx = np.array([])

        self._precompute_baselines()

    def _precompute_baselines(self) -> None:
        var_vals = [d[3][-1, 1] for d in self.step_data if len(d[2]) > 0]
        pot_vals = [d[3][-1, 2] for d in self.step_data if len(d[2]) > 0]

        self.V_var = float(np.mean(var_vals)) if var_vals else 0.0
        self.V_pot = float(np.mean(pot_vals)) if pot_vals else 0.0
        self.V_var_norm = self.V_var / self.var_norm

    def _compute_gradient(self, batch: List[Tuple]) -> Tuple[torch.Tensor, float]:
        alpha = self.theta[0]
        beta  = self.theta[1]

        total_ll   = 0.0
        g_alpha    = 0.0
        g_beta     = 0.0
        n          = 0

        for feats, masks, acts, returns in batch:
            if len(acts) == 0:
                continue
            
            feat_t = torch.tensor(feats[-1:], dtype=torch.float32, device=self.device)
            mask_t = torch.tensor(masks[-1:], dtype=torch.bool,    device=self.device)
            a_obs  = int(acts[-1])

            with torch.no_grad():
                base_logits, _ = self.target_network(feat_t, mask_t)

            raw_var  = float(returns[-1, 1])
            raw_pot  = float(returns[-1, 2])
            A_var_n  = (raw_var - self.V_var) / self.var_norm
            A_pot    = raw_pot  - self.V_pot

            shaping = alpha.item() * A_var_n + beta.item() * A_pot

            adj     = base_logits[0].clone().double()
            adj[a_obs] = adj[a_obs] + shaping

            legal = mask_t[0]
            log_z = torch.logsumexp(adj[legal], dim=0)
            ll    = (adj[a_obs] - log_z).item()

            pi_a = float(np.exp(np.clip(ll, -30, 0)))

            g_alpha += A_var_n * (1.0 - pi_a)
            g_beta  += A_pot   * (1.0 - pi_a)
            total_ll += ll
            n        += 1

        if n == 0:
            return torch.zeros(2, dtype=torch.float64, device=self.device), 0.0

        g_alpha /= n
        g_beta  /= n
        ll_mean  = total_ll / n

        prior_g_alpha = -float(alpha.item()) / (self.prior_sigma ** 2)
        prior_g_beta  = -float(beta.item())  / (self.prior_sigma ** 2)

        full_grad = torch.tensor(
            [g_alpha + prior_g_alpha, g_beta + prior_g_beta],
            dtype=torch.float64,
            device=self.device,
        )

        grad_norm = float(full_grad.norm().item())
        if grad_norm > IRL_GRAD_CLIP:
            full_grad = full_grad * (IRL_GRAD_CLIP / grad_norm)

        return full_grad, ll_mean

    def step(self, batch: List[Tuple]) -> float:
        if not batch:
            batch = self._sample_batch(IRL_BATCH_SIZE)
        full_grad, ll = self._compute_gradient(batch)

        if self._accum_grad is None:
            self._accum_grad = full_grad.clone()
        else:
            self._accum_grad += full_grad
        self._accum_ll    += ll
        self._accum_count += 1

        if self._accum_count >= self.grad_accum_steps:
            avg_grad = self._accum_grad / self._accum_count
            avg_ll   = self._accum_ll   / self._accum_count

            self.optimiser.zero_grad()
            self.theta.grad = (-avg_grad).to(dtype=self.theta.dtype)
            self.optimiser.step()

            self.alpha_history.append(self.current_alpha)
            self.beta_history.append(self.current_beta)
            self.ll_history.append(avg_ll)

            self._accum_grad  = None
            self._accum_ll    = 0.0
            self._accum_count = 0

        return ll

    @property
    def current_alpha(self) -> float:
        return float(-self.theta[0].item() * self.S / self.var_norm)

    @property
    def current_beta(self) -> float:
        return float(self.theta[1].item() * self.S)

    def mean_alpha_history(self, last_n: int) -> float:
        h = self.alpha_history[-last_n:]
        return float(np.mean(h)) if h else 0.0

    def mean_beta_history(self, last_n: int) -> float:
        h = self.beta_history[-last_n:]
        return float(np.mean(h)) if h else 0.0

    def _sample_batch(self, batch_size: int) -> List[Tuple]:
        valid_data = [d for d in self.step_data if len(d[2]) > 0]
        n = len(valid_data)
        if n == 0: return []
        idx = np.random.choice(n, min(batch_size, n), replace=False)
        return [valid_data[i] for i in idx]

    def posterior_on_eval(self) -> Tuple[float, float]:
        if len(self.eval_idx) == 0:
            return 0.0, 0.0
        valid_data = [d for d in self.step_data if len(d[2]) > 0]
        batch = [valid_data[i] for i in self.eval_idx]
        with torch.no_grad():
            _, ll = self._compute_gradient(batch)
        return 0.0, ll

    def ll_on_eval_for(self, alpha: float, beta: float) -> float:
        if alpha == 0.0 and beta == 0.0:
            with torch.no_grad():
                old_theta = self.theta.clone()
                self.theta[0] = 0.0
                self.theta[1] = 0.0
                valid_data = [d for d in self.step_data if len(d[2]) > 0]
                batch = [valid_data[i] for i in self.eval_idx]
                _, ll = self._compute_gradient(batch)
                self.theta[0] = old_theta[0]
                self.theta[1] = old_theta[1]
            return ll
        return 0.0

    def is_converged(self) -> bool:
        if len(self.alpha_history) < CONV_MIN_STEPS + CONV_WINDOW:
            return False
        std_alpha = float(np.std(self.alpha_history[-CONV_WINDOW:]))
        std_beta  = float(np.std(self.beta_history[-CONV_WINDOW:]))
        return std_alpha < CONV_THRESHOLD and std_beta < CONV_THRESHOLD


# ---------------------------------------------------------------------------
# """
content = content[:start_idx] + new_irl + content[end_idx:]

with open("code/step3_collect_and_run_irl.py", "w") as f:
    f.write(content)
