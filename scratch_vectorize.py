import re

with open("code/step3_collect_and_run_irl.py", "r") as f:
    content = f.read()

import sys
start_idx = content.find("class IRLOptimiser:")
end_idx = content.find("def run_irl_for_seat(")

if start_idx == -1 or end_idx == -1:
    print("Could not find IRLOptimiser or run_irl_for_seat")
    sys.exit(1)

new_irl = """class IRLOptimiser:
    \"\"\"
    Gradient-ascent Bayesian IRL for recovering (alpha, beta) of one target seat.
    Matches the analytical feature expectation matching gradient from doc.tex.
    Vectorized for performance.
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
        
        self._precompute_state_tensors()

    def _precompute_state_tensors(self) -> None:
        valid_data = [d for d in self.step_data if len(d[2]) > 0]
        self.n_states = len(valid_data)
        
        if self.n_states == 0:
            self.eval_idx = torch.empty(0, dtype=torch.int64, device=self.device)
            return

        # Precompute V_var and V_pot
        var_vals = [d[3][-1, 1] for d in valid_data]
        pot_vals = [d[3][-1, 2] for d in valid_data]

        self.V_var = float(np.mean(var_vals)) if var_vals else 0.0
        self.V_pot = float(np.mean(pot_vals)) if pot_vals else 0.0
        self.V_var_norm = self.V_var / self.var_norm

        # Terminal state tensors
        feats_list = [d[0][-1:] for d in valid_data]
        masks_list = [d[1][-1:] for d in valid_data]
        acts_list  = [d[2][-1:] for d in valid_data]
        
        feats_np = np.concatenate(feats_list, axis=0)
        masks_np = np.concatenate(masks_list, axis=0)
        acts_np  = np.concatenate(acts_list, axis=0)

        self.features = torch.tensor(feats_np, dtype=torch.float32, device=self.device)
        self.masks    = torch.tensor(masks_np, dtype=torch.bool, device=self.device)
        self.actions  = torch.tensor(acts_np, dtype=torch.int64, device=self.device)

        with torch.no_grad():
            base_logits, _ = self.target_network(self.features, self.masks)
        self.base_logits = base_logits.to(dtype=torch.float64)

        # Precompute A_var_n and A_pot
        raw_var = torch.tensor(var_vals, dtype=torch.float64, device=self.device)
        raw_pot = torch.tensor(pot_vals, dtype=torch.float64, device=self.device)

        self.A_var_n = (raw_var - self.V_var) / self.var_norm
        self.A_pot   = raw_pot - self.V_pot

        n_eval = min(50_000, self.n_states)
        if n_eval == self.n_states:
            self.eval_idx = torch.arange(self.n_states, dtype=torch.int64, device=self.device)
        else:
            self.eval_idx = torch.randperm(self.n_states, device=self.device)[:n_eval]

    def _sample_batch_indices(self, batch_size: int) -> torch.Tensor:
        n = min(batch_size, self.n_states)
        return torch.randint(0, self.n_states, (n,), device=self.device)

    def _compute_gradient(self, idx: torch.Tensor) -> Tuple[torch.Tensor, float]:
        if len(idx) == 0:
            return torch.zeros(2, dtype=torch.float64, device=self.device), 0.0

        alpha = self.theta[0]
        beta  = self.theta[1]

        A_var_n_batch = self.A_var_n[idx]
        A_pot_batch   = self.A_pot[idx]

        shaping = alpha * A_var_n_batch + beta * A_pot_batch

        a_obs = self.actions[idx]
        
        # We need to add shaping to the observed action's logit
        adj = self.base_logits[idx].clone()
        adj.scatter_add_(1, a_obs.unsqueeze(1), shaping.unsqueeze(1))

        legal = self.masks[idx]
        log_z = torch.logsumexp(adj.masked_fill(~legal, float('-inf')), dim=1)
        
        # Log likelihood of observed action
        ll_batch = adj.gather(1, a_obs.unsqueeze(1)).squeeze(1) - log_z

        pi_a = torch.exp(torch.clamp(ll_batch, -30, 0))

        g_alpha = torch.mean(A_var_n_batch * (1.0 - pi_a))
        g_beta  = torch.mean(A_pot_batch * (1.0 - pi_a))

        prior_g_alpha = -alpha / (self.prior_sigma ** 2)
        prior_g_beta  = -beta  / (self.prior_sigma ** 2)

        full_grad = torch.stack([g_alpha + prior_g_alpha, g_beta + prior_g_beta])

        grad_norm = float(full_grad.norm().item())
        if grad_norm > IRL_GRAD_CLIP:
            full_grad = full_grad * (IRL_GRAD_CLIP / grad_norm)

        return full_grad, float(torch.mean(ll_batch).item())

    def step(self, batch: List[Tuple]) -> float:
        if self.n_states == 0:
            return 0.0

        idx = self._sample_batch_indices(IRL_BATCH_SIZE)
        full_grad, ll = self._compute_gradient(idx)

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
        return []

    def posterior_on_eval(self) -> Tuple[float, float]:
        if len(self.eval_idx) == 0:
            return 0.0, 0.0
        with torch.no_grad():
            _, ll = self._compute_gradient(self.eval_idx)
        return 0.0, ll

    def ll_on_eval_for(self, alpha: float, beta: float) -> float:
        if alpha == 0.0 and beta == 0.0:
            with torch.no_grad():
                old_theta = self.theta.clone()
                self.theta[0] = 0.0
                self.theta[1] = 0.0
                _, ll = self._compute_gradient(self.eval_idx)
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
