import re

with open("code/step3_collect_and_run_irl.py", "r") as f:
    content = f.read()

new_loading = """    # ── 2a: Load neutral reference policy (Q0) for IRL ────────────────────
    target_net_sdicts: Dict[int, Dict]            = {}
    target_net_dims:   Dict[int, Tuple[int, int]] = {}

    ref_path = os.path.join(CHECKPOINT_DIR, "base_agent.pt")
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference base checkpoint not found: {ref_path}")
    ref_ckpt = torch.load(ref_path, map_location=device)
    ref_state = ref_ckpt["network_state"]
    ref_dims = (
        ref_ckpt.get("feature_dim", FEATURE_DIM),
        ref_ckpt.get("hidden_dim", 256),
    )

    for target_seat in seats_to_run:
        target_net_sdicts[target_seat] = ref_state
        target_net_dims[target_seat] = ref_dims

    log.info(
        "Using neutral reference policy for all seats: %s",
        ref_path,
    )"""

old_loading = """    # ── 2a: Load target agent policies (Q0) for IRL ────────────────────
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

content = content.replace(new_loading, old_loading)

with open("code/step3_collect_and_run_irl.py", "w") as f:
    f.write(content)
