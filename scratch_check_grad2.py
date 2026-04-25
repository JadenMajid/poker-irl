import torch
import numpy as np
import pickle

import sys
sys.path.append("code")

from step3_collect_and_run_irl import compute_rolling_variance_penalties, fill_var_penalties, compute_mc_returns_per_hand

with open("irl_results/hand_records.pkl", "rb") as f:
    hand_records = pickle.load(f)

var_per_hand, _ = compute_rolling_variance_penalties(hand_records, window=200)
fill_var_penalties(hand_records, var_per_hand)
mc_data = compute_mc_returns_per_hand(hand_records, var_per_hand)

seat = 2
data = mc_data[seat]
valid_data = [d for d in data if len(d[2]) > 0]

pot_vals = [d[3][-1, 2] for d in valid_data]
V_pot = float(np.mean(pot_vals))
A_pot = torch.tensor(pot_vals, dtype=torch.float64) - V_pot

feats_list = [d[0][-1:] for d in valid_data]
masks_list = [d[1][-1:] for d in valid_data]
acts_list  = [d[2][-1:] for d in valid_data]

feats_np = np.concatenate(feats_list, axis=0)
masks_np = np.concatenate(masks_list, axis=0)
acts_np  = np.concatenate(acts_list, axis=0)

features = torch.tensor(feats_np, dtype=torch.float32)
masks    = torch.tensor(masks_np, dtype=torch.bool)
actions  = torch.tensor(acts_np, dtype=torch.int64)

ckpt = torch.load("checkpoints/base_agent.pt", map_location="cpu")
from agent import ActorCriticNetwork
net = ActorCriticNetwork(165, 256)
net.load_state_dict(ckpt["network_state"])
net.eval()

with torch.no_grad():
    base_logits, _ = net(features, masks)
base_logits = base_logits.to(torch.float64)

legal = masks
log_z = torch.logsumexp(base_logits.masked_fill(~legal, float('-inf')), dim=1)
ll_batch = base_logits.gather(1, actions.unsqueeze(1)).squeeze(1) - log_z
pi_a = torch.exp(torch.clamp(ll_batch, -30, 0))

# Find folds and non-folds
is_fold = (actions == 0)
is_raise = (actions >= 2)

print(f"Total: {len(actions)}")
print(f"Folds: {is_fold.sum().item()}")
print(f"Raises: {is_raise.sum().item()}")

print(f"Folds - Mean A_pot: {A_pot[is_fold].mean().item()}")
print(f"Folds - Mean pi_a: {pi_a[is_fold].mean().item()}")
print(f"Folds - Mean A_pot * pi_a: {(A_pot * pi_a)[is_fold].mean().item()}")

print(f"Raises - Mean A_pot: {A_pot[is_raise].mean().item()}")
print(f"Raises - Mean pi_a: {pi_a[is_raise].mean().item()}")
print(f"Raises - Mean A_pot * pi_a: {(A_pot * pi_a)[is_raise].mean().item()}")

