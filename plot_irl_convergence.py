#!/usr/bin/env python3
"""
Plot IRL parameter convergence traces for doc.tex Figure 3.

Reads irl_results/irl_convergence_log.json and produces a two-panel figure:
  Left:  α̂ vs gradient step for all 4 seats
  Right: β̂ vs gradient step for all 4 seats
Each panel has dashed horizontal lines at the true parameter values.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np


def _resolve_conflicts(text: str, keep: str) -> str:
    """Resolve git conflict markers by keeping one side.

    Parameters
    ----------
    keep : "upper" keeps the section between <<<<<<< and =======,
           "lower" keeps the section between ======= and >>>>>>>.
    """
    lines = text.splitlines(keepends=True)
    out = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        if line.startswith("<<<<<<<"):
            i += 1
            upper = []
            while i < n and not lines[i].startswith("======="):
                upper.append(lines[i])
                i += 1
            i += 1  # skip =======
            lower = []
            while i < n and not lines[i].startswith(">>>>>>>"):
                lower.append(lines[i])
                i += 1
            i += 1  # skip >>>>>>>
            out.extend(upper if keep == "upper" else lower)
            continue
        out.append(line)
        i += 1

    return "".join(out)


def load_json_with_conflict_fallback(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Fast path: no conflict markers.
    if "<<<<<<<" not in raw:
        return json.loads(raw)

    # Try both conflict resolutions and use the one that parses.
    for keep in ("upper", "lower"):
        try:
            candidate = _resolve_conflicts(raw, keep)
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise ValueError(
        "Could not parse JSON because merge conflict markers are present in "
        f"{path}. Please resolve conflicts in the file."
    )

# ── Load data ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
IN_PATH = os.path.join(ROOT, "irl_results", "irl_convergence_log.json")
OUT_PNG = os.path.join(ROOT, "figs", "irl_convergence.png")
OUT_PDF = os.path.join(ROOT, "figs", "irl_convergence.pdf")

data = load_json_with_conflict_fallback(IN_PATH)

# Sort by seat
data.sort(key=lambda d: d["seat"])

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]
LABELS = {
    0: r"Seat 0 ($\alpha{=}+0.005,\ \beta{=}+0.3$)",
    1: r"Seat 1 ($\alpha{=}+0.005,\ \beta{=}-0.3$)",
    2: r"Seat 2 ($\alpha{=}-0.005,\ \beta{=}+0.3$)",
    3: r"Seat 3 ($\alpha{=}-0.005,\ \beta{=}-0.3$)",
}

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)

for i, rec in enumerate(data):
    seat = rec["seat"]
    color = COLORS[i % len(COLORS)]
    label = LABELS.get(seat, f"Seat {seat}")

    alpha_h = rec["alpha_history"]
    beta_h  = rec["beta_history"]
    steps   = np.arange(1, len(alpha_h) + 1)

    # α panel — apply moving average for readability
    window = min(50, len(alpha_h) // 10 + 1)
    if window > 1:
        kernel = np.ones(window) / window
        alpha_smooth = np.convolve(alpha_h, kernel, mode="valid")
        beta_smooth  = np.convolve(beta_h,  kernel, mode="valid")
        steps_smooth = steps[window - 1:]
    else:
        alpha_smooth = alpha_h
        beta_smooth  = beta_h
        steps_smooth = steps

    # Raw values (faint)
    ax_a.plot(steps, alpha_h, color=color, alpha=0.15, linewidth=0.5)
    ax_b.plot(steps, beta_h,  color=color, alpha=0.15, linewidth=0.5)

    # Smoothed (bold)
    ax_a.plot(steps_smooth, alpha_smooth, color=color, linewidth=1.8, label=label)
    ax_b.plot(steps_smooth, beta_smooth,  color=color, linewidth=1.8, label=label)

    # True values (dashed horizontal)
    ax_a.axhline(rec["true_alpha"], color=color, linestyle="--", linewidth=1.0, alpha=0.7)
    ax_b.axhline(rec["true_beta"],  color=color, linestyle="--", linewidth=1.0, alpha=0.7)

# ── Formatting ─────────────────────────────────────────────────────────────
ax_a.set_title(r"$\hat{\alpha}$ convergence")
ax_a.set_xlabel("Gradient step")
ax_a.set_ylabel(r"$\hat{\alpha}$")
ax_a.axhline(0, color="grey", linewidth=0.5, alpha=0.5)
ax_a.legend(loc="best", framealpha=0.9)

ax_b.set_title(r"$\hat{\beta}$ convergence")
ax_b.set_xlabel("Gradient step")
ax_b.set_ylabel(r"$\hat{\beta}$")
ax_b.axhline(0, color="grey", linewidth=0.5, alpha=0.5)
ax_b.legend(loc="best", framealpha=0.9)

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
print(f"Saved -> {OUT_PNG} and {OUT_PDF}")
plt.close(fig)
