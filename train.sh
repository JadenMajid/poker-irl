#! /bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

echo "[train] Workspace: $ROOT_DIR"

# 1. Train the base Nash-equilibrium agent
if [[ -f checkpoints/base_agent.pt ]]; then
	echo "[train] SKIP step1: checkpoints/base_agent.pt already exists"
else
	echo "[train] RUN  step1: training base agent"
	python code/step1_train_base_agent.py
fi

# 2. Fine-tune specialized agents with different playstyles
if [[ -f checkpoints/perturbed_agent_0.pt && -f checkpoints/perturbed_agent_1.pt && -f checkpoints/perturbed_agent_2.pt && -f checkpoints/perturbed_agent_3.pt ]]; then
	echo "[train] SKIP step2: final perturbed agent checkpoints already exist"
else
	echo "[train] RUN  step2: fine-tuning perturbed agents"
	python code/step2_train_perturbed_agents.py
fi

# 3. Collect data and run the IRL recovery process
if [[ -f irl_results/irl_estimates.json && -f irl_results/irl_convergence_log.json ]]; then
	echo "[train] SKIP step3: irl_results/irl_estimates.json and irl_results/irl_convergence_log.json already exist"
else
	echo "[train] RUN  step3: collect trajectories and run IRL"
	python code/step3_collect_and_run_irl.py
fi

# 4. Evaluate the accuracy of the IRL results
if [[ -f irl_results/evaluation_metrics.json && -f irl_results/evaluation_details.json ]]; then
	echo "[train] SKIP step4: evaluation outputs already exist"
else
	echo "[train] RUN  step4: evaluate IRL results"
	python code/step4_evaluate_results.py
fi

echo "[train] Pipeline complete"