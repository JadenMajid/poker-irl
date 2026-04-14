#! /bin/bash
# 1. Train the base Nash-equilibrium agent
python code/step1_train_base_agent.py

# 2. Fine-tune specialized agents with different playstyles
python code/step2_train_perturbed_agents.py

# 3. Collect data and run the IRL recovery process
python code/step3_collect_and_run_irl.py

# 4. Evaluate the accuracy of the IRL results
python code/step4_evaluate_results.py