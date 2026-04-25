import json
import matplotlib.pyplot as plt
import os
import sys

def create_irl_convergence_plot():
    log_path = 'irl_results/irl_convergence_log.json'
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found. Make sure step 3 has completed.")
        return

    # Try to load true params from checkpoints, otherwise use defaults from step 2
    params_path = 'checkpoints/perturbed_agent_params.json'
    true_alpha = {}
    true_beta = {}
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = json.load(f)
            for p in params:
                true_alpha[p['seat']] = p['alpha']
                true_beta[p['seat']] = p['beta']
    else:
        # Fallback to the known defaults if the file isn't ready
        true_alpha = {0: 0.005, 1: 0.005, 2: -0.005, 3: -0.005}
        true_beta = {0: 0.3, 1: -0.3, 2: 0.3, 3: -0.3}

    with open(log_path, 'r') as f:
        log_data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for entry in log_data:
        seat = entry['seat']
        color = colors[seat % len(colors)]
        
        alpha_hist = entry.get('alpha_history', [])
        beta_hist = entry.get('beta_history', [])
        steps = list(range(len(alpha_hist)))
        
        # Plot estimated values
        ax1.plot(steps, alpha_hist, color=color, label=f'Seat {seat} Est')
        ax2.plot(steps, beta_hist, color=color, label=f'Seat {seat} Est')
        
        # Plot true values
        if seat in true_alpha:
            ax1.axhline(y=true_alpha[seat], color=color, linestyle='--', alpha=0.7)
        if seat in true_beta:
            ax2.axhline(y=true_beta[seat], color=color, linestyle='--', alpha=0.7)

    ax1.set_title(r'$\hat{\alpha}$ vs Gradient Step')
    ax1.set_xlabel('Gradient Step')
    ax1.set_ylabel(r'$\alpha$ value')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.set_title(r'$\hat{\beta}$ vs Gradient Step')
    ax2.set_xlabel('Gradient Step')
    ax2.set_ylabel(r'$\beta$ value')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    os.makedirs('figs', exist_ok=True)
    out_path = 'figs/irl_convergence.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Successfully generated {out_path}")
    plt.close()

if __name__ == "__main__":
    create_irl_convergence_plot()
