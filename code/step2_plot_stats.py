import json
import os
import matplotlib.pyplot as plt

def plot_step2_stats(log_path, output_dir):
    if not os.path.exists(log_path):
        print(f"Log file not found at {log_path}")
        return

    with open(log_path, "r") as f:
        log_data = json.load(f)

    # Separate data by seat
    seats = sorted(list(set(entry["seat"] for entry in log_data)))
    
    metrics = ["mean_kl", "policy_loss", "value_loss", "entropy", "kl_penalty"]
    
    for seat in seats:
        seat_data = [entry for entry in log_data if entry["seat"] == seat]
        hands = [entry["hand"] for entry in seat_data]
        
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)), sharex=True)
        fig.suptitle(f"Step 2: Training Stats for Agent (Seat {seat})")
        
        for i, metric in enumerate(metrics):
            values = [entry[metric] for entry in seat_data]
            axes[i].plot(hands, values, label=metric)
            axes[i].set_ylabel(metric)
            axes[i].grid(True)
            if metric == "mean_kl":
                axes[i].axhline(y=0.01, color='r', linestyle='--', label="Threshold (0.01)")
            axes[i].legend()
            
        axes[-1].set_xlabel("Hands Played")
        
        output_path = os.path.join(output_dir, f"step2_stats_seat_{seat}.png")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_path)
        print(f"Plot for seat {seat} saved to {output_path}")

if __name__ == "__main__":
    LOG_FILE = "checkpoints/perturbed_training_log.json"
    OUTPUT_DIR = "."
    plot_step2_stats(LOG_FILE, OUTPUT_DIR)
