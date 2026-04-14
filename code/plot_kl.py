import json
import os
import matplotlib.pyplot as plt

def plot_mean_kl(log_path, output_path):
    if not os.path.exists(log_path):
        print(f"Log file not found at {log_path}")
        return

    with open(log_path, "r") as f:
        log_data = json.load(f)

    hands = [entry["hand"] for entry in log_data]
    mean_kls = [entry["mean_kl"] for entry in log_data]

    plt.figure(figsize=(10, 6))
    plt.plot(hands, mean_kls, label="Mean KL Divergence")
    plt.axhline(y=0.01, color='r', linestyle='--', label="Convergence Threshold (0.01)")
    plt.xlabel("Hands Played")
    plt.ylabel("Mean KL")
    plt.title("Mean KL Divergence over Training Time (Step 1)")
    plt.legend()
    plt.grid(True)
    
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    LOG_FILE = "checkpoints/base_training_log.json"
    OUTPUT_FILE = "kl_plot.png"
    plot_mean_kl(LOG_FILE, OUTPUT_FILE)
