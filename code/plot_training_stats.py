import json
import os
import matplotlib.pyplot as plt

def plot_training_stats(log_path, output_path):
    if not os.path.exists(log_path):
        print(f"Log file not found at {log_path}")
        return

    with open(log_path, "r") as f:
        log_data = json.load(f)

    hands = [entry["hand"] for entry in log_data]
    
    stats = {
        "mean_kl": {
            "label": "Mean KL",
            "color": "blue",
            "threshold": 0.01
        },
        "policy_loss": {
            "label": "Policy Loss",
            "color": "green"
        },
        "value_loss": {
            "label": "Value Loss",
            "color": "orange"
        },
        "entropy": {
            "label": "Entropy",
            "color": "purple"
        }
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, (key, info) in enumerate(stats.items()):
        values = [entry[key] for entry in log_data]
        axes[i].plot(hands, values, label=info["label"], color=info["color"])
        axes[i].set_xlabel("Hands Played")
        axes[i].set_ylabel(info["label"])
        axes[i].set_title(f"{info['label']} over Time")
        axes[i].grid(True)
        
        if "threshold" in info:
            axes[i].axhline(y=info["threshold"], color='r', linestyle='--', label=f"Threshold ({info['threshold']})")
            axes[i].legend()

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    LOG_FILE = "checkpoints/base_training_log.json"
    OUTPUT_FILE = "training_stats_plot.png"
    plot_training_stats(LOG_FILE, OUTPUT_FILE)
