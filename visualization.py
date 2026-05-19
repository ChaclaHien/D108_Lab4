import numpy as np
import matplotlib.pyplot as plt


def plot_f1_comparison(results: dict, save_path="f1_comparison.png"):
    """
    results = {
        "Model Name": {"f1_macro": 0.xx, "f1_weighted": 0.xx},
        ...
    }
    """
    models = list(results.keys())
    f1_macro = [results[m]["f1_macro"] for m in models]
    f1_weighted = [results[m]["f1_weighted"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, f1_macro, width, label="F1 Macro", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, f1_weighted, width, label="F1 Weighted", color="#DD8452")

    ax.set_xlabel("Model")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score Comparison – Wine Quality Dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    # Example: run after main scripts to visualize
    sample_results = {
        "DT (NumPy)": {"f1_macro": 0.72, "f1_weighted": 0.73},
        "RF (NumPy)": {"f1_macro": 0.78, "f1_weighted": 0.79},
        "DT (sklearn)": {"f1_macro": 0.74, "f1_weighted": 0.75},
        "RF (sklearn)": {"f1_macro": 0.82, "f1_weighted": 0.83},
    }
    plot_f1_comparison(sample_results)
