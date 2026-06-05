import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_accuracy_vs_class_count(result):
    class_counts = []
    class_accs = []

    for c in range(10):

        count = result[f"class_{c}_train_count"]
        acc = result[f"class_{c}_acc"]

        class_counts.append(count)
        class_accs.append(acc)

    plt.figure(figsize=(6, 4))

    plt.scatter(class_counts, class_accs)

    for c in range(10):
        plt.annotate(str(c), (class_counts[c], class_accs[c]))

    plt.xscale("log")

    plt.xlabel("Training samples per class (log scale)")
    plt.ylabel("Per-class accuracy")
    plt.title("Accuracy vs class frequency")

    plt.tight_layout()
    plt.show()

    
def plot_avg_accuracy_by_rank(avg_df, noise_rate=0.0, savepath="figures/tmp"):
    df = avg_df[avg_df["noise_rate"] == noise_rate].copy()

    plt.figure(figsize=(7, 4))

    for imbalance_ratio in sorted(df["imbalance_ratio"].unique()):
        sub = df[df["imbalance_ratio"] == imbalance_ratio]
        sub = sub.sort_values("class_rank")

        plt.plot(
            sub["class_rank"],
            sub["mean_class_acc"],
            marker="o",
            label=f"IF={imbalance_ratio}",
        )

        plt.fill_between(
            sub["class_rank"],
            sub["mean_class_acc"] - sub["std_class_acc"].fillna(0),
            sub["mean_class_acc"] + sub["std_class_acc"].fillna(0),
            alpha=0.15,
        )

    plt.xlabel("Class rank")
    plt.ylabel("Mean per-class accuracy")
    plt.title(f"Per-class accuracy by long-tail rank. Noise Rate: {noise_rate}")
    plt.xticks(range(10))
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()


def plot_avg_accuracy_by_count(avg_df, noise_rate=0.0, savepath="figures/tmp"):
    df = avg_df[avg_df["noise_rate"] == noise_rate].copy()

    plt.figure(figsize=(7, 4))

    for imbalance_ratio in sorted(df["imbalance_ratio"].unique()):
        sub = df[df["imbalance_ratio"] == imbalance_ratio]
        sub = sub.sort_values("mean_train_count", ascending=False)

        plt.plot(
            sub["mean_train_count"],
            sub["mean_class_acc"],
            marker="o",
            label=f"IF={imbalance_ratio}",
        )

    plt.xscale("log")
    plt.xlabel("Mean training samples per class")
    plt.ylabel("Mean per-class accuracy")
    plt.title(f"Per-class accuracy vs class frequency. Noise Rate: {noise_rate}")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()
    
    