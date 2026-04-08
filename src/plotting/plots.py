import os
import matplotlib.pyplot as plt


def plot_fairness_bars(summary_df, out_dir):
    metrics = ["dp_diff_mean", "eo_diff_mean", "eop_diff_mean"]

    summary_df.set_index("model")[metrics].plot(kind="bar")
    plt.ylabel("Fairness metric value")
    plt.title("Fairness metrics by model")
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "fairness_bars.png"))
    plt.close()


def plot_fairness_vs_auc(summary_df, out_dir):
    x = summary_df["dp_diff_mean"]
    y = summary_df["auc_mean"]

    plt.scatter(x, y)

    for i, model in enumerate(summary_df["model"]):
        plt.annotate(model, (x[i], y[i]))

    plt.xlabel("Demographic parity difference")
    plt.ylabel("AUC")
    plt.title("Fairness vs Accuracy trade-off")
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "fairness_vs_auc.png"))
    plt.close()

