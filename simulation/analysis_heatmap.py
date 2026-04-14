import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

RESULTS_DIR = "results_advanced"

def load_actions(file):
    return np.load(file, allow_pickle=True)


def plot_action_heatmap(action_log, steps_per_episode=50, bins=20):

    total_steps = len(action_log)
    total_episodes = total_steps // steps_per_episode

    heatmap = np.zeros((4, bins))

    for i, action in enumerate(action_log):

        episode = i // steps_per_episode
        bin_index = int((episode / total_episodes) * bins)

        if bin_index >= bins:
            bin_index = bins - 1

        heatmap[action][bin_index] += 1

    # normalize
    heatmap = heatmap / np.maximum(heatmap.sum(axis=1, keepdims=True), 1)

    plt.figure(figsize=(8, 5))

    sns.heatmap(
        heatmap,
        cmap="viridis",
        yticklabels=["Pause", "Hint", "Socratic", "Explain"],
        linewidths=0.5,   # 👈 adds grid lines
        linecolor='gray'
    )

    plt.title("Action Distribution Across Training", fontsize=14)
    plt.xlabel("Training Progress", fontsize=12)
    plt.ylabel("Action Type", fontsize=12)

    plt.savefig(os.path.join(RESULTS_DIR, "action_heatmap.png"))
    plt.close()
    total_steps = len(action_log)
    total_episodes = total_steps // steps_per_episode

    heatmap = np.zeros((4, bins))

    for i, action in enumerate(action_log):

        # compute episode number
        episode = i // steps_per_episode

        # map episode → bin
        bin_index = int((episode / total_episodes) * bins)

        if bin_index >= bins:
            bin_index = bins - 1

        heatmap[action][bin_index] += 1

    # normalize (VERY IMPORTANT for better visualization)
    heatmap = heatmap / np.maximum(heatmap.sum(axis=1, keepdims=True), 1)

    plt.figure(figsize=(14, 4))

    sns.heatmap(
        heatmap,
        cmap="viridis",   # better than "cool"
        xticklabels=[f"{int(i*(total_episodes/bins))}" for i in range(bins)],
        yticklabels=["Pause", "Hint", "Socratic", "Explain"]
    )

    plt.title("Action Distribution Across Training")
    plt.xlabel("Training Progress (Episodes)")
    plt.ylabel("Action Type")

    plt.savefig(os.path.join(RESULTS_DIR, "action_heatmap.png"))
    plt.close()

if __name__ == "__main__":
    action_log = load_actions(os.path.join(RESULTS_DIR, "actions.npy"))
    plot_action_heatmap(action_log)