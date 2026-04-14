import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FOLDERS = {
    "Beginner": "learner_type_results/Beginner",
    "Intermediate": "learner_type_results/Intermediate",
    "Advanced": "learner_type_results/Advanced"
}


# ----------------------------------------
# Build Heatmap
# ----------------------------------------
def build_heatmap(action_log, steps_per_episode=50, bins=20):

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

    return heatmap


# ----------------------------------------
# Plot Individual Heatmap
# ----------------------------------------
def plot_single_heatmap(name, heatmap, save_path):

    plt.figure(figsize=(8, 5))

    sns.heatmap(
        heatmap,
        cmap="viridis",
        yticklabels=["Pause", "Hint", "Socratic", "Explain"],
        linewidths=0.5,
        linecolor="gray"
    )

    plt.title(f"{name} Learner - Action Distribution")
    plt.xlabel("Training Progress")
    plt.ylabel("Action")

    plt.savefig(save_path)
    plt.close()


# ----------------------------------------
# Plot Combined Heatmap
# ----------------------------------------
def plot_combined_heatmaps(all_heatmaps):

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (name, heatmap) in zip(axes, all_heatmaps.items()):

        sns.heatmap(
            heatmap,
            cmap="viridis",
            ax=ax,
            cbar=False,
            yticklabels=["Pause", "Hint", "Socratic", "Explain"]
        )

        ax.set_title(name)
        ax.set_xlabel("Progress")

    plt.tight_layout()

    output = os.path.join(BASE_DIR, "combined_heatmap.png")
    plt.savefig(output)
    plt.close()

    print(f"Saved combined heatmap → {output}")


# ----------------------------------------
# MAIN
# ----------------------------------------
if __name__ == "__main__":

    all_heatmaps = {}

    for name, folder in FOLDERS.items():

        path = os.path.join(BASE_DIR, folder, "actions.npy")

        if not os.path.exists(path):
            print(f"Missing: {path}")
            continue

        actions = np.load(path, allow_pickle=True)

        heatmap = build_heatmap(actions)

        save_path = os.path.join(BASE_DIR, folder, f"{name.lower()}_heatmap.png")

        plot_single_heatmap(name, heatmap, save_path)

        all_heatmaps[name] = heatmap

        print(f"{name} heatmap saved.")

    # combined plot
    if len(all_heatmaps) == 3:
        plot_combined_heatmaps(all_heatmaps)