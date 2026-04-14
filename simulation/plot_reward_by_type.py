import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------------------------------------
# Paths
# ------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FOLDERS = {
    "Beginner": "learner_type_results/Beginner",
    "Intermediate": "learner_type_results/Intermediate",
    "Advanced": "learner_type_results/Advanced"
}


# ------------------------------------------------
# Smooth function
# ------------------------------------------------
def smooth(data, window=50):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


# ------------------------------------------------
# Plot separate reward curves
# ------------------------------------------------
for name, folder in FOLDERS.items():

    path = os.path.join(BASE_DIR, folder, "rewards.npy")

    if not os.path.exists(path):
        print(f"Missing: {path}")
        continue

    rewards = np.load(path)

    # 🔥 smooth to remove noise
    rewards_smooth = smooth(rewards)

    plt.figure(figsize=(8,5))
    plt.plot(rewards_smooth, linewidth=2)

    plt.title(f"{name} Learner — Reward Curve")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")

    save_path = os.path.join(BASE_DIR, folder, f"{name.lower()}_reward_curve.png")
    plt.savefig(save_path)

    plt.close()

    print(f"{name} reward curve saved → {save_path}")