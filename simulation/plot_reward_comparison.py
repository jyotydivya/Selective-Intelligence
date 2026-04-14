import numpy as np
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FOLDERS = {
    "Beginner": "learner_type_results/Beginner",
    "Intermediate": "learner_type_results/Intermediate",
    "Advanced": "learner_type_results/Advanced"
}


def smooth(data, window=30):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


plt.figure(figsize=(10,5))

for name, folder in FOLDERS.items():

    path = os.path.join(BASE_DIR, folder, "reward_curve.png")

    # 🔴 Try loading rewards.npy instead
    reward_path = os.path.join(BASE_DIR, folder, "rewards.npy")

    if os.path.exists(reward_path):
        rewards = np.load(reward_path)
        rewards = smooth(rewards)
        plt.plot(rewards, label=name)

    else:
        print(f"⚠ No rewards.npy in {folder}")


plt.title("Reward Comparison Across Learner Types")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.legend()

output = os.path.join(BASE_DIR, "reward_comparison.png")
plt.savefig(output)
plt.show()

print(f"Saved → {output}")