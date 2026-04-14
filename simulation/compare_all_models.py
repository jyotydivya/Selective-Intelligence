import numpy as np
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LEARNERS = ["Beginner", "Intermediate", "Advanced"]


# ------------------------------------------------
# Smooth
# ------------------------------------------------
def smooth(data, window=40):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


# ------------------------------------------------
# Plot Q vs DQN per learner
# ------------------------------------------------
for learner in LEARNERS:

    q_path = os.path.join(BASE_DIR, f"learner_type_results/{learner}/rewards.npy")
    dqn_path = os.path.join(BASE_DIR, f"results_dqn_{learner}/rewards_dqn.npy")

    if not os.path.exists(q_path) or not os.path.exists(dqn_path):
        print(f"Missing data for {learner}")
        continue

    q = smooth(np.load(q_path))
    dqn = smooth(np.load(dqn_path))

    plt.figure(figsize=(9,5))

    plt.plot(q, linewidth=3, label="Q-Learning")
    plt.plot(dqn, linewidth=3, label="DQN")

    plt.title(f"{learner} — Q vs DQN", fontsize=13)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")

    plt.grid(alpha=0.3)
    plt.legend()

    save_path = os.path.join(BASE_DIR, f"{learner}_q_vs_dqn.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved → {save_path}")


# ------------------------------------------------
# Plot ALL learners (Q-learning only)
# ------------------------------------------------
plt.figure(figsize=(10,6))

for learner in LEARNERS:

    q_path = os.path.join(BASE_DIR, f"learner_type_results/{learner}/rewards.npy")

    if not os.path.exists(q_path):
        continue

    q = smooth(np.load(q_path))
    plt.plot(q, linewidth=3, label=f"{learner} (Q)")

plt.title("Q-Learning Across Learner Types")
plt.xlabel("Episodes")
plt.ylabel("Reward")

plt.legend()
plt.grid(alpha=0.3)

plt.savefig(os.path.join(BASE_DIR, "q_learning_all.png"), dpi=300)
plt.close()


# ------------------------------------------------
# Plot ALL learners (DQN)
# ------------------------------------------------
plt.figure(figsize=(10,6))

for learner in LEARNERS:

    dqn_path = os.path.join(BASE_DIR, f"results_dqn_{learner}/rewards_dqn.npy")

    if not os.path.exists(dqn_path):
        continue

    dqn = smooth(np.load(dqn_path))
    plt.plot(dqn, linewidth=3, label=f"{learner} (DQN)")

plt.title("Deep Q-Learning Across Learner Types")
plt.xlabel("Episodes")
plt.ylabel("Reward")

plt.legend()
plt.grid(alpha=0.3)

plt.savefig(os.path.join(BASE_DIR, "dqn_all.png"), dpi=300)
plt.close()