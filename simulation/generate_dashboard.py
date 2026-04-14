import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(project_root, "results_advanced")

# load data
rewards = np.load(os.path.join(RESULTS_DIR, "all_rewards.npy"), allow_pickle=True)
autonomy = np.load(os.path.join(RESULTS_DIR, "all_autonomy.npy"), allow_pickle=True)
stability = np.load(os.path.join(RESULTS_DIR, "all_stability.npy"), allow_pickle=True)

skill = np.load(os.path.join(RESULTS_DIR, "all_skill.npy"), allow_pickle=True)
difficulty = np.load(os.path.join(RESULTS_DIR, "all_difficulty.npy"), allow_pickle=True)
frustration = np.load(os.path.join(RESULTS_DIR, "all_frustration.npy"), allow_pickle=True)
performance = np.load(os.path.join(RESULTS_DIR, "all_performance.npy"), allow_pickle=True)

actions = np.load(os.path.join(RESULTS_DIR, "actions.npy"), allow_pickle=True)

# average across trials
reward_mean = np.mean(rewards, axis=0)
autonomy_mean = np.mean(autonomy, axis=0)
stability_mean = np.mean(stability, axis=0)

skill_mean = np.mean(skill, axis=0)
difficulty_mean = np.mean(difficulty, axis=0)

frustration_mean = np.mean(frustration, axis=0)
performance_mean = np.mean(performance, axis=0)

# dashboard layout
fig, axs = plt.subplots(3, 2, figsize=(16, 14))

# ---------------------------------------------------
# 1 Reward curve
axs[0,0].plot(reward_mean, color="cyan")
axs[0,0].set_title("Reward Curve")
axs[0,0].set_xlabel("Episode")
axs[0,0].set_ylabel("Reward")

# ---------------------------------------------------
# 2 Autonomy curve
axs[0,1].plot(autonomy_mean, color="green")
axs[0,1].set_title("Autonomy Growth")
axs[0,1].set_xlabel("Episode")
axs[0,1].set_ylabel("Autonomy")

# ---------------------------------------------------
# 3 Policy stability
axs[1,0].plot(stability_mean, color="magenta")
axs[1,0].set_title("Policy Stability (Q-table convergence)")
axs[1,0].set_xlabel("Checkpoint")
axs[1,0].set_ylabel("Distance")

# ---------------------------------------------------
# 4 Difficulty vs Skill
axs[1,1].plot(difficulty_mean, label="Difficulty", color="red")
axs[1,1].plot(skill_mean, label="Skill", color="green")
axs[1,1].legend()
axs[1,1].set_title("Difficulty-Skill Alignment")

# ---------------------------------------------------
# 5 Frustration vs Performance
axs[2,0].scatter(frustration_mean, performance_mean, alpha=0.3, color="purple")
axs[2,0].set_title("Frustration vs Performance")
axs[2,0].set_xlabel("Frustration")
axs[2,0].set_ylabel("Performance")

# ---------------------------------------------------
# 6 Action heatmap
bins = 50
heatmap = np.zeros((4, bins))

for i, a in enumerate(actions):
    heatmap[a][i % bins] += 1

sns.heatmap(
    heatmap,
    ax=axs[2,1],
    cmap="cool",
    yticklabels=["Pause", "Hint", "Socratic", "Explain"]
)

axs[2,1].set_title("Action Distribution")

plt.suptitle("Selective Intelligence RL Tutor — Experiment Dashboard", fontsize=18)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "research_dashboard.png"))
plt.show()

print("\nDashboard saved to results_advanced/research_dashboard.png")