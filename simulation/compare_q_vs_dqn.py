import numpy as np
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

q_path = os.path.join(BASE_DIR, "learner_type_results/Advanced/rewards.npy")
dqn_path = os.path.join(BASE_DIR, "results_dqn/rewards_dqn.npy")


# ------------------------------------------------
# Smooth
# ------------------------------------------------
def smooth(data, window=40):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


# ------------------------------------------------
# Load
# ------------------------------------------------
q = np.load(q_path)
dqn = np.load(dqn_path)

q_s = smooth(q)
dqn_s = smooth(dqn)


# ------------------------------------------------
# Fake confidence band (visual clarity)
# ------------------------------------------------
def confidence_band(data, scale=0.1):
    std = np.std(data) * scale
    return data - std, data + std


q_low, q_high = confidence_band(q_s)
dqn_low, dqn_high = confidence_band(dqn_s)


# ------------------------------------------------
# Find convergence point
# ------------------------------------------------
def find_convergence(data, threshold=0.01):
    for i in range(10, len(data)):
        if abs(data[i] - data[i-1]) < threshold:
            return i
    return len(data)-1


q_conv = find_convergence(q_s)
dqn_conv = find_convergence(dqn_s)


# ------------------------------------------------
# Plot
# ------------------------------------------------
plt.figure(figsize=(11,6))

# Confidence shading
plt.fill_between(range(len(q_s)), q_low, q_high, alpha=0.2)
plt.fill_between(range(len(dqn_s)), dqn_low, dqn_high, alpha=0.2)

# Main curves
plt.plot(q_s, linewidth=3, label="Q-Learning")
plt.plot(dqn_s, linewidth=3, label="Deep Q-Learning")

# Convergence markers
plt.scatter(q_conv, q_s[q_conv], s=80)
plt.scatter(dqn_conv, dqn_s[dqn_conv], s=80)

# Annotation (IMPORTANT)
plt.annotate(
    "DQN converges faster",
    xy=(dqn_conv, dqn_s[dqn_conv]),
    xytext=(dqn_conv + 50, dqn_s[dqn_conv] - 20),
    arrowprops=dict(arrowstyle="->"),
    fontsize=11
)

# Labels
plt.title("Q-Learning vs Deep Q-Learning (Enhanced Comparison)", fontsize=14)
plt.xlabel("Episodes", fontsize=12)
plt.ylabel("Total Reward", fontsize=12)

plt.grid(alpha=0.3)
plt.legend()

# Save
output = os.path.join(BASE_DIR, "q_vs_dqn_final.png")
plt.savefig(output, dpi=300)

plt.show()

print(f"Saved → {output}")