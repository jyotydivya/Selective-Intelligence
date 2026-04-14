import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# FIX: Add project root for imports
# ---------------------------------------------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from rl_agent.q_learning_advanced import AdvancedQLearningAgent
from environment.advanced_learner_env import AdvancedLearnerEnv


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
EPISODES = 5000
TRIALS = 10
RESULTS_DIR = os.path.join(project_root, "results_advanced")

os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------
# Autonomy Score
# ---------------------------------------------------------
def compute_autonomy(successes, explanation_successes):
    if successes == 0:
        return 0
    return (successes - explanation_successes) / successes


# ---------------------------------------------------------
# RUN ONE TRAINING SESSION
# ---------------------------------------------------------
def run_single_training():

    agent = AdvancedQLearningAgent()
    env = AdvancedLearnerEnv()

    reward_history = []
    autonomy_history = []
    stability_history = []
    actions_list = []

    difficulty_list = []
    skill_list = []
    frustration_list = []
    performance_list = []
    helpdep_list = []

    total_success = 0
    success_after_expl = 0

    Q_prev = agent.Q.copy()

    for ep in range(EPISODES):

        total_reward = 0

        # Run one episode (50 learning steps)
        for _ in range(50):
            state = env.get_state()
            action = agent.choose_action(state)
            next_state, reward, success = env.step(action)

            agent.update(state, action, reward, next_state)

            total_reward += reward
            actions_list.append(action)

            if success:
                total_success += 1
                if action == 3:
                    success_after_expl += 1

        # Per-episode logging
        reward_history.append(total_reward)
        autonomy_history.append(compute_autonomy(total_success, success_after_expl))

        difficulty_list.append(env.difficulty)
        skill_list.append(env.skill_level)
        frustration_list.append(env.frustration)
        performance_list.append(env.performance)
        helpdep_list.append(env.help_dependency)

        # Q-table stability measurement
        if ep % 200 == 0 and ep > 0:
            dist = agent.policy_distance(Q_prev)
            stability_history.append(dist)
            Q_prev = agent.Q.copy()

    return (
        reward_history,
        autonomy_history,
        stability_history,
        actions_list,
        difficulty_list,
        skill_list,
        frustration_list,
        performance_list,
        helpdep_list
    )


# ---------------------------------------------------------
# MONTE CARLO TRAINING LOOP
# ---------------------------------------------------------
all_rewards = []
all_autonomy = []
all_stability = []
all_actions = []

all_difficulty = []
all_skill = []
all_frustration = []
all_performance = []
all_helpdep = []

print("\n========== Running Monte Carlo trials ==========\n")

for trial in range(TRIALS):
    print(f"Trial {trial+1}/{TRIALS}")

    (rewards, autonomy, stability, actions,
     difficulty, skill, frust, perf, helpdep) = run_single_training()

    all_rewards.append(rewards)
    all_autonomy.append(autonomy)
    all_stability.append(stability)
    all_actions.extend(actions)

    all_difficulty.append(difficulty)
    all_skill.append(skill)
    all_frustration.append(frust)
    all_performance.append(perf)
    all_helpdep.append(helpdep)


# ---------------------------------------------------------
# Convert to numpy arrays
# ---------------------------------------------------------
all_rewards = np.array(all_rewards)
all_autonomy = np.array(all_autonomy)
all_stability = np.array(all_stability)
all_actions = np.array(all_actions, dtype=int)

all_difficulty = np.array(all_difficulty)
all_skill = np.array(all_skill)
all_frustration = np.array(all_frustration)
all_performance = np.array(all_performance)
all_helpdep = np.array(all_helpdep)


# ---------------------------------------------------------
# Save raw data for all analysis modules
# ---------------------------------------------------------
np.save(os.path.join(RESULTS_DIR, "all_rewards.npy"), all_rewards)
np.save(os.path.join(RESULTS_DIR, "all_autonomy.npy"), all_autonomy)
np.save(os.path.join(RESULTS_DIR, "all_stability.npy"), all_stability)
np.save(os.path.join(RESULTS_DIR, "actions.npy"), all_actions)

np.save(os.path.join(RESULTS_DIR, "all_difficulty.npy"), all_difficulty)
np.save(os.path.join(RESULTS_DIR, "all_skill.npy"), all_skill)
np.save(os.path.join(RESULTS_DIR, "all_frustration.npy"), all_frustration)
np.save(os.path.join(RESULTS_DIR, "all_performance.npy"), all_performance)
np.save(os.path.join(RESULTS_DIR, "all_helpdep.npy"), all_helpdep)


# ---------------------------------------------------------
# Plot 1 — Reward Curve
# ---------------------------------------------------------
mean_rewards = np.mean(all_rewards, axis=0)
std_rewards = np.std(all_rewards, axis=0)

plt.figure(figsize=(10,4))
plt.plot(mean_rewards, label="Mean Reward", color="cyan")
plt.fill_between(
    range(EPISODES),
    mean_rewards - std_rewards,
    mean_rewards + std_rewards,
    alpha=0.3, color="cyan"
)
plt.title("Reward Curve (10-Trial Average)")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.savefig(os.path.join(RESULTS_DIR, "reward_curve.png"))
plt.close()


# ---------------------------------------------------------
# Plot 2 — Autonomy Curve
# ---------------------------------------------------------
mean_autonomy = np.mean(all_autonomy, axis=0)
std_autonomy = np.std(all_autonomy, axis=0)

plt.figure(figsize=(10,4))
plt.plot(mean_autonomy, color="green", label="Autonomy")
plt.fill_between(
    range(EPISODES),
    mean_autonomy - std_autonomy,
    mean_autonomy + std_autonomy,
    alpha=0.3, color="lightgreen"
)
plt.title("Autonomy Score Over Time")
plt.xlabel("Episodes")
plt.ylabel("Autonomy")
plt.savefig(os.path.join(RESULTS_DIR, "autonomy_curve.png"))
plt.close()


# ---------------------------------------------------------
# Plot 3 — Policy Stability Curve
# ---------------------------------------------------------
mean_stability = np.mean(all_stability, axis=0)

plt.figure(figsize=(8,4))
plt.plot(mean_stability, color="magenta")
plt.title("Policy Stability (Frobenius Norm ΔQ)")
plt.xlabel("Checkpoint (200-ep intervals)")
plt.ylabel("Q-table Distance")
plt.savefig(os.path.join(RESULTS_DIR, "policy_stability.png"))
plt.close()


print("\n========== Training Complete ==========")
print(f"Results saved to: {RESULTS_DIR}\n")