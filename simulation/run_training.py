import sys
import os

# ---------------------------------------------------------
# FIX: Add project root to Python path
# ---------------------------------------------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# ---------------------------------------------------------
# FIX: Auto-create the 'results' folder
# ---------------------------------------------------------
results_path = os.path.join(project_root, "results")
os.makedirs(results_path, exist_ok=True)

# ---------------------------------------------------------
# Project modules
# ---------------------------------------------------------
from environment.learner_env import LearnerEnv
from rl_agent.q_learning import QLearningAgent

import matplotlib.pyplot as plt
import numpy as np


def train(episodes=5000):
    agent = QLearningAgent()

    # Tracking containers
    rewards = []
    action_counts = np.zeros(4)
    max_q_values = []
    mean_q_values = []
    action_trend = []  # stores distribution every 100 episodes

    for ep in range(episodes):
        env = LearnerEnv()
        total_reward = 0

        # Count actions in this episode
        ep_action_count = np.zeros(4)

        for _ in range(50):
            state = env.get_state()
            action = agent.choose_action(state)
            ep_action_count[action] += 1
            action_counts[action] += 1

            next_state, reward, success = env.step(action)
            agent.update(state, action, reward, next_state)
            total_reward += reward

        rewards.append(total_reward)

        # Capture action trend every 100 episodes
        if (ep + 1) % 100 == 0:
            action_trend.append(ep_action_count / 50)  # normalize

        # Q-table convergence tracking
        max_q_values.append(np.max(agent.Q))
        mean_q_values.append(np.mean(agent.Q))

        # Training progress log
        if (ep + 1) % 500 == 0:
            print(f"Episode {ep+1}/{episodes} | Reward: {total_reward}")

    # ---------------------------------------------------------
    # 1. Reward Curve
    # ---------------------------------------------------------
    plt.figure()
    plt.plot(rewards)
    plt.title("Reward vs Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig(os.path.join(results_path, "reward_curve.png"))
    plt.close()

    # ---------------------------------------------------------
    # 2. Action Frequency Bar Chart
    # ---------------------------------------------------------
    plt.figure()
    actions = ["Pause", "Hint", "Question", "Explanation"]
    plt.bar(actions, action_counts)
    plt.title("Total Action Usage")
    plt.savefig(os.path.join(results_path, "action_frequency.png"))
    plt.close()

    # ---------------------------------------------------------
    # 3. Policy Convergence
    # ---------------------------------------------------------
    plt.figure()
    plt.plot(max_q_values, label="Max Q")
    plt.plot(mean_q_values, label="Mean Q")
    plt.title("Policy Convergence Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Q-value")
    plt.legend()
    plt.savefig(os.path.join(results_path, "policy_convergence.png"))
    plt.close()

    # ---------------------------------------------------------
    # 4. Action Trend Over Time (NEW)
    # ---------------------------------------------------------
    trend = np.array(action_trend)
    plt.figure()
    for i, label in enumerate(actions):
        plt.plot(trend[:, i], label=label)

    plt.title("Action Distribution Over Training")
    plt.xlabel("x100 Episodes")
    plt.ylabel("Proportion of Action Taken")
    plt.legend()
    plt.savefig(os.path.join(results_path, "action_trend.png"))
    plt.close()

    return agent


if __name__ == "__main__":
    trained_agent = train()