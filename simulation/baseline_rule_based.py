import sys
import os
import numpy as np
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from environment.learner_env import LearnerEnv


def rule_policy(state):
    performance, help_dep, difficulty, frustration = state

    if frustration >= 2:
        return 3  # Explanation
    elif performance <= 1:
        return 1  # Hint
    else:
        return 0  # Pause


def run_rule_based(episodes=5000):
    rewards = []

    for ep in range(episodes):
        env = LearnerEnv()
        total_reward = 0

        for _ in range(50):
            state = env.get_state()
            action = rule_policy(state)
            next_state, reward, success = env.step(action)
            total_reward += reward

        rewards.append(total_reward)

    plt.plot(rewards)
    plt.title("Rule-Based Baseline Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig(os.path.join(project_root, "results/rule_based_reward.png"))
    plt.show()


if __name__ == "__main__":
    run_rule_based()