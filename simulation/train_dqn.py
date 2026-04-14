import numpy as np
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from rl_agent.dqn_agent import DQNAgent
from environment.advanced_learner_env import AdvancedLearnerEnv


EPISODES = 600
STEPS = 30


def run_training(learner_type, label):

    print(f"\nRunning DQN for {label}...")

    agent = DQNAgent()
    env = AdvancedLearnerEnv(learner_type=learner_type)

    rewards = []

    for ep in range(EPISODES):

        total_reward = 0

        for _ in range(STEPS):

            state = np.array(env.get_state())
            action = agent.choose_action(state)

            next_state, reward, success = env.step(action)
            next_state = np.array(next_state)

            agent.remember(state, action, reward, next_state)

            total_reward += reward

        # train once per episode
        agent.replay(batch_size=16)

        rewards.append(total_reward)

        if (ep+1) % 100 == 0:
            print(f"{label} → Episode {ep+1}/{EPISODES}")

    # ✅ SAVE CORRECTLY
    save_dir = os.path.join(project_root, f"results_dqn_{label}")
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "rewards_dqn.npy"), rewards)

    print(f"✅ Saved → {save_dir}")


# ------------------------------------------------
# RUN FOR ALL LEARNERS
# ------------------------------------------------
if __name__ == "__main__":

    run_training(0, "Beginner")
    run_training(1, "Intermediate")
    run_training(2, "Advanced")

    print("\nAll DQN training completed!")