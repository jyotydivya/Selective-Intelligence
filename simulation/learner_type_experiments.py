import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Add project root path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from rl_agent.q_learning_advanced import AdvancedQLearningAgent
from environment.advanced_learner_env import AdvancedLearnerEnv


EPISODES = 4000     # Fewer episodes for per-type runs (faster)
RESULTS_DIR = os.path.join(project_root, "learner_type_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ======================================================
# REUSABLE EXPERIMENT FUNCTION
# ======================================================
def run_experiment(learner_type, label):

    print(f"\nRunning experiment for {label}...")

    agent = AdvancedQLearningAgent()
    env = AdvancedLearnerEnv(learner_type=learner_type)

    rewards = []
    autonomy = []
    difficulty = []
    skill = []
    frustration = []
    performance = []
    helpdep = []
    actions = []

    total_success = 0
    success_after_expl = 0

    for ep in range(EPISODES):

        ep_reward = 0

        for _ in range(50):
            state = env.get_state()
            action = agent.choose_action(state)
            next_state, reward, success = env.step(action)
            agent.update(state, action, reward, next_state)

            ep_reward += reward
            actions.append(action)

            if success:
                total_success += 1
                if action == 3:
                    success_after_expl += 1

        rewards.append(ep_reward)

        # Autonomy
        if total_success == 0:
            autonomy.append(0)
        else:
            autonomy.append((total_success - success_after_expl) / total_success)

        difficulty.append(env.difficulty)
        skill.append(env.skill_level)
        frustration.append(env.frustration)
        performance.append(env.performance)
        helpdep.append(env.help_dependency)

    # Convert to numpy
    rewards = np.array(rewards)
    autonomy = np.array(autonomy)
    difficulty = np.array(difficulty)
    skill = np.array(skill)
    frustration = np.array(frustration)
    performance = np.array(performance)
    helpdep = np.array(helpdep)
    actions = np.array(actions)

    # Save raw files
    save_dir = os.path.join(RESULTS_DIR, label)
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "rewards.npy"), rewards)
    np.save(os.path.join(save_dir, "autonomy.npy"), autonomy)
    np.save(os.path.join(save_dir, "difficulty.npy"), difficulty)
    np.save(os.path.join(save_dir, "skill.npy"), skill)
    np.save(os.path.join(save_dir, "frustration.npy"), frustration)
    np.save(os.path.join(save_dir, "performance.npy"), performance)
    np.save(os.path.join(save_dir, "helpdep.npy"), helpdep)
    np.save(os.path.join(save_dir, "actions.npy"), actions)

    # ======================================================
    # PLOTS FOR THIS LEARNER TYPE
    # ======================================================

    # Reward curve
    def smooth(data, window=50):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    smoothed_rewards = smooth(rewards)

    plt.figure(figsize=(8,5))
    plt.plot(smoothed_rewards, color="cyan", linewidth=2)
    plt.title(f"Reward Curve — {label}")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.savefig(os.path.join(save_dir, "reward_curve.png"))
    plt.close()

    # Autonomy curve
    plt.figure()
    plt.plot(autonomy, color="green")
    plt.title(f"Autonomy Curve — {label}")
    plt.xlabel("Episodes")
    plt.ylabel("Autonomy")
    plt.savefig(os.path.join(save_dir, "autonomy_curve.png"))
    plt.close()

    # Difficulty–Skill alignment
    plt.figure()
    plt.plot(difficulty, label="Difficulty", color="red")
    plt.plot(skill, label="Skill", color="green")
    plt.title(f"Difficulty–Skill Alignment — {label}")
    plt.legend()
    plt.xlabel("Episodes")
    plt.ylabel("Level (0–3)")
    plt.savefig(os.path.join(save_dir, "difficulty_skill.png"))
    plt.close()

    # Cognitive scatter — frustration vs performance
    plt.figure()
    plt.scatter(frustration, performance, alpha=0.3)
    plt.title(f"Frustration vs Performance — {label}")
    plt.xlabel("Frustration")
    plt.ylabel("Performance")
    plt.savefig(os.path.join(save_dir, "frustration_perf.png"))
    plt.close()

    # Action heatmap
    bins = EPISODES // 100
    heatmap = np.zeros((4, bins))
    ep = 0
    step = 0

    for act in actions:
        if step % 50 == 0 and step != 0:
            ep += 1
        if ep < EPISODES:
            heatmap[act][ep // 100] += 1
        step += 1

    plt.figure(figsize=(14, 4))
    sns.heatmap(
        heatmap,
        cmap="cool",
        yticklabels=["Pause", "Hint", "Socratic", "Explain"]
    )
    plt.title(f"Action Heatmap — {label}")
    plt.savefig(os.path.join(save_dir, "action_heatmap.png"))
    plt.close()

    print(f"Done → saved in {save_dir}")


# ======================================================
# RUN ALL THREE EXPERIMENTS
# ======================================================
if __name__ == "__main__":

    run_experiment(0, "Beginner")
    run_experiment(1, "Intermediate")
    run_experiment(2, "Advanced")

    print("\nAll learner-type experiments completed!\n")