import numpy as np
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "results_advanced"

def plot_skill_difficulty():

    all_difficulty = np.load(os.path.join(RESULTS_DIR, "all_difficulty.npy"), allow_pickle=True)
    all_skill = np.load(os.path.join(RESULTS_DIR, "all_skill.npy"), allow_pickle=True)

    # Take average over trials
    mean_difficulty = np.mean(all_difficulty, axis=0)
    mean_skill = np.mean(all_skill, axis=0)

    episodes = len(mean_difficulty)

    plt.figure(figsize=(12, 5))
    plt.plot(mean_difficulty, label="Difficulty", color="red")
    plt.plot(mean_skill, label="Skill Level", color="green")

    plt.title("Difficulty–Skill Alignment Over Time")
    plt.xlabel("Episode")
    plt.ylabel("State Level (0–3)")
    plt.legend()
    
    plt.savefig(os.path.join(RESULTS_DIR, "difficulty_skill_alignment.png"))
    plt.close()


if __name__ == "__main__":
    plot_skill_difficulty()