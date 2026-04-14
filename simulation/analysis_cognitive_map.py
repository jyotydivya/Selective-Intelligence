import numpy as np
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "results_advanced"

def plot_cognitive_maps():

    # Load averaged data
    skill = np.mean(np.load(os.path.join(RESULTS_DIR, "all_skill.npy"), allow_pickle=True), axis=0)
    difficulty = np.mean(np.load(os.path.join(RESULTS_DIR, "all_difficulty.npy"), allow_pickle=True), axis=0)
    frustration = np.mean(np.load(os.path.join(RESULTS_DIR, "all_frustration.npy"), allow_pickle=True), axis=0)
    performance = np.mean(np.load(os.path.join(RESULTS_DIR, "all_performance.npy"), allow_pickle=True), axis=0)
    helpdep = np.mean(np.load(os.path.join(RESULTS_DIR, "all_helpdep.npy"), allow_pickle=True), axis=0)

    # Episodes
    episodes = np.arange(len(skill))

    # --------------------------------------------------
    # 1. Skill Trajectory
    # --------------------------------------------------
    plt.figure(figsize=(12,5))
    plt.plot(episodes, skill, label="Skill Level", color="green")
    plt.plot(episodes, difficulty, label="Difficulty", color="red")
    plt.xlabel("Episodes")
    plt.ylabel("State Level (0-3)")
    plt.title("Skill & Difficulty Evolution")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "cognitive_skill_map.png"))
    plt.close()


    # --------------------------------------------------
    # 2. Frustration → Performance scatter
    # --------------------------------------------------
    plt.figure(figsize=(7,6))
    plt.scatter(frustration, performance, alpha=0.3, color="purple")
    plt.xlabel("Frustration Level (0-2)")
    plt.ylabel("Performance (0-2)")
    plt.title("Frustration vs. Performance Trajectory")
    plt.savefig(os.path.join(RESULTS_DIR, "cognitive_frustration_perf.png"))
    plt.close()


    # --------------------------------------------------
    # 3. Help-Dependency → Performance scatter
    # --------------------------------------------------
    plt.figure(figsize=(7,6))
    plt.scatter(helpdep, performance, alpha=0.3, color="blue")
    plt.xlabel("Help Dependency (0-2)")
    plt.ylabel("Performance (0-2)")
    plt.title("Help Dependency vs. Performance")
    plt.savefig(os.path.join(RESULTS_DIR, "cognitive_helpdep_perf.png"))
    plt.close()


if __name__ == "__main__":
    plot_cognitive_maps()