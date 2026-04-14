import os
import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(project_root, "results_advanced")
OUTPUT_FILE = os.path.join(project_root, "combined_results.csv")


def load_array(name):
    return np.load(os.path.join(RESULTS_DIR, name), allow_pickle=True)


def main():

    rewards = load_array("all_rewards.npy").flatten()
    autonomy = load_array("all_autonomy.npy").flatten()
    skill = load_array("all_skill.npy").flatten()
    difficulty = load_array("all_difficulty.npy").flatten()
    frustration = load_array("all_frustration.npy").flatten()
    performance = load_array("all_performance.npy").flatten()
    helpdep = load_array("all_helpdep.npy").flatten()
    actions = load_array("actions.npy").flatten()

    length = min(
        len(rewards),
        len(autonomy),
        len(skill),
        len(difficulty),
        len(frustration),
        len(performance),
        len(helpdep),
    )

    df = pd.DataFrame({
        "episode": np.arange(length),
        "reward": rewards[:length],
        "autonomy": autonomy[:length],
        "skill": skill[:length],
        "difficulty": difficulty[:length],
        "frustration": frustration[:length],
        "performance": performance[:length],
        "help_dependency": helpdep[:length],
    })

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nCombined CSV created:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()