import os
import numpy as np
import pandas as pd

# Project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(project_root, "results_advanced")
CSV_DIR = os.path.join(project_root, "results_csv")

os.makedirs(CSV_DIR, exist_ok=True)


def export_array(file_name, column_name):

    path = os.path.join(RESULTS_DIR, file_name)

    if not os.path.exists(path):
        print(f"Skipping {file_name} (not found)")
        return

    data = np.load(path, allow_pickle=True)

    # Flatten if needed
    if len(data.shape) > 1:
        data = data.flatten()

    df = pd.DataFrame({column_name: data})

    csv_path = os.path.join(CSV_DIR, file_name.replace(".npy", ".csv"))
    df.to_csv(csv_path, index=False)

    print(f"Saved {csv_path}")


def main():

    print("\nExporting results to CSV...\n")

    export_array("all_rewards.npy", "reward")
    export_array("all_autonomy.npy", "autonomy")
    export_array("all_stability.npy", "stability")

    export_array("all_skill.npy", "skill")
    export_array("all_difficulty.npy", "difficulty")
    export_array("all_frustration.npy", "frustration")
    export_array("all_performance.npy", "performance")
    export_array("all_helpdep.npy", "help_dependency")

    export_array("actions.npy", "action")

    print("\nCSV export completed.\n")
    print(f"Files saved in: {CSV_DIR}")


if __name__ == "__main__":
    main()