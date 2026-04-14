import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Save location
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_path = os.path.join(project_root, "results_advanced", "system_architecture_clean.png")

fig, ax = plt.subplots(figsize=(12,7))

def draw_box(text, x, y, w=2.8, h=1):
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02",
        linewidth=2,
        edgecolor="black",
        facecolor="#E8F0FE"
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text,
            ha='center', va='center',
            fontsize=11, weight="bold")

def draw_arrow(x1,y1,x2,y2):
    ax.annotate("",
        xy=(x2,y2),
        xytext=(x1,y1),
        arrowprops=dict(arrowstyle="->", lw=2)
    )

# Boxes
draw_box("Learner Environment\n(Cognitive Simulation)", 0,4)
draw_box("State Representation\n(Skill, Frustration, Performance,\nHelp Dependency, Difficulty)", 4,4)
draw_box("RL Agent\n(Q-Learning)", 8,4)

draw_box("Action Selection\nPause | Hint | Socratic | Explain", 4,2)
draw_box("Environment Update", 0,2)
draw_box("Reward + Next State", 8,2)

# Arrows
draw_arrow(2.8,4.5,4,4.5)
draw_arrow(6.8,4.5,8,4.5)

draw_arrow(9.4,4,9.4,3)
draw_arrow(8,2.5,6.8,2.5)
draw_arrow(4,2.5,2.8,2.5)
draw_arrow(1.4,3,1.4,4)

# Labels
ax.text(9.6,3.5,"Action", fontsize=10)
ax.text(3.2,2.7,"Intervention", fontsize=10)
ax.text(1.5,3.5,"Updated State", fontsize=10)

# Formatting
ax.set_xlim(-1,12)
ax.set_ylim(1,6)
ax.axis('off')

plt.title(
"Selective Intelligence: Reinforcement Learning Framework for Adaptive AI Restraint",
fontsize=14,
weight="bold"
)

plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.show()

print(f"\nArchitecture diagram saved to:\n{output_path}")