import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output = os.path.join(project_root,"results_advanced","full_system_pipeline.png")

fig, ax = plt.subplots(figsize=(14,8))

def box(text,x,y,w=3,h=1.2,color="#E8F0FE"):
    rect = patches.FancyBboxPatch(
        (x,y),w,h,
        boxstyle="round,pad=0.02",
        linewidth=2,
        edgecolor="black",
        facecolor=color
    )
    ax.add_patch(rect)
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=10)

def arrow(x1,y1,x2,y2):
    ax.annotate("",
        xy=(x2,y2),
        xytext=(x1,y1),
        arrowprops=dict(arrowstyle="->",lw=2)
    )

# Layer 1
box("Learner Environment\n(Cognitive Simulation)\nSkill | Frustration | Performance\nHelp Dependency | Difficulty",0,5)

# Layer 2
box("State Representation\nEncoded RL State",4,5)

# Layer 3
box("RL Agent\n(Q-Learning Policy)",8,5)

# Layer 4
box("Intervention Actions\nPause | Hint | Socratic | Explain",4,3)

# Layer 5
box("Environment Update\nLearner State Transition",0,3)

# Layer 6
box("Reward Signal\nLearning Progress",8,3)

# Layer 7
box("Policy Update\nQ-Table Update",8,1)

# Layer 8
box("Evaluation & Analytics\nReward | Autonomy | Stability\nSkill-Difficulty Alignment\nCognitive Maps | Heatmaps",4,1,color="#E6FFE6")

# Arrows
arrow(3,5.6,4,5.6)
arrow(7,5.6,8,5.6)

arrow(9.5,5,9.5,4)
arrow(8,3.6,7,3.6)
arrow(4,3.6,3,3.6)
arrow(1.5,4,1.5,5)

arrow(9.5,3,9.5,2.2)
arrow(7,1.6,4,1.6)

# Formatting
ax.set_xlim(-1,12)
ax.set_ylim(0,7)
ax.axis("off")

plt.title(
"Selective Intelligence Framework\nReinforcement Learning System for Adaptive AI Restraint",
fontsize=15,
weight="bold"
)

plt.tight_layout()
plt.savefig(output,dpi=300)
plt.show()

print("\nFull system pipeline diagram saved to:\n",output)