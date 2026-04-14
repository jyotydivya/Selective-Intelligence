import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
import os
import random

# ------------------------------------------------
# Add project root
# ------------------------------------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from rl_agent.q_learning_advanced import AdvancedQLearningAgent
from environment.advanced_learner_env import AdvancedLearnerEnv

# ------------------------------------------------
# Page config
# ------------------------------------------------
st.set_page_config(page_title="Selective Intelligence RL Tutor", layout="wide")

# ------------------------------------------------
# Dark neon CSS styling
# ------------------------------------------------
st.markdown("""
<style>

div[data-testid="stMetric"] {
    background-color: #161B22;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #00F5FF;
}

div[data-testid="stSidebar"] {
    background-color: #161B22;
}

.stTextInput input {
    background-color: #161B22;
    color: white;
}

.stButton button {
    background-color: #00F5FF;
    color: black;
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)

st.title("🧠 Selective Intelligence RL Tutor")
st.caption("Adaptive Reinforcement Learning Tutoring System")

# ------------------------------------------------
# Sidebar controls
# ------------------------------------------------
st.sidebar.header("Simulation Settings")

learner_type = st.sidebar.selectbox(
    "Learner Type",
    ["Beginner", "Intermediate", "Advanced"]
)

steps = st.sidebar.slider("Simulation Steps", 10, 150, 60)

speed = st.sidebar.slider("Animation Speed", 0.0, 0.4, 0.05)

run_button = st.sidebar.button("Run Simulation")

type_map = {
    "Beginner":0,
    "Intermediate":1,
    "Advanced":2
}

# ------------------------------------------------
# Layout
# ------------------------------------------------
left, right = st.columns(2)

state_panel = left.empty()
action_panel = right.empty()

charts = st.empty()
heatmap_panel = st.empty()

# ------------------------------------------------
# Avatar function
# ------------------------------------------------
def learner_avatar(env):

    if env.frustration == 2:
        return "😖"
    elif env.performance == 2:
        return "😃"
    elif env.help_dependency == 2:
        return "🤔"
    else:
        return "🙂"

# ------------------------------------------------
# Gauge drawing
# ------------------------------------------------
def draw_gauge(label,value,max_value):

    percentage = value/max_value

    fig, ax = plt.subplots(figsize=(4,0.5))

    ax.barh([0],[percentage],color="#00F5FF")
    ax.barh([0],[1-percentage],left=[percentage],color="#333")

    ax.set_xlim(0,1)
    ax.axis("off")

    ax.set_title(f"{label}: {value}")

    st.pyplot(fig)

# ------------------------------------------------
# Tutor reasoning
# ------------------------------------------------
def tutor_reason(action):

    if action == 0:
        return "Learner appears confident. Allowing independent thinking."

    if action == 1:
        return "Learner struggling slightly. Providing a hint."

    if action == 2:
        return "Encouraging deeper reasoning with Socratic questioning."

    if action == 3:
        return "Learner needs help. Providing explanation."

# ------------------------------------------------
# RL Simulation
# ------------------------------------------------
if run_button:

    env = AdvancedLearnerEnv(learner_type=type_map[learner_type])
    agent = AdvancedQLearningAgent()

    rewards=[]
    autonomy=[]
    actions=[]

    total_success=0
    success_after_expl=0

    for step in range(steps):

        state=env.get_state()

        action=agent.choose_action(state)

        next_state,reward,success=env.step(action)

        agent.update(state,action,reward,next_state)

        rewards.append(reward)
        actions.append(action)

        if success:
            total_success+=1
            if action==3:
                success_after_expl+=1

        if total_success==0:
            autonomy.append(0)
        else:
            autonomy.append((total_success-success_after_expl)/total_success)

        action_names=["Pause","Hint","Socratic","Explain"]

        # ----------------------------
        # Cognitive State Panel
        # ----------------------------
        with state_panel.container():

            st.subheader("Learner State")

            avatar=learner_avatar(env)
            st.markdown(f"# {avatar}")

            draw_gauge("Skill",env.skill_level,3)
            draw_gauge("Frustration",env.frustration,2)
            draw_gauge("Performance",env.performance,2)
            draw_gauge("Help Dependency",env.help_dependency,2)
            draw_gauge("Difficulty",env.difficulty,3)

        # ----------------------------
        # Tutor Action Panel
        # ----------------------------
        with action_panel.container():

            st.subheader("Tutor Decision")

            st.metric("Action",action_names[action])
            st.metric("Reward",reward)

            st.success(tutor_reason(action))

        # ----------------------------
        # Charts
        # ----------------------------
        fig,ax=plt.subplots(1,2,figsize=(10,4))

        ax[0].plot(rewards,color="cyan")
        ax[0].set_title("Reward Progress")

        ax[1].plot(autonomy,color="green")
        ax[1].set_title("Autonomy Growth")

        charts.pyplot(fig)

        # ----------------------------
        # Action heatmap
        # ----------------------------
        bins=20
        heatmap=np.zeros((4,bins))

        for i,a in enumerate(actions):
            heatmap[a][i%bins]+=1

        fig2,ax2=plt.subplots(figsize=(8,3))

        sns.heatmap(
            heatmap,
            cmap="cool",
            yticklabels=["Pause","Hint","Socratic","Explain"],
            ax=ax2
        )

        ax2.set_title("Action Distribution")

        heatmap_panel.pyplot(fig2)

        time.sleep(speed)

# ------------------------------------------------
# Question Answer Tutor
# ------------------------------------------------
st.markdown("---")
st.header("Ask the Tutor")

question=st.text_input("Enter your question")

knowledge_base={
    "what is reinforcement learning":"Reinforcement learning is a machine learning approach where an agent learns by interacting with an environment and receiving rewards.",
    "what is q learning":"Q-learning is a reinforcement learning algorithm that learns the value of actions in states to maximize cumulative reward.",
    "what is autonomy":"Autonomy refers to the learner solving problems independently without excessive AI intervention.",
    "what is a policy":"A policy defines how an agent selects actions based on the current state."
}

if st.button("Get Tutor Response"):

    q=question.lower()

    if q in knowledge_base:
        st.success(knowledge_base[q])

    else:
        st.warning("I'm not sure about that. Let me give you a hint instead.")
        st.info("Try asking about reinforcement learning, Q-learning, policy, or autonomy.")

# ------------------------------------------------
# Adaptive Learning Exercise
# ------------------------------------------------
st.markdown("---")
st.header("Learning Exercise")

questions = {
    "What does RL stand for?": "reinforcement learning",
    "Which algorithm is used in this tutor?": "q learning",
    "What does the agent maximize?": "reward"
}

selected_question = st.selectbox(
    "Choose a question",
    list(questions.keys())
)

user_answer = st.text_input("Your Answer")

if st.button("Submit Answer"):

    correct = questions[selected_question]

    if user_answer.lower() == correct:

        st.success("Correct! 🎉")

    else:

        st.error("Not quite correct.")
        st.info("Tutor Hint: Think about how the agent learns from rewards.")

st.markdown("---")
st.caption("Interactive Demonstration of the Selective Intelligence RL Tutor")