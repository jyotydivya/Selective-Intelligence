# Selective Intelligence: A Reinforcement Learning Framework for Adaptive AI Restraint

> **An RL-powered intelligent tutoring system that learns when to help, hint, question, or stay silent.**

Selective Intelligence is an interactive AI tutoring prototype that uses **Reinforcement Learning (Q-Learning)** to decide **how and when an AI tutor should intervene** during learning.

Instead of always giving direct answers, the system learns to choose among different tutoring strategies such as:

- **Pause** - allow independent thinking
- **Hint** - provide minimal guidance
- **Socratic Question** - encourage reasoning
- **Explanation** - provide direct help

The project simulates learner behavior using cognitive state variables such as **skill level, frustration, performance, help dependency, and difficulty**. It trains an RL agent to maximize learning while minimizing unnecessary intervention.

The main idea is:

> **AI should not always give the answer immediately; it should learn when to step back.**

---

## Project Motivation

Modern AI tutors often **over-help**. This can make learners dependent on the system instead of encouraging independent problem solving.

This project explores the idea of **Adaptive AI Restraint**, where an AI tutor learns:

- when to **intervene**
- when to **give a hint**
- when to **ask a guiding question**
- and when to **stay silent**

This approach helps the tutor better meet real educational goals such as:

- improving learner autonomy
- reducing over-dependence on AI
- adapting to different learner states

---

## Objectives

- Build an **interactive AI tutoring system**
- Use **Q-Learning** to learn adaptive tutoring strategies
- Model learner cognition using simulated states
- Encourage **autonomous learning**
- Analyze tutoring effectiveness using learning analytics
- Show how AI can optimize **when** to help, not just **what** to say

---

## Key Features

### Reinforcement Learning Tutor
A Q-Learning agent learns the best intervention strategy based on the learner’s current cognitive state.

### Cognitive Learner Model
The simulated learner is modeled using:

- **Skill Level**
- **Frustration**
- **Performance**
- **Help Dependency**
- **Difficulty**

### Adaptive Tutoring Actions
The AI tutor can choose from:

- **Pause**
- **Hint**
- **Socratic Question**
- **Explanation**

### Learning Analytics
The system tracks:

- Reward progression
- Autonomy score
- Policy stability
- Action distribution
- Cognitive state evolution

### Interactive UI
A Streamlit-based interface allows users to:

- answer learning questions
- observe AI tutor decisions
- track learner state changes
- view reward and autonomy curves in real time

---

## System Architecture

The system follows this pipeline:

```text
Learner State
   ↓
RL Agent (Q-Learning)
   ↓
Tutoring Action Selection
   ↓
Learner Response
   ↓
Reward + State Update
   ↓
Policy Learning
```

### Cognitive State Representation

The learner is represented as:

```text
(learner_type, skill_level, performance, frustration, help_dependency, difficulty)
```

---

## Reinforcement Learning Methodology

This project uses **Q-Learning**, a model-free reinforcement learning algorithm.

### State Space
The tutor observes the learner state:

- learner type
- skill level
- performance
- frustration
- help dependency
- difficulty

### Action Space
The tutor can select one of four actions:

| Action ID | Action |
|----------|--------|
| 0 | Pause |
| 1 | Hint |
| 2 | Socratic Question |
| 3 | Explanation |

### Reward Design
Rewards are assigned based on:

- learner success
- intervention efficiency
- promoting autonomy
- reducing excessive explanation dependency

### Exploration Strategy
The RL agent uses:

- **ε-greedy exploration**
- **softmax action selection**

This allows the tutor to balance **exploration** and **exploitation** while learning the best policy.

---

## Experimental Components

The project includes several analysis modules:

### 1. Reward Curve
Tracks how the RL tutor improves over time.

### 2. Autonomy Curve
Measures whether the learner becomes more independent.

### 3. Policy Stability
Tracks convergence of the Q-table.

### 4. Action Heatmap
Visualizes which interventions are chosen most often.

### 5. Difficulty–Skill Alignment
Shows whether the tutor adapts difficulty appropriately.

### 6. Cognitive Maps
Plots learner state evolution such as:

- frustration vs performance
- help dependency vs performance
- skill progression

### 7. Learner-Type Experiments
Separate experiments for:

- Beginner learners
- Intermediate learners
- Advanced learners

---

## Interactive Demo

The Streamlit UI allows users to interact with the RL tutor in real time.

### Features:
- Answer learning questions
- Observe tutor actions
- View learner state updates
- See reward and autonomy graphs
- Experience adaptive AI tutoring

### Example flow:
1. User selects a question
2. User submits an answer
3. RL tutor decides whether to:
   - pause
   - hint
   - ask a Socratic question
   - explain
4. Learner state updates
5. Reward and autonomy graphs update live

---

## Project Structure

```text
selective_intelligence/
│
├── environment/
│   └── advanced_learner_env.py
│
├── rl_agent/
│   └── q_learning_advanced.py
│
├── simulation/
│   ├── train_advanced.py
│   ├── analysis_heatmap.py
│   ├── analysis_cognitive_map.py
│   ├── learner_type_experiments.py
│   ├── export_combined_csv.py
│   ├── generate_dashboard.py
│   ├── generate_architecture_diagram_clean.py
│   └── generate_full_system_diagram.py
│
├── ui/
│   └── app.py
│
├── results_advanced/
│   ├── reward_curve.png
│   ├── autonomy_curve.png
│   ├── action_heatmap.png
│   ├── policy_stability.png
│   └── ...
│
├── combined_results.csv
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/selective-intelligence.git
cd selective-intelligence
```

### 2. Install dependencies

```bash
pip install numpy matplotlib seaborn streamlit pandas
```

---

## Running the Project

### Train the RL tutor

```bash
python simulation/train_advanced.py
```

### Generate analysis plots

```bash
python simulation/analysis_heatmap.py
python simulation/analysis_cognitive_map.py
python simulation/generate_dashboard.py
```

### Run learner-type experiments

```bash
python simulation/learner_type_experiments.py
```

### Export results as CSV

```bash
python simulation/export_combined_csv.py
```

### Launch the interactive UI

```bash
streamlit run ui/app.py
```

---

## Sample Outputs

The project generates:

- Reward learning curves
- Autonomy growth curves
- Action heatmaps
- Policy stability graphs
- Cognitive state maps
- Difficulty-skill alignment plots
- Research-style dashboard visualizations

These outputs can be used in:

- project reports
- viva presentations
- technical demonstrations

---

## Research Relevance

This project connects ideas from:

- **Reinforcement Learning**
- **Intelligent Tutoring Systems**
- **Learning Analytics**
- **Human-AI Interaction**
- **Educational AI**

It shows how AI systems can be designed not just to be **helpful**, but to be **selectively helpful**.

---

## Future Improvements

Possible extensions include:

- integrating real student datasets
- using Deep Q-Learning instead of tabular Q-Learning
- adding NLP-based answer evaluation
- integrating large language models for natural tutoring responses
- building personalized learner profiles
- deploying the tutor as a web-based educational assistant

---

## Author

**Divya Jyoty**

Project developed as part of an academic exploration into **adaptive AI tutoring, reinforcement learning, and learner autonomy**.

---

## License

This project is for **academic and educational use**.

---

## Final Note

Selective Intelligence explores a simple but powerful question:

> **Can an AI tutor learn when helping less leads to learning more?**

That is the central idea behind this project.
