# Selective Intelligence: Adaptive Tutoring with Reinforcement Learning

## Overview

Selective Intelligence is an innovative project that leverages reinforcement learning (RL) to create adaptive tutoring systems. The system intelligently decides when and how to intervene in a learner's problem-solving process, optimizing for different learner types: Beginner, Intermediate, and Advanced. By using Q-learning and Deep Q-Network (DQN) agents, the project explores how AI can provide personalized educational support that balances learner autonomy with necessary guidance.

The project simulates realistic learner environments where agents learn to choose from four intervention strategies:
- **Pause**: Allow the learner to continue independently
- **Hint**: Provide subtle guidance
- **Socratic**: Ask guiding questions
- **Explanation**: Give direct instructional support

## Key Features

### Multi-Agent Reinforcement Learning
- **Q-Learning Agent**: Traditional tabular Q-learning with advanced exploration strategies
- **DQN Agent**: Deep neural network-based Q-learning for complex state spaces
- **Hybrid Exploration**: Combines ε-greedy with softmax action selection

### Learner Modeling
- **Three Learner Types**: Beginner, Intermediate, and Advanced with distinct characteristics
- **Dynamic State Tracking**: Monitors skill level, performance, frustration, help dependency, and difficulty
- **Realistic Progression**: Learners grow skills, experience frustration, and develop dependencies on help

### Comprehensive Simulation Framework
- **Experiment Runner**: Automated experiments for different learner types
- **Baseline Comparisons**: Rule-based and always-help strategies for benchmarking
- **Performance Analysis**: Detailed metrics including autonomy, performance, and frustration levels

### Interactive Dashboard
- **Streamlit UI**: Real-time visualization of agent decisions and learner states
- **Live Simulation**: Watch agents learn and adapt in real-time
- **Performance Metrics**: Interactive charts showing learning curves and decision patterns

### Advanced Analytics
- **Heatmap Analysis**: Visualize action distributions across episodes
- **Cognitive Mapping**: Explore relationships between frustration, performance, and skill
- **Comparative Studies**: Q-learning vs DQN performance across learner types

## Project Structure

```
selective_intelligence/
├── environment/
│   ├── __init__.py
│   ├── advanced_learner_env.py    # Learner environment simulation
│   └── state_manager.py           # State management utilities
├── rl_agent/
│   ├── __init__.py
│   ├── q_learning_advanced.py     # Q-learning agent implementation
│   ├── dqn_agent.py               # DQN agent implementation
│   └── reward_scheme.py           # Reward calculation logic
├── simulation/
│   ├── __init__.py
│   ├── learner_type_experiments.py    # Main experiment runner
│   ├── train_advanced.py              # Advanced training scripts
│   ├── train_dqn.py                   # DQN training scripts
│   ├── analysis_*.py                  # Various analysis scripts
│   ├── baseline_*.py                  # Baseline comparison scripts
│   ├── compare_*.py                   # Model comparison scripts
│   ├── plot_*.py                      # Visualization scripts
│   ├── export_*.py                    # Data export utilities
│   └── generate_*.py                  # Diagram generation
├── ui/
│   ├── __init__.py
│   └── app.py                         # Streamlit dashboard
├── results/                           # Experiment results
├── learner_type_results/              # Per-learner type results
├── results_csv/                       # CSV exports
├── main.py                            # Main entry point (currently empty)
└── README.md                          # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Dependencies
Install the required packages:

```bash
pip install numpy matplotlib seaborn streamlit torch
```

For GPU acceleration (optional):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd selective_intelligence
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

*Note: A requirements.txt file may need to be created based on the dependencies listed above.*

## Usage

### Running Experiments

#### Single Learner Type Experiment
Run experiments for specific learner types:

```bash
cd simulation
python learner_type_experiments.py
```

This will run 4000 episodes for Beginner, Intermediate, and Advanced learners, generating plots and saving results.

#### Training RL Agents
Train the Q-learning agent:
```bash
python train_advanced.py
```

Train the DQN agent:
```bash
python train_dqn.py
```

#### Comparative Analysis
Compare Q-learning vs DQN performance:
```bash
python compare_q_vs_dqn.py
```

Run all model comparisons:
```bash
python compare_all_models.py
```

### Interactive Dashboard
Launch the Streamlit dashboard:

```bash
cd ui
streamlit run app.py
```

The dashboard provides:
- Real-time agent decision visualization
- Learner state monitoring
- Interactive performance charts
- Live simulation controls

### Analysis and Visualization

#### Generate Heatmaps
```bash
python analysis_heatmap.py
python analysis_heatmap_by_type.py
```

#### Plot Reward Comparisons
```bash
python plot_reward_comparison.py
python plot_reward_by_type.py
```

#### Export Results
Export results to CSV:
```bash
python export_results_csv.py
python export_combined_csv.py
```

#### Generate Diagrams
Create system architecture diagrams:
```bash
python generate_architecture_diagram.py
python generate_full_system_diagram.py
```

Create dashboard:
```bash
python generate_dashboard.py
```

## Understanding the Model

### State Space
The learner state is represented by 6 dimensions:
- **Learner Type**: 0 (Beginner), 1 (Intermediate), 2 (Advanced)
- **Skill Level**: 0-3 (increasing proficiency)
- **Performance**: 0-2 (recent success rate)
- **Frustration**: 0-2 (emotional state)
- **Help Dependency**: 0-2 (reliance on assistance)
- **Difficulty**: 0-3 (task complexity)

### Action Space
Four intervention strategies:
1. **Pause** (0): No intervention, encourage independence
2. **Hint** (1): Minimal guidance
3. **Socratic** (2): Question-based guidance
4. **Explanation** (3): Direct instructional support

### Reward Structure
Rewards are differentiated by learner type and success:
- **Beginners**: Prefer explanations, penalize failures heavily
- **Intermediates**: Balanced rewards across actions
- **Advanced**: Prefer autonomy, minimal help rewards

### Learning Dynamics
- **Skill Growth**: Learners improve skills upon success
- **Frustration Management**: Unsuccessful attempts increase frustration
- **Help Dependency**: Frequent explanations increase dependency
- **Difficulty Adjustment**: Adaptive task complexity based on performance

## Results and Analysis

### Key Findings
- Advanced learners achieve higher autonomy with minimal interventions
- Beginners benefit most from structured explanations
- Q-learning shows stable convergence across learner types
- DQN provides more nuanced decision-making in complex scenarios

### Performance Metrics
- **Autonomy**: Ratio of independent successes to total successes
- **Performance**: Recent success rate (0-2 scale)
- **Frustration**: Emotional state affecting learning (0-2 scale)
- **Help Dependency**: Reliance on external assistance (0-2 scale)

### Visualization Outputs
- Reward convergence curves
- Action distribution heatmaps
- Cognitive state scatter plots
- Comparative performance charts

## Configuration

### Hyperparameters
Key parameters can be adjusted in the agent classes:

**Q-Learning Agent** (`q_learning_advanced.py`):
- `alpha`: Learning rate (default: 0.18)
- `gamma`: Discount factor (default: 0.92)
- `epsilon_max/min`: Exploration bounds (1.0/0.05)
- `decay_rate`: Exploration decay (0.0008)
- `temperature`: Softmax temperature (0.7)

**DQN Agent** (`dqn_agent.py`):
- Network architecture: Configurable hidden layers
- Replay buffer size: Experience storage capacity
- Target network update frequency

### Environment Settings
Modify learner characteristics in `advanced_learner_env.py`:
- Initial state distributions
- Reward scaling factors
- Growth and frustration rates
- Success probability modifiers

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Include unit tests for new features
- Update documentation for API changes

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Future Work

- **Multi-Agent Systems**: Collaborative tutoring agents
- **Curriculum Learning**: Adaptive difficulty sequencing
- **User Studies**: Real-world validation with human learners
- **Extended Domains**: Apply to other educational contexts
- **Transfer Learning**: Cross-domain knowledge transfer

## Citation

If you use this work in your research, please cite:

```
@misc{selective_intelligence_2024,
  title={Selective Intelligence: Adaptive Tutoring with Reinforcement Learning},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/selective_intelligence}
}
```

## Contact

For questions or collaborations, please open an issue on GitHub or contact the maintainers.