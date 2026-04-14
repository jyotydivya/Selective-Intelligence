import numpy as np
import random
import math

class AdvancedQLearningAgent:

    def __init__(self, 
                 alpha=0.18, 
                 gamma=0.92,
                 epsilon_max=1.0,
                 epsilon_min=0.05,
                 decay_rate=0.0008,
                 temperature=0.7):

        # Hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.decay_rate = decay_rate
        self.temperature = temperature

        # -------------------------------------------------------------
        # Correct State Dimensions (MATCHES ENVIRONMENT EXACTLY)
        # learner_type:     3   (0–2)
        # skill_level:      4   (0–3)
        # performance:      3   (0–2)
        # frustration:      3   (0–2)
        # help_dependency:  3   (0–2)
        # difficulty:       4   (0–3)
        # -------------------------------------------------------------
        self.state_dims = (3, 4, 3, 3, 3, 4)
        self.num_states = np.prod(self.state_dims)

        # 4 possible actions: Pause, Hint, Socratic, Explanation
        self.num_actions = 4

        # Initialize Q-Table
        self.Q = np.zeros((self.num_states, self.num_actions))

        # timestep counter for epsilon decay
        self.t = 0


    # ---------------------------------------------------
    # State → Integer Encoding (Corrected for 4 difficulty values)
    # ---------------------------------------------------
    def encode_state(self, state):

        (lt, skill, perf, frust, dep, diff) = state

        # Multiply last by 4 (difficulty range 0–3)
        idx = (
            ((((lt * 4 + skill) * 3 + perf) * 3 + frust) * 3 + dep) * 4 + diff
        )
        return idx


    # ---------------------------------------------------
    # Epsilon Decay Function
    # ---------------------------------------------------
    def epsilon(self):
        return self.epsilon_min + (self.epsilon_max - self.epsilon_min) * math.exp(-self.decay_rate * self.t)


    # ---------------------------------------------------
    # Softmax Action Selection
    # ---------------------------------------------------
    def softmax_action(self, q_values):
        tau = self.temperature
        exp_q = np.exp(q_values / tau)
        probs = exp_q / np.sum(exp_q)
        return np.random.choice(len(q_values), p=probs)


    # ---------------------------------------------------
    # Hybrid Exploration Strategy:
    # ε-greedy + Softmax
    # ---------------------------------------------------
    def choose_action(self, state):

        state_idx = self.encode_state(state)
        q_values = self.Q[state_idx]

        eps = self.epsilon()
        self.t += 1  # update exploration schedule

        if random.random() < eps:
            return random.randint(0, self.num_actions - 1)
        else:
            return self.softmax_action(q_values)


    # ---------------------------------------------------
    # Q-Learning Update Rule
    # ---------------------------------------------------
    def update(self, state, action, reward, next_state):

        s = self.encode_state(state)
        s_next = self.encode_state(next_state)

        best_next = np.max(self.Q[s_next])

        td_target = reward + self.gamma * best_next
        td_error = td_target - self.Q[s][action]

        self.Q[s][action] += self.alpha * td_error


    # ---------------------------------------------------
    # Frobenius Norm Distance Between Q-table Snapshots
    # ---------------------------------------------------
    def policy_distance(self, Q_prev):
        return np.linalg.norm(self.Q - Q_prev)