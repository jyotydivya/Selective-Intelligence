import numpy as np
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


# -------------------------------
# Neural Network
# -------------------------------
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------
# DQN Agent
# -------------------------------
class DQNAgent:

    def __init__(self, state_size=6, action_size=4):

        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=5000)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.lr = 0.001

        self.model = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def choose_action(self, state):

        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        state = torch.FloatTensor(state)
        q_values = self.model(state)

        return torch.argmax(q_values).item()

    def remember(self, s, a, r, s_next):

        self.memory.append((s, a, r, s_next))

    def replay(self, batch_size=32):

        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)

        for s, a, r, s_next in batch:

            s = torch.FloatTensor(s)
            s_next = torch.FloatTensor(s_next)

            target = r + self.gamma * torch.max(self.model(s_next)).item()

            target_f = self.model(s).detach().clone()
            target_f[a] = target

            output = self.model(s)

            loss = self.criterion(output, target_f)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay