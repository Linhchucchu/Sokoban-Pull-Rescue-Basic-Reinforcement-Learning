"""
Chapter 8: Value Function Approximation (Linear FA & Deep Q-Network).
PyTorch Deep Q-Network (DQN) and Double DQN with Replay Buffer and Target Network.
Accelerated on GPU (CUDA) if available.
"""

from typing import Tuple, List, Dict, Any, Optional
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        gamma: float = 0.9,
        lr: float = 1e-3,
        double_dqn: bool = True,
        buffer_size: int = 50000,
        batch_size: int = 64,
        target_update_interval: int = 500,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.double_dqn = double_dqn

        self.q_net = QNetwork(state_dim, num_actions).to(self.device)
        self.target_net = QNetwork(state_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber Loss
        self.replay_buffer = ReplayBuffer(buffer_size)

        self.step_count = 0

    def select_action(self, state: np.ndarray, epsilon: float = 0.05) -> int:
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def update(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        s_t = torch.FloatTensor(states).to(self.device)
        a_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        r_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_s_t = torch.FloatTensor(next_states).to(self.device)
        done_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Q(s, a)
        q_eval = self.q_net(s_t).gather(1, a_t)

        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: action selected by online net, evaluated by target net
                best_actions = self.q_net(next_s_t).argmax(dim=1, keepdim=True)
                q_next = self.target_net(next_s_t).gather(1, best_actions)
            else:
                q_next = self.target_net(next_s_t).max(dim=1, keepdim=True)[0]

            q_target = r_t + self.gamma * q_next * (1.0 - done_t)

        loss = self.loss_fn(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())
