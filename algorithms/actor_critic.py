"""
Chapter 10: Actor-Critic Methods (Fundamental Tools).
Implements Advantage Actor-Critic (A2C) with TD error and Entropy Regularization.
Accelerated on GPU (CUDA) if available.
"""

from typing import Tuple, List, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import numpy as np


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 128):
        super().__init__()
        # Shared feature trunk
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Actor head (policy logits)
        self.actor_head = nn.Linear(hidden_dim, num_actions)
        # Critic head (state value)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.trunk(x)
        logits = self.actor_head(feat)
        value = self.critic_head(feat)
        return logits, value


class A2CAgent:
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        gamma: float = 0.9,
        lr: float = 7e-4,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef

        self.ac_net = ActorCriticNetwork(state_dim, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=lr)

    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor, torch.Tensor]:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits, value = self.ac_net(state_t)
        categorical = dist.Categorical(logits=logits)
        action = categorical.sample()
        log_prob = categorical.log_prob(action)
        return int(action.item()), log_prob, value

    def compute_step_loss(
        self,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> torch.Tensor:
        """
        Calculates 1-step TD Actor-Critic loss.
        """
        with torch.no_grad():
            if done:
                target_value = torch.tensor([[reward]], dtype=torch.float32, device=self.device)
            else:
                next_s_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                _, next_val = self.ac_net(next_s_t)
                target_value = reward + self.gamma * next_val

            advantage = target_value - value

        # Actor Loss
        actor_loss = -log_prob * advantage

        # Critic Loss
        critic_loss = nn.functional.smooth_l1_loss(value, target_value)

        # Total Loss
        total_loss = actor_loss + self.value_loss_coef * critic_loss
        return total_loss

    def update(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.ac_net.parameters(), max_norm=0.5)
        self.optimizer.step()
