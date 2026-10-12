"""
Chapter 9: Policy Function Approximation (Policy Gradient).
Implements the REINFORCE algorithm with State-Value Baseline in PyTorch.
Accelerated on GPU (CUDA) if available.
"""

from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import numpy as np


class PolicyNetwork(nn.Module):
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
        logits = self.net(x)
        return logits


class ValueBaselineNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class REINFORCEAgent:
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        gamma: float = 0.9,
        lr_policy: float = 1e-3,
        lr_value: float = 1e-3,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.gamma = gamma
        self.num_actions = num_actions

        self.policy_net = PolicyNetwork(state_dim, num_actions).to(self.device)
        self.baseline_net = ValueBaselineNetwork(state_dim).to(self.device)

        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=lr_policy)
        self.optimizer_baseline = optim.Adam(self.baseline_net.parameters(), lr=lr_value)

        self.saved_log_probs: List[torch.Tensor] = []
        self.saved_states: List[torch.Tensor] = []
        self.rewards: List[float] = []

    def select_action(self, state: np.ndarray) -> int:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits = self.policy_net(state_t)
        categorical = dist.Categorical(logits=logits)
        action = categorical.sample()

        self.saved_log_probs.append(categorical.log_prob(action))
        self.saved_states.append(state_t)
        return int(action.item())

    def record_reward(self, reward: float):
        self.rewards.append(reward)

    def update(self) -> Tuple[float, float]:
        """
        Updates policy and baseline networks via Policy Gradient with baseline.
        """
        G = 0.0
        returns = []
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns_t = torch.FloatTensor(returns).unsqueeze(1).to(self.device)
        # Normalize returns for training stability
        if len(returns) > 1 and returns_t.std() > 1e-6:
            returns_norm = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
        else:
            returns_norm = returns_t

        states_t = torch.cat(self.saved_states, dim=0)
        baseline_vals = self.baseline_net(states_t)
        advantages = returns_norm - baseline_vals.detach()

        # Policy loss: - sum log_prob * Advantage
        log_probs = torch.stack(self.saved_log_probs)
        policy_loss = -(log_probs * advantages).mean()

        # Baseline loss: MSE(V(s), Return)
        baseline_loss = nn.functional.mse_loss(baseline_vals, returns_t)

        # Optimize
        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        self.optimizer_policy.step()

        self.optimizer_baseline.zero_grad()
        baseline_loss.backward()
        self.optimizer_baseline.step()

        # Reset episode buffer
        self.saved_log_probs.clear()
        self.saved_states.clear()
        self.rewards.clear()

        return float(policy_loss.item()), float(baseline_loss.item())
