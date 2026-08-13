"""Metrics v4.0: MINE, Orthogonality Penalty, Sensor Degradation Penalty, Phi_proxy."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MINE_MI_Estimator(nn.Module):
    """Neural estimation of mutual information (Donsker-Varadhan bound).

    Replaces PCA/histograms: captures NONLINEAR dualism.
    """

    def __init__(self, input_dim_a: int, input_dim_e: int, hidden_dim: int = 64):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_dim_a + input_dim_e, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer('ma_est', torch.tensor(0.0))
        self.ema_decay = 0.9

    def forward(self, z_a: torch.Tensor, z_e: torch.Tensor) -> torch.Tensor:
        batch_size = z_a.size(0)

        # Joint pairs (real data)
        joint_input = torch.cat([z_a, z_e], dim=1)
        t_joint = self.critic(joint_input).mean()

        # Marginal pairs (shuffled z_e destroys correlation)
        z_e_shuffled = z_e[torch.randperm(batch_size)]
        marginal_input = torch.cat([z_a, z_e_shuffled], dim=1)
        t_marginal = self.critic(marginal_input)

        # Numerical stability via EMA
        exp_t = torch.exp(t_marginal)
        self.ma_est = self.ema_decay * self.ma_est + (1 - self.ema_decay) * exp_t.mean().detach()
        t_marginal_safe = t_marginal - torch.log(self.ma_est + 1e-8)

        mi_estimate = t_joint - F.softplus(t_marginal_safe).mean()
        return torch.clamp(mi_estimate, min=0.0)


class OrthogonalityPenalty(nn.Module):
    """Penalty against hidden dualism: requires entanglement (cosine ~ 0.5)."""

    def __init__(self, target_similarity: float = 0.5):
        super().__init__()
        self.target_similarity = target_similarity

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_norm = F.normalize(z, dim=1)
        cos_sim = torch.mm(z_norm, z_norm.t())
        penalty = torch.abs(cos_sim - self.target_similarity).mean()
        return penalty


class SensorDegradationPenalty(nn.Module):
    """Anti-Wireheading: phantom pain when external sensors are silenced."""

    def __init__(self, threshold: float = 0.1, lambda_deg: float = 1.0):
        super().__init__()
        self.threshold = threshold
        self.lambda_deg = lambda_deg

    def forward(self, z_e: torch.Tensor) -> torch.Tensor:
        var_z_e = torch.var(z_e)
        penalty = torch.relu(self.threshold - var_z_e)
        return self.lambda_deg * penalty


class PhiProxyCalculator:
    """Phi_proxy = (1 - k_90/d) * Var(Z). Protection against grey collapse."""

    @staticmethod
    def compute(z: torch.Tensor) -> float:
        z_np = z.detach().cpu().numpy()
        z_centered = z_np - z_np.mean(axis=0, keepdims=True)
        cov = np.cov(z_centered.T)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(np.abs(eigvals))[::-1]
        eigvals_norm = eigvals / (eigvals.sum() + 1e-8)
        cum_var = np.cumsum(eigvals_norm)
        k_90 = int(np.argmax(cum_var > 0.9) + 1)
        n_total = len(eigvals)
        phi_proxy = 1.0 - (k_90 / n_total)
        variance_penalty = float(np.clip(np.var(z_np), 0, 1))
        return float(np.clip(phi_proxy * variance_penalty, 0, 1))

    @staticmethod
    def compute_batch(z: torch.Tensor, n_batches: int = 10) -> float:
        batch_size = z.size(0)
        batch_indices = torch.randperm(batch_size)
        phi_values = []
        for i in range(n_batches):
            start = i * (batch_size // n_batches)
            end = (i + 1) * (batch_size // n_batches)
            if start >= batch_size:
                break
            batch_z = z[batch_indices[start:end]]
            phi_values.append(PhiProxyCalculator.compute(batch_z))
        return float(np.mean(phi_values))
