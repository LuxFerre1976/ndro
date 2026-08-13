"""Shared Encoder: unified stream without split_point.

The boundary between 'Self' and 'World' is dissolved at the
computational level. Diagnostic projections are computed under
torch.no_grad() and never influence training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from .config import NonDualConfig


class SharedEncoder(nn.Module):
    """Unified encoder for all signals (internal + external)."""

    def __init__(self, config: NonDualConfig):
        super().__init__()
        self.config = config

        total_input_dim = config.input_dim_int + config.input_dim_ext

        # 1. Unified input layer (concatenation WITHOUT separation)
        self.input_projection = nn.Linear(total_input_dim, config.hidden_dim)

        # 2. Unified hidden layers
        hidden_layers = []
        for _ in range(config.num_hidden_layers):
            layer = [nn.Linear(config.hidden_dim, config.hidden_dim)]
            if config.use_batch_norm:
                layer.append(nn.BatchNorm1d(config.hidden_dim))
            layer.append(nn.GELU())
            if config.dropout_rate > 0:
                layer.append(nn.Dropout(config.dropout_rate))
            hidden_layers.extend(layer)
        self.hidden_layers = nn.Sequential(*hidden_layers)

        # 3. Unified latent projection (NO BOUNDARY HERE)
        self.latent_projection = nn.Linear(config.hidden_dim, config.latent_dim)

        # 4. Diagnostic probes (metrics only, no gradients)
        self.probe_a = nn.Linear(config.latent_dim, config.latent_dim // 2, bias=False)
        self.probe_e = nn.Linear(config.latent_dim, config.latent_dim // 2, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x_int: torch.Tensor, x_ext: torch.Tensor) -> Dict[str, torch.Tensor]:
        # All signals into a single stream
        x_combined = torch.cat([x_int, x_ext], dim=-1)

        h = F.gelu(self.input_projection(x_combined))
        h = self.hidden_layers(h)

        # Unified latent representation
        z = self.latent_projection(h)

        # Diagnostics without gradients
        with torch.no_grad():
            z_a_proxy = self.probe_a(z)
            z_e_proxy = self.probe_e(z)

        return {
            'z': z,
            'z_a_proxy': z_a_proxy,
            'z_e_proxy': z_e_proxy,
        }


class PolicyHead(nn.Module):
    """Policy head for reinforcement learning."""

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.policy_net(z)


class ValueHead(nn.Module):
    """Value head (actor-critic)."""

    def __init__(self, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.value_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.value_net(z)


class NonDualAgent(nn.Module):
    """Full agent: SharedEncoder + PolicyHead + ValueHead."""

    def __init__(self, config: NonDualConfig, action_dim: int):
        super().__init__()
        self.config = config
        self.encoder = SharedEncoder(config)
        self.policy_head = PolicyHead(config.latent_dim, action_dim)
        self.value_head = ValueHead(config.latent_dim)

    def forward(self, x_int: torch.Tensor, x_ext: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc_out = self.encoder(x_int, x_ext)
        z = enc_out['z']
        return {
            'z': z,
            'z_a_proxy': enc_out['z_a_proxy'],
            'z_e_proxy': enc_out['z_e_proxy'],
            'action_logits': self.policy_head(z),
            'value': self.value_head(z),
        }

    def get_action(self, x_int: torch.Tensor, x_ext: torch.Tensor,
                   deterministic: bool = False) -> Tuple[int, Dict[str, torch.Tensor]]:
        outputs = self.forward(x_int, x_ext)
        logits = outputs['action_logits']
        if deterministic:
            action = torch.argmax(logits, dim=-1).item()
        else:
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
        return action, outputs
