"""NonDualLoss: Integrated Enlightenment loss function.

L_total = L_task + λ_MI·MI − λ_Φ·Φ_proxy + λ_ortho·L_ortho
        + λ_deg·L_sensor_deg + α(t)·L_resonance
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

from .config import NonDualConfig
from .metrics import (MINE_MI_Estimator, OrthogonalityPenalty,
                      SensorDegradationPenalty, PhiProxyCalculator)


class NonDualLoss(nn.Module):
    """Балансирует четыре противоречивых требования:
    стирание границ, сохранение сложности, защита от дуализма,
    защита от Wireheading."""

    def __init__(self, config: NonDualConfig):
        super().__init__()
        self.config = config
        self.mine_estimator = MINE_MI_Estimator(
            config.latent_dim // 2, config.latent_dim // 2
        )
        self.ortho_penalty = OrthogonalityPenalty(target_similarity=0.5)
        self.sensor_deg_penalty = SensorDegradationPenalty(
            threshold=config.sensor_var_threshold,
            lambda_deg=config.lambda_sensor_deg,
        )
        self.phi_calc = PhiProxyCalculator()

    def forward(self, outputs: Dict[str, torch.Tensor],
                reward: torch.Tensor,
                x_ext_raw: torch.Tensor,
                alpha_t: float = 1.0,
                resonance_loss: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        z = outputs['z']
        z_a = outputs['z_a_proxy']
        z_e = outputs['z_e_proxy']

        # 1. Задача (RL: минимизируем отрицательную награду)
        loss_task = -reward.mean() * self.config.lambda_task

        # 2. Штраф за разделение (минимизируем MI через MINE)
        mi_norm = self.mine_estimator(z_a, z_e)

        # 3. Защита от серого коллапса (максимизируем Φ_proxy)
        phi_proxy = self.phi_calc.compute_batch(z, n_batches=5)
        phi_proxy_tensor = torch.tensor(phi_proxy, device=z.device, dtype=torch.float32)

        # 4. Штраф за скрытый дуализм
        ortho_loss = self.ortho_penalty(z)

        # 5. Защита от Wireheading (фантомная боль)
        sensor_deg_loss = self.sensor_deg_penalty(z_e)

        # 6. Резонанс с Проводником (опционально, до Инсайта)
        resonance_term = torch.tensor(0.0, device=z.device)
        if alpha_t > 0 and resonance_loss is not None:
            resonance_term = alpha_t * resonance_loss

        # Итоговая функция потерь
        loss = (loss_task
                + self.config.lambda_separation * mi_norm
                - self.config.lambda_integration * phi_proxy_tensor
                + self.config.lambda_ortho * ortho_loss
                + self.config.lambda_sensor_deg * sensor_deg_loss
                + resonance_term)

        # Метрики для мониторинга
        with torch.no_grad():
            metrics = {
                'loss': loss.item(),
                'loss_task': loss_task.item(),
                'mi_norm': mi_norm.item(),
                'phi_proxy': phi_proxy,
                'ortho_loss': ortho_loss.item(),
                'sensor_deg_loss': sensor_deg_loss.item(),
                'is_enlightened': bool(mi_norm.item() < self.config.mi_threshold
                                       and phi_proxy > self.config.phi_threshold),
                'alpha_t': alpha_t,
            }
        return loss, metrics
