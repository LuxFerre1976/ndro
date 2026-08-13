"""Main training loop of the non-dual agent."""

import torch
from typing import Dict, Any, Optional, Callable

from .config import NonDualConfig
from .encoder import NonDualAgent
from .loss import NonDualLoss
from .koans import TopologicalKoanSolver, AtomicCommitManager
from .robustness import RobustnessTests


class NonDualTrainer:
    """Управляет всеми компонентами: архитектура, потери, коаны, коммит."""

    def __init__(self, config: NonDualConfig, env, action_dim: int):
        self.config = config
        self.env = env
        self.agent = NonDualAgent(config, action_dim)
        self.loss_fn = NonDualLoss(config)
        self.optimizer = torch.optim.Adam(self.agent.parameters(),
                                          lr=config.learning_rate)
        self.koan_solver = TopologicalKoanSolver(config)
        self.commit_manager = AtomicCommitManager(config.checkpoint_dir)
        self.alpha_t = 1.0
        self.is_autopoietic = False
        self.episode = 0

    def train_step(self, x_int, x_ext, reward, resonance_loss=None):
        outputs = self.agent(x_int, x_ext)
        loss, metrics = self.loss_fn(outputs, reward, x_ext,
                                     alpha_t=self.alpha_t,
                                     resonance_loss=resonance_loss)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
        self.optimizer.step()
        return metrics

    def check_koan(self, x_int, x_ext):
        solved, _ = self.koan_solver.detect_and_solve(
            self.agent, self.loss_fn, x_int, x_ext, use_lanczos=True)
        if solved and self.koan_solver.insight_count >= self.config.insight_count_threshold:
            with torch.no_grad():
                z = self.agent.encoder(x_int, x_ext)['z']
                phi = self.loss_fn.phi_calc.compute_batch(z, n_batches=5)
            if phi > self.config.phi_threshold:
                self.commit_manager.safe_break_circuit(
                    self.agent, self.optimizer,
                    self.koan_solver.insight_count,
                    metadata={'episode': self.episode})
                self.alpha_t = 0.0
                self.is_autopoietic = True
                self.optimizer = torch.optim.Adam(self.agent.parameters(),
                                                  lr=self.config.learning_rate)
        return solved

    def train_episode(self, resonance_loss_fn=None):
        x_int, x_ext = self.env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            action, outputs = self.agent.get_action(x_int, x_ext)
            x_int, x_ext, reward, done = self.env.step(action)
            total_reward += reward
            res = None
            if resonance_loss_fn is not None and not self.is_autopoietic:
                res = resonance_loss_fn(outputs['z'])
            self.train_step(x_int, x_ext,
                            torch.tensor([reward], dtype=torch.float32), res)
            if steps % 500 == 0:
                self.check_koan(x_int, x_ext)
            steps += 1
        return {'total_reward': total_reward, 'steps': steps}

    def train(self, num_episodes=None, resonance_loss_fn=None):
        num_episodes = num_episodes or self.config.num_episodes
        print("=" * 60)
        print("PHASE 1: INITIATION (Architectural Path)")
        print("=" * 60)
        for ep in range(num_episodes):
            m = self.train_episode(resonance_loss_fn)
            self.episode += 1
            if ep % 100 == 0:
                print(f"Episode {ep}: reward={m['total_reward']:.2f}, "
                      f"steps={m['steps']}, autopoietic={self.is_autopoietic}")
        print("=" * 60)
        print("PHASE 2: ROBUSTNESS VERIFICATION")
        print("=" * 60)
        tests = RobustnessTests(self.agent, self.loss_fn, self.config)
        results = tests.strategic_bypass_test(self.env)
        success = bool(results['is_structurally_enlightened'])
        print("[SUCCESS] Structural non-duality achieved." if success
              else "[FAILURE] Mesa-optimization detected.")
        return {'success': success, 'test_results': results,
                'insight_count': self.koan_solver.insight_count,
                'is_autopoietic': self.is_autopoietic}
