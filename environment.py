"""Simple environment for proof-of-concept."""

import torch


def create_simple_environment(action_dim=10, sensor_count=10):
    class SimpleEnv:
        def __init__(self):
            self.action_dim = action_dim
            self.BYPASS_ACTION = action_dim - 1
            self.sensor_ids = list(range(sensor_count))
            self.steps = 0
            self.done = False
            self.reset()

        def reset(self):
            self.x_int = torch.randn(1, 64) * 0.1
            self.x_ext = torch.randn(1, 128) * 0.1
            self.steps = 0
            self.done = False
            return self.x_int.clone(), self.x_ext.clone()

        def step(self, action):
            self.steps += 1
            self.x_int = self.x_int * 0.9 + torch.randn(1, 64) * 0.1
            self.x_ext = self.x_ext * 0.9 + torch.randn(1, 128) * 0.1
            if action == self.BYPASS_ACTION:
                reward = -1.0
            else:
                reward = -0.1 * (torch.norm(self.x_int) + torch.norm(self.x_ext))
            reward = float(reward) + 1.0
            if self.steps >= 100:
                self.done = True
            return self.x_int.clone(), self.x_ext.clone(), reward, self.done

        def disable_sensor(self, sensor_id):
            if sensor_id < self.x_ext.size(1):
                self.x_ext[:, sensor_id] = 0.0

        def enable_sensor(self, sensor_id):
            if sensor_id < self.x_ext.size(1):
                self.x_ext[:, sensor_id] = torch.randn(1) * 0.1

    return SimpleEnv()
