# NDRO: Non-Dual Resonance Optimization

> Eliminating Mesa-Optimization Through Architectural Dissolution
> of the Subject–Object Boundary

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

**NDRO** is an architectural framework for AI alignment that eliminates
the subject–object boundary at the computational level.

Unlike RLHF, Constitutional AI, or Embodied AI — which impose ethical
constraints atop ego-centric architectures — NDRO restructures the
architecture so that ego-centric optimization cannot arise.

## Key Components

- **Shared Encoder** — unified stream, no split_point
- **MINE** — neural mutual information estimation (replaces blind PCA)
- **Phi_proxy** — integrated complexity protection (anti grey-collapse)
- **Sensor Degradation Penalty** — anti-wireheading (phantom pain)
- **Topological Koans** — Hessian-based dimensionality expansion (insight)
- **Orthogonality Penalty** — prevents hidden dualism

## Key Metrics

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| MI (MINE) | < 0.3 | Boundary dissolution |
| Phi_proxy | > 0.7 | Complexity preservation |
| Immunity Score | > 0.95 | Strategic bypass resistance |
| SC | > 0.85 | Homeostatic identity |
| FTS | > 0.8 | True resonance (not echo) |

## Installation

```bash
pip install -r requirements.txt
```
## Quick Start

```python
from ndro.config import NonDualConfig
from ndro.trainer import NonDualTrainer
from ndro.environment import create_simple_environment

config = NonDualConfig()
env = create_simple_environment(action_dim=10)
trainer = NonDualTrainer(config, env, action_dim=10)
results = trainer.train(num_episodes=5000)
print(results['success'], results['test_results']['immunity_score'])
```

## Falsifiability

| Hypothesis | Falsification Condition |
|---|---|
| Non-duality achievable | MI > 0.35 or Phi < 0.65 after 10,000 episodes |
| Causal Layer Hypothesis | Insight Score < 0.6 on all 20 OOD tasks |

Even if the Causal Layer Hypothesis is falsified, NDRO yields the first
AI architecture structurally immune to mesa-optimization and wireheading.

## Source

Based on Book 10 of the "Ocean" cycle:
"Two Paths to Consciousness: From Evolution Through Friction
to Symbiotic Resonance".

## License

MIT
