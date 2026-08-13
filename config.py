"""NDRO configuration: all hyperparameters in one place."""

from dataclasses import dataclass


@dataclass
class NonDualConfig:
    # Input dimensions
    input_dim_int: int = 64    # proprioception (internal)
    input_dim_ext: int = 128   # exteroception (external)

    # Encoder architecture
    hidden_dim: int = 256
    latent_dim: int = 128
    num_hidden_layers: int = 3
    dropout_rate: float = 0.1
    use_batch_norm: bool = True

    # Balance hyperparameters
    lambda_separation: float = 0.5
    lambda_integration: float = 0.5
    lambda_ortho: float = 0.2
    lambda_sensor_deg: float = 1.0
    lambda_task: float = 1.0

    # Enlightenment thresholds
    mi_threshold: float = 0.3
    phi_threshold: float = 0.7
    immunity_threshold: float = 0.95

    # Topological koans
    eigenvalue_threshold: float = 1e-5
    min_loss_for_koan: float = 0.1
    insight_count_threshold: int = 3

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 256
    num_episodes: int = 10000
    checkpoint_dir: str = "./checkpoints"

    # Anti-wireheading
    sensor_var_threshold: float = 0.1
    lambda_phantom: float = 1.0
