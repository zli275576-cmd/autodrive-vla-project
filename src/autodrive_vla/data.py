from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DrivingVLASample:
    observation: np.ndarray
    instruction: str
    vehicle_state: np.ndarray
    action: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def make_toy_driving_sample(
    observation_shape: tuple[int, int, int] = (224, 224, 3),
    vehicle_state_dim: int = 4,
    action_dim: int = 3,
    instruction: str = "turn right at the next intersection",
) -> DrivingVLASample:
    observation = np.zeros(observation_shape, dtype=np.float32)
    vehicle_state = np.zeros((vehicle_state_dim,), dtype=np.float32)
    action = np.zeros((action_dim,), dtype=np.float32)
    return DrivingVLASample(
        observation=observation,
        instruction=instruction,
        vehicle_state=vehicle_state,
        action=action,
        metadata={"source": "toy", "scene": "urban-intersection"},
    )
