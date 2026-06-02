from dataclasses import dataclass, field
from typing import Any

import numpy as np

DRIVING_INSTRUCTIONS = [
    "keep lane",
    "turn left",
    "turn right",
    "slow down",
    "stop",
    "change lane left",
    "change lane right",
]


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


def make_synthetic_driving_sample(
    rng: np.random.Generator,
    observation_shape: tuple[int, int, int] = (64, 64, 3),
    vehicle_state_dim: int = 4,
) -> DrivingVLASample:
    instruction_id = int(rng.integers(0, len(DRIVING_INSTRUCTIONS)))
    instruction = DRIVING_INSTRUCTIONS[instruction_id]
    observation = rng.normal(0.0, 0.08, observation_shape).astype(np.float32)
    vehicle_state = rng.uniform(-1.0, 1.0, vehicle_state_dim).astype(np.float32)

    steering_hint = _instruction_to_steering(instruction)
    speed = vehicle_state[0]
    lane_offset = vehicle_state[1] if vehicle_state_dim > 1 else 0.0
    lead_distance = vehicle_state[2] if vehicle_state_dim > 2 else 0.0

    steering = np.clip(steering_hint - 0.25 * lane_offset, -1.0, 1.0)
    throttle = np.clip(0.45 - 0.20 * speed + 0.10 * lead_distance, 0.0, 1.0)
    brake = 0.0
    if instruction in {"slow down", "stop"}:
        throttle = 0.05 if instruction == "stop" else 0.15
        brake = 0.85 if instruction == "stop" else 0.45

    action = np.asarray([steering, throttle, brake], dtype=np.float32)
    return DrivingVLASample(
        observation=observation,
        instruction=instruction,
        vehicle_state=vehicle_state,
        action=action,
        metadata={"source": "synthetic", "instruction_id": instruction_id},
    )


def _instruction_to_steering(instruction: str) -> float:
    if instruction in {"turn left", "change lane left"}:
        return -0.55
    if instruction in {"turn right", "change lane right"}:
        return 0.55
    return 0.0
