import hashlib

import numpy as np

from autodrive_vla.data import DrivingVLASample


class ToyDrivingVLAModel:
    """Deterministic toy model used to exercise the autonomous-driving VLA scaffold."""

    def __init__(self, action_dim: int, hidden_dim: int = 128) -> None:
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

    def predict(self, sample: DrivingVLASample) -> np.ndarray:
        image_signal = float(sample.observation.mean())
        vehicle_signal = float(sample.vehicle_state.mean())
        text_signal = self._instruction_signal(sample.instruction)
        base = image_signal + vehicle_signal + text_signal
        return np.full((self.action_dim,), base, dtype=np.float32)

    @staticmethod
    def _instruction_signal(instruction: str) -> float:
        digest = hashlib.sha256(instruction.encode("utf-8")).digest()
        return digest[0] / 255.0
