import numpy as np
import pytest

from autodrive_vla.config import ExperimentConfig
from autodrive_vla.data import DRIVING_INSTRUCTIONS, make_synthetic_driving_sample, make_toy_driving_sample
from autodrive_vla.model import ToyDrivingVLAModel
from autodrive_vla.policy import DrivingVLAPolicy


def test_default_config_loads() -> None:
    config = ExperimentConfig()
    assert config.data.action_dim == 3


def test_policy_returns_action_vector() -> None:
    sample = make_toy_driving_sample(action_dim=3)
    policy = DrivingVLAPolicy(ToyDrivingVLAModel(action_dim=3))
    action = policy.act(sample)
    assert isinstance(action, np.ndarray)
    assert action.shape == (3,)


def test_synthetic_sample_has_driving_fields() -> None:
    rng = np.random.default_rng(42)
    sample = make_synthetic_driving_sample(rng)
    assert sample.observation.shape == (64, 64, 3)
    assert sample.vehicle_state.shape == (4,)
    assert sample.action.shape == (3,)
    assert sample.instruction in DRIVING_INSTRUCTIONS


def test_torch_model_forward_if_available() -> None:
    torch = pytest.importorskip("torch")
    from autodrive_vla.torch_model import DrivingVLAModel

    model = DrivingVLAModel()
    image = torch.zeros((2, 3, 64, 64), dtype=torch.float32)
    instruction_id = torch.zeros((2,), dtype=torch.long)
    vehicle_state = torch.zeros((2, 4), dtype=torch.float32)
    action = model(image, instruction_id, vehicle_state)
    assert action.shape == (2, 3)
