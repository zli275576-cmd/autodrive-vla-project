import numpy as np

from autodrive_vla.config import ExperimentConfig
from autodrive_vla.data import make_toy_driving_sample
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
