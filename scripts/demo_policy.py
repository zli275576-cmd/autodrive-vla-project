import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autodrive_vla.config import load_config
from autodrive_vla.data import make_toy_driving_sample
from autodrive_vla.model import ToyDrivingVLAModel
from autodrive_vla.policy import DrivingVLAPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a toy autonomous-driving VLA policy demo.")
    parser.add_argument("--config", default="configs/example.yaml", help="Path to experiment config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    sample = make_toy_driving_sample(
        observation_shape=config.data.observation_shape,
        vehicle_state_dim=config.data.vehicle_state_dim,
        action_dim=config.data.action_dim,
    )
    model = ToyDrivingVLAModel(
        action_dim=config.data.action_dim,
        hidden_dim=config.model.hidden_dim,
    )
    policy = DrivingVLAPolicy(model)
    action = policy.act(sample)
    print(f"instruction: {sample.instruction}")
    print(f"vehicle_state: {sample.vehicle_state.tolist()}")
    print(f"action [steering, throttle, brake]: {action.tolist()}")


if __name__ == "__main__":
    main()
