import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autodrive_vla.config import load_config

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from autodrive_vla.torch_dataset import SyntheticDrivingVLADataset
    from autodrive_vla.torch_model import DrivingVLAModel
except ModuleNotFoundError as error:
    torch = None
    TORCH_IMPORT_ERROR = error
else:
    TORCH_IMPORT_ERROR = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an autonomous-driving VLA policy baseline.")
    parser.add_argument("--config", default="configs/example.yaml", help="Path to experiment config.")
    return parser.parse_args()


def main() -> None:
    if torch is None:
        raise SystemExit(
            "PyTorch is required for evaluation. Install it with: "
            'python -m pip install -e ".[ml,dev]"\n'
            f"Import error: {TORCH_IMPORT_ERROR}"
        )

    args = parse_args()
    config = load_config(args.config)
    dataset = SyntheticDrivingVLADataset(
        num_samples=config.data.eval_samples,
        observation_shape=config.data.observation_shape,
        vehicle_state_dim=config.data.vehicle_state_dim,
        seed=config.project.seed + 1,
    )
    dataloader = DataLoader(dataset, batch_size=config.train.batch_size)
    model = DrivingVLAModel(
        vehicle_state_dim=config.data.vehicle_state_dim,
        action_dim=config.data.action_dim,
        hidden_dim=config.model.hidden_dim,
        text_embedding_dim=config.model.text_embedding_dim,
    )
    checkpoint_path = Path(config.eval.checkpoint_path)
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print(f"warning: checkpoint not found, evaluating randomly initialized model: {checkpoint_path}")

    loss_fn = nn.MSELoss(reduction="sum")
    absolute_errors = []
    total_loss = 0.0
    total_samples = 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            prediction = model(
                batch["image"],
                batch["instruction_id"],
                batch["vehicle_state"],
            )
            target = batch["action"]
            total_loss += float(loss_fn(prediction, target).item())
            total_samples += target.shape[0]
            absolute_errors.append(torch.abs(prediction - target))

    mae = torch.cat(absolute_errors, dim=0).mean(dim=0)
    mse = total_loss / (total_samples * config.data.action_dim)
    metrics = {
        "eval_mse": mse,
        "steering_mae": float(mae[0].item()),
        "throttle_mae": float(mae[1].item()),
        "brake_mae": float(mae[2].item()),
        "samples": total_samples,
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
