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
    parser = argparse.ArgumentParser(description="Train an autonomous-driving VLA policy baseline.")
    parser.add_argument("--config", default="configs/example.yaml", help="Path to experiment config.")
    return parser.parse_args()


def main() -> None:
    if torch is None:
        raise SystemExit(
            "PyTorch is required for training. Install it with: "
            'python -m pip install -e ".[ml,dev]"\n'
            f"Import error: {TORCH_IMPORT_ERROR}"
        )

    args = parse_args()
    config = load_config(args.config)

    torch.manual_seed(config.project.seed)
    dataset = SyntheticDrivingVLADataset(
        num_samples=config.data.train_samples,
        observation_shape=config.data.observation_shape,
        vehicle_state_dim=config.data.vehicle_state_dim,
        seed=config.project.seed,
    )
    dataloader = DataLoader(dataset, batch_size=config.train.batch_size, shuffle=True)
    model = DrivingVLAModel(
        vehicle_state_dim=config.data.vehicle_state_dim,
        action_dim=config.data.action_dim,
        hidden_dim=config.model.hidden_dim,
        text_embedding_dim=config.model.text_embedding_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)
    loss_fn = nn.MSELoss()

    step = 0
    losses: list[float] = []
    model.train()
    while step < config.train.max_steps:
        for batch in dataloader:
            prediction = model(
                batch["image"],
                batch["instruction_id"],
                batch["vehicle_state"],
            )
            loss = loss_fn(prediction, batch["action"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            losses.append(float(loss.item()))
            if step % 25 == 0 or step == 1:
                print(f"step={step:04d} train_mse={losses[-1]:.6f}")
            if step >= config.train.max_steps:
                break

    checkpoint_path = Path(config.train.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.model_dump(),
            "final_train_mse": losses[-1],
        },
        checkpoint_path,
    )
    metrics_path = checkpoint_path.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "final_train_mse": losses[-1],
                "best_train_mse": min(losses),
                "steps": step,
                "dataset": config.data.dataset_name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved checkpoint: {checkpoint_path}")
    print(f"saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
