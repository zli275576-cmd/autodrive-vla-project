from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    name: str = "autodrive-vla"
    seed: int = 42


class DataConfig(BaseModel):
    dataset_name: str = "toy-driving-vla"
    split: str = "train"
    observation_shape: tuple[int, int, int] = (64, 64, 3)
    vehicle_state_dim: int = 4
    action_dim: int = 3
    train_samples: int = 512
    eval_samples: int = 128


class ModelConfig(BaseModel):
    name: str = "driving-vla-baseline"
    hidden_dim: int = 128
    text_embedding_dim: int = 32


class TrainConfig(BaseModel):
    batch_size: int = 32
    learning_rate: float = 1e-3
    max_steps: int = 200
    checkpoint_path: str = "outputs/checkpoints/driving_vla_baseline.pt"


class EvalConfig(BaseModel):
    episodes: int = 128
    checkpoint_path: str = "outputs/checkpoints/driving_vla_baseline.pt"


class ExperimentConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    return ExperimentConfig.model_validate(payload)
