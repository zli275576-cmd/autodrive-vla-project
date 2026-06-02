from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    name: str = "autodrive-vla"
    seed: int = 42


class DataConfig(BaseModel):
    dataset_name: str = "toy-driving-vla"
    split: str = "train"
    observation_shape: tuple[int, int, int] = (224, 224, 3)
    vehicle_state_dim: int = 4
    action_dim: int = 3


class ModelConfig(BaseModel):
    name: str = "toy-driving-vla-model"
    hidden_dim: int = 128


class TrainConfig(BaseModel):
    batch_size: int = 8
    learning_rate: float = 1e-4
    max_steps: int = 100


class EvalConfig(BaseModel):
    episodes: int = 5


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
