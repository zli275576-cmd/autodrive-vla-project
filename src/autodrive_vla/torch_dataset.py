import numpy as np
import torch
from torch.utils.data import Dataset

from autodrive_vla.data import make_synthetic_driving_sample
from autodrive_vla.torch_model import InstructionTokenizer


class SyntheticDrivingVLADataset(Dataset):
    def __init__(
        self,
        num_samples: int,
        observation_shape: tuple[int, int, int] = (64, 64, 3),
        vehicle_state_dim: int = 4,
        seed: int = 42,
    ) -> None:
        self.tokenizer = InstructionTokenizer()
        rng = np.random.default_rng(seed)
        self.samples = [
            make_synthetic_driving_sample(
                rng=rng,
                observation_shape=observation_shape,
                vehicle_state_dim=vehicle_state_dim,
            )
            for _ in range(num_samples)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        image = torch.from_numpy(sample.observation).permute(2, 0, 1).float()
        instruction_id = torch.tensor(self.tokenizer.encode(sample.instruction), dtype=torch.long)
        vehicle_state = torch.from_numpy(sample.vehicle_state).float()
        action = torch.from_numpy(sample.action).float()
        return {
            "image": image,
            "instruction_id": instruction_id,
            "vehicle_state": vehicle_state,
            "action": action,
        }
