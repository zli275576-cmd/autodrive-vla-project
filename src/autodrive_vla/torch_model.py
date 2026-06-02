import torch
from torch import nn

from autodrive_vla.data import DRIVING_INSTRUCTIONS


class InstructionTokenizer:
    def __init__(self, instructions: list[str] | None = None) -> None:
        vocabulary = instructions or DRIVING_INSTRUCTIONS
        self.token_to_id = {text: index for index, text in enumerate(vocabulary)}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, instruction: str) -> int:
        return self.token_to_id.get(instruction, 0)


class DrivingVLAModel(nn.Module):
    """Small multimodal policy: camera image + instruction + vehicle state -> driving action."""

    def __init__(
        self,
        vehicle_state_dim: int = 4,
        action_dim: int = 3,
        vocab_size: int = len(DRIVING_INSTRUCTIONS),
        hidden_dim: int = 128,
        text_embedding_dim: int = 32,
    ) -> None:
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, hidden_dim),
            nn.ReLU(),
        )
        self.text_encoder = nn.Sequential(
            nn.Embedding(vocab_size, text_embedding_dim),
            nn.Flatten(),
            nn.Linear(text_embedding_dim, hidden_dim),
            nn.ReLU(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(vehicle_state_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        image: torch.Tensor,
        instruction_id: torch.Tensor,
        vehicle_state: torch.Tensor,
    ) -> torch.Tensor:
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(instruction_id)
        state_features = self.state_encoder(vehicle_state)
        fused = torch.cat([image_features, text_features, state_features], dim=-1)
        return self.action_head(fused)
