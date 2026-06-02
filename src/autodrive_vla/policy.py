import numpy as np

from autodrive_vla.data import DrivingVLASample
from autodrive_vla.model import ToyDrivingVLAModel


class DrivingVLAPolicy:
    def __init__(self, model: ToyDrivingVLAModel) -> None:
        self.model = model

    def act(self, sample: DrivingVLASample) -> np.ndarray:
        return self.model.predict(sample)
