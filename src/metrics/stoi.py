import torch
from torchmetrics.audio import ShortTimeObjectiveIntelligibility

from .base_metric import BaseMetric


class STOIMetric(BaseMetric):
    def __init__(self, sample_rate: int, extended: bool = False, name=None):
        super().__init__(name)
        self.sample_rate = sample_rate

        self.stoi = ShortTimeObjectiveIntelligibility(self.sample_rate, extended)

    @torch.no_grad()
    def forward(self, batch):
        return {
            self.name: float(
                self.stoi(
                    batch["decoded"][:, 0, :].detach().cpu(),
                    batch["input"][:, 0, :].detach().cpu(),
                )
            )
        }
