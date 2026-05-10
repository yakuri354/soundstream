import torch
from torchmetrics.audio import ShortTimeObjectiveIntelligibility

from .base_metric import BaseMetric


class STOIMetric(BaseMetric):
    def __init__(self, sample_rate: int, extended: bool = False, name=None, **kwargs):
        super().__init__(name, **kwargs)
        self.sample_rate = sample_rate

        self.stoi = ShortTimeObjectiveIntelligibility(self.sample_rate, extended)

    @torch.no_grad()
    def forward(self, batch):
        res = torch.stack(
            [
                self.stoi(preds[0, : orig_size.item()], input[0, : orig_size.item()])
                for preds, input, orig_size in zip(
                    batch["decoded"], batch["input"], batch["input_original_len"]
                )
            ]
        ).mean(dim=0)

        return {self.name: res.item()}

        # return {
        #     self.name: float(
        #         self.stoi(
        #             batch["decoded"][:, 0, :].detach().cpu(),
        #             batch["input"][:, 0, :].detach().cpu(),
        #         )
        #     )
        # }
