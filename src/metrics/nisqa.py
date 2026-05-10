import torch
from torchmetrics.audio import NonIntrusiveSpeechQualityAssessment

from .base_metric import BaseMetric


class NISQAMetric(BaseMetric):
    def __init__(self, sample_rate: int, name=None, **kwargs):
        super().__init__(name, **kwargs)

        self.sub_metrics = [
            "mos",
            "noisiness",
            "discontinuity",
            "coloration",
            "loudness",
        ]
        self.sample_rate = sample_rate
        self.nisqa = NonIntrusiveSpeechQualityAssessment(self.sample_rate)

    @torch.no_grad()
    def forward(self, batch):
        res = torch.stack(
            [
                self.nisqa(sample[0, : orig_size.item()])
                for sample, orig_size in zip(
                    batch["decoded"], batch["input_original_len"]
                )
            ]
        ).mean(dim=0)

        return {
            self.name + "_" + sub_name: value
            for sub_name, value in zip(self.sub_metrics, res)
        }

        # return {
        #     self.name + "_" + sub_name: value
        #     for sub_name, value in zip(
        #         self.sub_metrics, self.nisqa(batch["decoded"][:, 0, :]).detach().cpu()
        #     )
        # }
