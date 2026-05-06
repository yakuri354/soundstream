import torch
from torch import Tensor, nn
from torchaudio.transforms import MelSpectrogram

from src.loss.ssloss import SSLoss


class AdversarialLoss(SSLoss):
    def __init__(
        self, sample_rate: int, s_min: int = 6, s_max: int = 11, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.sample_rate = sample_rate

        self.s_values = list(range(s_min, s_max + 1))

        self.mels = nn.ModuleList(
            [
                MelSpectrogram(
                    sample_rate=sample_rate, n_fft=s, hop_length=s // 4, n_mels=64
                )
                for s in self.s_values
            ]
        )

        self.alphas = [(s / 2) ** 0.5 for s in self.s_values]

    def compute_one(
        self, mel: MelSpectrogram, alpha: float, x: Tensor, G: Tensor
    ) -> Tensor:
        S_x = mel(x).transpose(-1, -2)  # (B, T', Mels)
        S_G_x = mel(G).transpose(-1, -2)  # (B, T', Mels)

        l1 = torch.abs(S_G_x - S_x).mean(dim=-1).sum(dim=-1)
        l2_log = (
            ((torch.log(S_G_x + 1e-12) - torch.log(S_x + 1e-12)) ** 2).mean(dim=-1)
            ** 0.5
        ).sum(dim=-1)

        return l1 + alpha * l2_log  # (B,)

    def compute(
        self, x: Tensor, G: Tensor, D_x: list[list[Tensor]], D_G: list[list[Tensor]]
    ):
        return torch.stack(
            [
                self.compute_one(mel, alpha, x, G) # type: ignore
                for mel, alpha in zip(self.mels, self.alphas)
            ]
        ).mean()
