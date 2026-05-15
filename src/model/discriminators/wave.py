from typing import Any

import torch.nn as nn
from torch import Tensor
from torch.nn.utils.parametrizations import weight_norm


class WaveDiscriminator(nn.Module):
    def __init__(
        self,
        factor: int = 4,
        max_chans: int = 1024,
        layers: int = 4,
        lrelu_slope: float = 0.2,
    ):
        super().__init__()

        self.act = nn.LeakyReLU(lrelu_slope)

        self.conv0 = weight_norm(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=15, padding="same")
        )

        self.middle = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        in_channels=min(16 * factor**i, max_chans),
                        out_channels=min(16 * factor ** (i + 1), max_chans),
                        kernel_size=41,
                        stride=factor,
                        padding=20,
                        groups=factor ** (i + 1),
                    )
                )
                for i in range(layers)
            ]
        )

        final_dim = min(16 * factor**layers, max_chans)

        self.conv1 = weight_norm(
            nn.Conv1d(
                in_channels=final_dim,
                out_channels=final_dim,
                kernel_size=5,
                padding="same",
            )
        )

        self.conv2 = weight_norm(
            nn.Conv1d(
                in_channels=final_dim, out_channels=1, kernel_size=3, padding="same"
            )
        )

    def forward(self, x: Tensor) -> list[Tensor]:
        activations = []

        y = x[:, None, :]  # (B, 1, T)
        y = self.conv0(y)
        y = self.act(y)
        activations.append(y)

        for layer in self.middle:
            y = layer(y)
            y = self.act(y)
            activations.append(y)

        y = self.conv1(y)
        activations.append(y)

        y = self.conv2(y)
        activations.append(y)

        return activations


class MultiResWaveDiscriminator(nn.Module):
    def __init__(self, layers: int = 3, *args, **kwargs) -> None:
        super().__init__()

        self.layers = layers

        self.models = nn.ModuleList(
            [WaveDiscriminator(*args, **kwargs) for _ in range(self.layers)]
        )

        self.downsample = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)

    def forward(self, x: Tensor) -> list[list[Tensor]]:
        results = []

        for model in self.models:
            results.append(model(x))
            x = self.downsample(x)

        return results
