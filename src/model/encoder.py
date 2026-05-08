from torch import nn, Tensor
from components import *

# sus Verify pad mode


class EncoderBlock(nn.Module):
    def __init__(self, n: int, s: int) -> None:
        super().__init__()
        in_channels = n // 2

        self.convs = nn.ModuleList(
            [ResUnit(in_channels, in_channels, dilation) for dilation in DILATIONS]
        )

        self.conv0 = CausalConv(in_channels, n, kernel_size=2 * s, stride=s)

    def forward(self, x: Tensor) -> Tensor:
        y = x

        for conv in self.convs:
            y = conv(y)

        return self.conv0(y)


class Encoder(nn.Module):
    def __init__(self, c: int, k: int) -> None:
        super().__init__()

        self.conv0 = CausalConv(in_channels=1, out_channels=c, kernel_size=7)

        self.blocks = nn.ModuleList(
            [EncoderBlock(n=2 ** (i + 1) * c, s=s) for i, s in enumerate(STRIDES)]
        )

        self.conv1 = CausalConv(
            in_channels=2 ** len(STRIDES) * c, out_channels=k, kernel_size=3
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv0(x)

        for block in self.blocks:
            y = block(y)

        y = self.conv1(y)
        return y
