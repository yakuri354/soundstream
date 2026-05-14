from torch import Tensor, nn

from .components import *


class DecoderBlock(nn.Module):
    def __init__(self, n: int, s: int):
        super().__init__()
        self.conv0 = CausalConvT(n * 2, n, kernel_size=s * 2, stride=s)
        self.convs = nn.ModuleList([ResUnit(n, n, dilation) for dilation in DILATIONS])

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv0(x)

        for conv in self.convs:
            y = conv(y)

        return y


class Decoder(nn.Module):
    def __init__(self, c: int, k: int, strides: list[int]) -> None:
        super().__init__()

        self.conv0 = CausalConv(
            in_channels=k, out_channels=2 ** len(strides) * c, kernel_size=7
        )

        self.blocks = nn.ModuleList(
            [DecoderBlock(n=2**i * c, s=s) for i, s in list(enumerate(strides))[::-1]]
        )

        self.conv1 = CausalConv(in_channels=c, out_channels=1, kernel_size=7)

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv0(x)

        for block in self.blocks:
            y = block(y)

        return self.conv1(y)
