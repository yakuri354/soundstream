from typing import Literal
from torch import nn, Tensor

STRIDES = [2, 4, 5, 8]
DILATIONS = [1, 3, 9]


class CausalConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        left_padding: int | Literal["same"] = "same",
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding="valid",
            dilation=dilation,
        )

        if left_padding == "same":
            left_padding = 1 + dilation * (kernel_size - 1) - stride

        self.pad = nn.ZeroPad1d((left_padding, 0))

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.pad(x))


class CausalConvT(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1
    ) -> None:
        super().__init__()

        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride)
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        last_dim = x.shape[-1] * self.stride

        return self.conv(x)[..., :last_dim]


class ResUnit(nn.Module):
    def __init__(self, in_channels: int, n: int, dilation: int) -> None:
        super().__init__()

        self.conv0 = CausalConv(
            in_channels, n, kernel_size=7, left_padding="same", dilation=dilation
        )
        self.conv1 = nn.Conv1d(n, n, kernel_size=1)

        self.act = nn.ELU()

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv0(x)
        y = self.act(y)
        y = self.conv1(y)

        return x + y
