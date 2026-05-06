from typing import Any

from torch import nn, Tensor

class CausalConv(nn.Module):
    def __init__(self, kernel_size, padding, stride, dilation) -> None:
        ...
        
    def forward(self, input: Tensor) -> Tensor:
        ...


class ResUnit(nn.Module):
    def __init__(self, in_channels: int, n: int, dilation: int) -> None:
        super().__init__(self)

        # self.conv0 = 

class EncoderBlock(nn.Module):
    def __init__(self, n: int, s: int) -> None:
        super().__init__(self)



class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__(self)
