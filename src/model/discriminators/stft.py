import torch
from torch import Tensor
import torch.nn as nn


class STFTResidualUnit(nn.Module):
    def __init__(self, n: int, m: int, s: tuple[int, int]):
        super().__init__()

        st, sf = s

        self.conv0 = nn.Conv2d(
            kernel_size=3, in_channels=n, out_channels=n, padding="same"
        )
        self.act = nn.ELU()

        self.conv1 = nn.Conv2d(
            kernel_size=(st + 2, sf + 2),
            in_channels=n,
            out_channels=n * m,
            stride=s,
            padding=(1, 1),
        )

        self.res_conv = nn.Conv2d(
            kernel_size=s, in_channels=n, out_channels=n * m, stride=s
        )

    def forward(self, x: Tensor):
        y = self.conv0(x)
        y = self.act(y)
        y = self.conv1(y)

        return y + self.res_conv(x)


class STFTDiscriminator(nn.Module):
    def __init__(self, c: int = 32, w: int = 1024, h: int = 256):
        super().__init__()

        self.c = c
        self.w = w
        self.h = h

        self.act = nn.ELU()

        self.conv0 = nn.Conv2d(
            in_channels=2, out_channels=c, kernel_size=7, padding="same"
        )

        self.units = nn.ModuleList(
            [
                STFTResidualUnit(n=c, m=2, s=(1, 2)),
                STFTResidualUnit(n=2 * c, m=2, s=(2, 2)),
                STFTResidualUnit(n=4 * c, m=1, s=(1, 2)),
                STFTResidualUnit(n=4 * c, m=2, s=(2, 2)),
                STFTResidualUnit(n=8 * c, m=1, s=(1, 2)),
                STFTResidualUnit(n=8 * c, m=2, s=(2, 2)),
            ]
        )

        f = self.w // 2 + 1

        self.conv1 = nn.Conv2d(
            in_channels=16 * c,
            out_channels=1,
            kernel_size=(1, f // 2**6),
            padding="valid",
        )

    def forward(self, x: Tensor) -> list[Tensor]:
        """
        Expects x: waveform with shape = (B, T)
        """

        activations = []

        f = torch.stft(
            x,
            n_fft=self.w,
            hop_length=self.h,
            window=torch.hann_window(self.w),
            return_complex=True,
        )  # (B, W // 2 + 1, T // H + 1) = (B, F, T')

        f = torch.view_as_real(f)  # (B, F, T', 2)
        f = f.permute(0, 3, 2, 1)  # (B, 2, T', F)

        f = self.conv0(f)
        # sus -- do we need to apply an activation here?

        activations.append(f.transpose(-1, -2))

        for unit in self.units:
            f = unit(f)
            activations.append(f.transpose(-1, -2))

        f = self.conv1(f)
        activations.append(f.transpose(-1, -2))

        return activations
