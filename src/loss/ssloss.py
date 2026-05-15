import torch
import torch.nn as nn
from torch import Tensor


class SSLoss(nn.Module):
    def __init__(self, n_disc: int = 4):
        super().__init__()
        self.n_disc = n_disc

    def compute(
        self, x: Tensor, G: Tensor, D_x: list[list[Tensor]], D_G: list[list[Tensor]]
    ) -> Tensor:
        raise NotImplementedError

    def forward(
        self, x: Tensor, G: Tensor, D_x: list[list[Tensor]], D_G: list[list[Tensor]]
    ) -> Tensor:
        assert G.shape == x.shape
        assert len(D_x) == self.n_disc
        assert len(D_G) == self.n_disc

        return self.compute(x, G, D_x, D_G)
