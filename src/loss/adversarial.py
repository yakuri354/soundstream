import torch
from torch import Tensor

from src.loss.ssloss import SSLoss


class DiscriminatorLoss(SSLoss):
    """
    Trains the discriminator to discern real and generated images
    """
    def compute(
        self, x: Tensor, G: Tensor, D_x: list[list[Tensor]], D_G: list[list[Tensor]]
    ):
        real = torch.stack(
            [(1 - D_k_x[-1]).clamp(min=0.0).mean() for D_k_x in D_x]
        ).mean()
        gen = torch.stack(
            [(1 + D_k_G[-1]).clamp(min=0.0).mean() for D_k_G in D_G]
        ).mean()

        return real + gen


class GeneratorAdvLoss(SSLoss):
    """
    Trains the generator to mislead the discriminator
    """
    def compute(
        self, x: Tensor, G: Tensor, D_x: list[list[Tensor]], D_G: list[list[Tensor]]
    ):
        return torch.stack(
            [(1 - D_k_G[-1]).clamp(min=0.0).mean() for D_k_G in D_G]
        ).mean()


class FeatureLoss(SSLoss):
    """
    Trains the generator to produce images that are similar in discriminator latent space
    """
    def compute(
        self, x: Tensor, G: Tensor, D_x: list[list[Tensor]], D_G: list[list[Tensor]]
    ):
        return torch.stack(
            [
                torch.stack(
                    [
                        torch.abs(D_k_l_x - D_k_l_G).mean()
                        for (D_k_l_x, D_k_l_G) in zip(D_k_x, D_k_G)
                    ]
                ).mean()
                for (D_k_x, D_k_G) in zip(D_x, D_G)
            ]
        ).mean()


