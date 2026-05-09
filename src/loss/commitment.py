from torch import Tensor, nn


class RVQCommitmentLoss(nn.Module):
    """
    Trains the encoder to produce embeddings that are close to the codebook
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, z, z_hat_d) -> Tensor:
        """
        [z: (B, N), z_hat_d: (B..., D, N)] -> loss: (B...,)
        """

        # # This is the commitment loss from the RVQ paper which is probably wrong
        # # Need to check if this is worse than just simply doing || Z_hat - Z ||_2 -> min

        # diff = z[:, None, :] - z_hat_d.detach() # -> (B, D, N)

        # return (diff ** 2).sum(dim=-1).sum(dim=-1) # -> (B,)

        return ((z - z_hat_d[..., -1, :].detach()) ** 2).sum(dim=-1).mean()
