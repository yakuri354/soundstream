from typing import Any

from torch import nn, Tensor
from torch_kmeans import KMeans
from logging import debug
import torch.nn.functional as F
import torch


class MultiRVQ(nn.Module):
    def __init__(
        self,
        K: int,
        d: int,
        emb_dim: int,
        gamma: float = 0.99,
        freq_threshold: float = 2.0,
    ) -> None:
        super().__init__()

        self.K = K  # codebook size for each quantizer
        self.d = d  # depth, i.e. number of quantizers
        self.N = emb_dim
        self.codes = nn.Parameter(torch.empty((d, K, emb_dim)))

        self.freq = nn.Buffer(torch.empty((d, K)), requires_grad=False)
        self.sums = nn.Buffer(torch.empty((d, K, emb_dim)), requires_grad=False)
        self.gamma = gamma
        self.freq_threshold = freq_threshold

    @torch.no_grad()
    def initialize(self, z: Tensor) -> None:
        """
        Initializes the embeddings via k-means on the given batch

        z: (B..., N)
        """

        r = z

        debug("Initializing RVQ via KMeans")

        for level in range(self.d):
            kmeans = KMeans(n_clusters=self.K, verbose=False)

            self.codes[level] = kmeans(r.reshape(-1, self.N)).centers  # sus

            e_k, _ = self.quantize(r, level)
            r = r - e_k

        self.sums = self.codes
        self.freq.fill_(self.freq_threshold)

    @torch.no_grad()
    def retire_dead_codes(self, level: int, z: Tensor) -> None:
        mask = self.freq[level, :] < self.freq_threshold
        z_flat = z.reshape(-1, self.N)

        B_total, _ = z_flat.shape

        samples = torch.randint(0, B_total, (self.K,), device=z.device)  # (K,)
        self.codes[level, mask] = z_flat[samples][mask]
        self.freq[level, mask] = self.freq_threshold  # sus?

        mask_size = mask.sum().int().item()

        if mask_size > 0:
            debug(f"Replaced {mask_size} dead codes")

    @torch.no_grad()
    def update_codebook(self, level: int, k: Tensor, z: Tensor) -> None:
        one_hot = F.one_hot(k, self.K).reshape(-1, self.K).float()  # (B', K)

        batch_freq = one_hot.sum(dim=0)
        self.freq[level] = self.freq[level] * self.gamma + batch_freq * (1 - self.gamma)

        batch_sum = one_hot.T @ z.reshape(-1, self.N)  # (K, B') @ (B', N) -> (K, N)
        self.sums[level] = self.sums[level] * self.gamma + batch_sum * (
            1 - self.gamma
        )  # sus: will this overflow?

        self.codes[level] = self.sums[level] / self.freq[level]

        self.retire_dead_codes(level, z)

    def quantize(
        self, z: Tensor, level: int, update: bool = False
    ) -> tuple[Tensor, Tensor]:
        """
        r: (B..., N) -> [e(k): (B..., N), k: (B...,)]
        finds the nearest neighbor of z and its index within a level
        """

        d = torch.cdist(
            z, self.codes[level]
        )  # cdist of (B..., N) and (K, N) -> (B..., K)
        k = torch.argmin(d, dim=-1)  # -> (B...,)

        if update:
            assert self.training

            self.update_codebook(level, k, z)

        return self.codes[level, k], k

    def forward(self, z: Tensor, update: bool = False) -> tuple[Tensor, Tensor]:
        """
        z: (B..., N)
        returns [(B..., D, N), (B..., D)] i.e. prefix sums and indices
        """

        z_hat = torch.zeros_like(z)
        z_hat_d = torch.empty((), device=z.device)

        k_d = torch.empty((), device=z.device)
        r = z

        for level in range(self.d):
            e_k, k = self.quantize(r, level, update)
            r = r - e_k

            z_hat = z_hat + e_k
            z_hat_d = torch.stack([z_hat_d, z_hat[..., None, :]], dim=-2)

            k_d = torch.stack([k_d, k[..., None]], dim=-2)

        return z_hat_d, k_d
