from logging import debug

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_kmeans import KMeans


class MultiRVQ(nn.Module):
    def __init__(
        self,
        k: int,
        d: int,
        emb_dim: int,
        gamma: float = 0.99,
        freq_threshold: float = 2.0,
        init_kmeans: bool = False,
    ) -> None:
        super().__init__()

        self.k = k  # codebook size for each quantizer
        self.d = d  # depth, i.e. number of quantizers
        self.n = emb_dim
        self.codes = nn.Parameter(torch.empty((d, k, emb_dim)), requires_grad=False)

        self.freq = nn.Buffer(torch.empty((d, k)), persistent=True)
        self.sums = nn.Buffer(torch.empty((d, k, emb_dim)), persistent=True)
        self.gamma = gamma
        self.freq_threshold = freq_threshold
        self.init_kmeans = init_kmeans

        self.is_init = False

    @torch.no_grad()
    def initialize(self, z: Tensor) -> None:
        """
        Initializes the embeddings via k-means on the given batch

        z: (B..., N)
        """

        r = z

        debug("Initializing RVQ")

        for level in range(self.d):
            if self.init_kmeans:
                kmeans = KMeans(n_clusters=self.k, verbose=False)
                self.codes[level] = kmeans(r.reshape(1, -1, self.n)).centers[0, ...]
            else:
                r_flat = r.reshape(-1, self.n)

                B_total, _ = r_flat.shape

                samples = torch.randint(0, B_total, (self.k,), device=z.device)  # (K,)
                self.codes[level].copy_(r_flat[samples])

            e_k, _ = self.quantize(r, level)
            r = r - e_k

        self.sums.copy_(self.codes * self.freq_threshold)
        self.freq.fill_(self.freq_threshold)
        self.is_init = True

    @torch.no_grad()
    def retire_dead_codes(self, level: int, z: Tensor) -> None:
        mask = self.freq[level, :] < self.freq_threshold
        z_flat = z.reshape(-1, self.n)

        B_total, _ = z_flat.shape

        samples = torch.randint(0, B_total, (self.k,), device=z.device)  # (K,)
        self.codes[level, mask] = z_flat[samples][mask]
        self.freq[level, mask] = self.freq_threshold

        mask_size = mask.sum().int().item()

        if mask_size > 0:
            debug(f"Replaced {mask_size} dead codes")

    @torch.no_grad()
    def update_codebook(self, level: int, k: Tensor, z: Tensor) -> None:
        assert self.training

        one_hot = F.one_hot(k, self.k).reshape(-1, self.k).float()  # (B', K)

        batch_freq = one_hot.sum(dim=0)
        self.freq[level] = self.freq[level] * self.gamma + batch_freq * (1 - self.gamma)

        batch_sum = one_hot.T @ z.reshape(-1, self.n)  # (K, B') @ (B', N) -> (K, N)
        self.sums[level] = self.sums[level] * self.gamma + batch_sum * (1 - self.gamma)

        self.codes[level] = self.sums[level] / self.freq[level][..., None]

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

        z_hat = self.codes[level, k] - z.detach() + z

        if update:
            self.update_codebook(level, k, z)

        return z_hat, k

    def forward(self, z: Tensor, update: bool = False) -> tuple[Tensor, Tensor]:
        """
        z: (B..., N)
        returns [(B..., D, N), (B..., D)] i.e. prefix sums and indices
        """

        assert self.is_init

        z_hat = torch.zeros_like(z)
        z_hat_d = torch.tensor((), device=z.device)

        k_d = torch.tensor((), device=z.device)
        r = z

        for level in range(self.d):
            e_k, k = self.quantize(r, level, update)
            r = r - e_k

            z_hat = z_hat + e_k
            z_hat_d = torch.cat([z_hat_d, z_hat[..., None, :]], dim=-2)

            k_d = torch.cat([k_d, k[..., None]], dim=-1)

        return z_hat_d, k_d
