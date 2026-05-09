import torch
from torch import Tensor, nn


class BatchRandomCropPad(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        min_duration: float = 0.5,
        max_duration: float = 0.7,
        is_eval: bool = False,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.min_samples = int(sample_rate * min_duration)
        self.max_duration = max_duration
        self.max_samples = int(sample_rate * max_duration)
        self.is_eval = is_eval

    def __call__(self, data: Tensor):
        if self.min_samples != self.max_samples:
            n_samples = int(
                torch.randint(self.min_samples, self.max_samples, ()).item()
            )
        else:
            n_samples = self.min_samples

        ext_factor = 1 + n_samples // data.shape[-1]

        if ext_factor != 1:
            data = data.repeat(1, 1, ext_factor)

        b, _, t = data.shape

        l_max = t - self.min_samples

        if self.is_eval:
            lefts = torch.zeros((b,), device=data.device)
        else:
            lefts = torch.randint(0, l_max, (b,), device=data.device)

        indices = (
            torch.arange(0, n_samples, device=data.device)
            .view((1, n_samples))
            .repeat((b, 1))
        )

        indices = (indices + lefts[:, None]) % t

        return torch.gather(data, 2, indices[:, None, :])


class InstanceRandomCropPad(BatchRandomCropPad):
    def __call__(self, data: Tensor):
        assert data.dim() == 2 and data.shape[0] == 1  # (1, T)

        return super().__call__(data[None, ...])[0, ...]
