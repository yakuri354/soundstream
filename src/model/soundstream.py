from itertools import chain
from typing import Iterable

from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn

from ..loss.adversarial import DiscriminatorLoss, FeatureLoss, GeneratorAdvLoss
from ..loss.commitment import RVQCommitmentLoss
from ..loss.reconstruction import ReconstructionLoss
from .decoder import Decoder
from .discriminators.stft import STFTDiscriminator
from .discriminators.wave import MultiResWaveDiscriminator
from .encoder import Encoder
from .quant import MultiRVQ


class SoundStreamBase(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        sample_rate: int = 16000,
        emb_dim: int = 512,
        enc_dec_c: int = 32,
        codebook_size: int = 1024,
        codebook_depth: int = 8,
        strides: list[int] = [2, 4, 5, 5],
        rec_loss_lambda: float = 1.0,
        commit_loss_lambda: float = 1.0,
        init_kmeans: bool = False,
        rvq_gamma: float = 0.99,
        rec_loss_s_min: int = 6,
        rec_loss_s_max: int = 11,
    ):
        super().__init__()

        self.sample_rate = sample_rate

        self.encoder = Encoder(c=enc_dec_c, k=emb_dim, strides=strides)
        self.decoder = Decoder(c=enc_dec_c, k=emb_dim, strides=strides)

        self.rvq = MultiRVQ(
            k=codebook_size,
            d=codebook_depth,
            emb_dim=emb_dim,
            gamma=rvq_gamma,
            init_kmeans=init_kmeans,
        )

        self.commitment_loss = RVQCommitmentLoss()
        self.rec_loss = ReconstructionLoss(
            sample_rate, s_min=rec_loss_s_min, s_max=rec_loss_s_max
        )

        self.rec_loss_lambda = rec_loss_lambda
        self.commit_loss_lambda = commit_loss_lambda

        self.is_init = False

    def parameter_group(self, key: str) -> Iterable:
        if key == "generator":
            return chain(
                self.encoder.parameters(),
                self.decoder.parameters(),
            )
        else:
            raise KeyError()

    def initialize(self, x: Tensor) -> None:
        enc = self.encoder(x).transpose(-1, -2)
        self.rvq.initialize(enc)
        self.is_init = True

    @classmethod
    def _from_pretrained(cls, *args, **kwargs):
        self = super()._from_pretrained(*args, **kwargs)
        self.is_init = True
        self.rvq.is_init = True
        return self

    def compute_loss(
        self, x: Tensor, g: Tensor, z: Tensor, z_hat_d: Tensor
    ) -> dict[str, Tensor]:
        rec = self.rec_loss(x, g)
        commit = self.commitment_loss(z, z_hat_d)

        return {
            "rec_loss": rec,
            "commit_loss": commit,
            "gen_loss": rec * self.rec_loss_lambda + commit * self.commit_loss_lambda,
        }

    def forward(
        self,
        input: Tensor,
        compute_loss: bool = False,
        update_codebook: bool = False,
        **kwargs,
    ) -> dict[str, Tensor]:
        enc = self.encoder(input).transpose(-1, -2)

        z_hat_d, k = self.rvq(enc, update=update_codebook)
        z_hat = z_hat_d[..., -1, :]

        dec = self.decoder(z_hat.transpose(-1, -2))

        loss = {}
        if compute_loss:
            loss = self.compute_loss(input, dec, enc, z_hat_d)

        return {"encoded": k, "decoded": dec, **loss}


class SoundStream(SoundStreamBase):
    def __init__(
        self,
        sample_rate: int = 16000,
        emb_dim: int = 512,
        enc_dec_c: int = 32,
        codebook_size: int = 1024,
        codebook_depth: int = 8,
        strides: list[int] = [2, 4, 5, 5],
        rec_loss_lambda: float = 1.0,
        commit_loss_lambda: float = 1.0,
        feat_loss_lambda: float = 100.0,
        adv_loss_lambda: float = 1.0,
        init_kmeans: bool = False,
        rvq_gamma: float = 0.99,
        rec_loss_s_min: int = 6,
        rec_loss_s_max: int = 11,
    ):
        super().__init__(
            sample_rate,
            emb_dim,
            enc_dec_c,
            codebook_size,
            codebook_depth,
            strides,
            rec_loss_lambda,
            commit_loss_lambda,
            init_kmeans,
            rvq_gamma,
            rec_loss_s_min,
            rec_loss_s_max,
        )

        self.stft_discr = STFTDiscriminator(enc_dec_c)
        self.wave_discr = MultiResWaveDiscriminator()

        self.feat_loss = FeatureLoss()
        self.gen_adv_loss = GeneratorAdvLoss()

        self.discr_loss = DiscriminatorLoss()

        self.feat_loss_lambda = feat_loss_lambda
        self.adv_loss_lambda = adv_loss_lambda

    def parameter_group(self, key: str) -> Iterable:
        if key == "discriminator":
            return chain(self.stft_discr.parameters(), self.wave_discr.parameters())
        else:
            return super().parameter_group(key)

    def discr_forward(self, x: Tensor) -> list[list[Tensor]]:
        assert x.shape[1] == 1

        return [self.stft_discr(x[:, 0, :])] + self.wave_discr(x[:, 0, :])

    def discr_compute_loss(
        self, x: Tensor, g: Tensor, D_x: list[list[Tensor]], D_g: list[list[Tensor]]
    ) -> dict[str, Tensor]:
        return {"discr_loss": self.discr_loss(x, g, D_x, D_g)}

    def compute_loss(
        self, x: Tensor, g: Tensor, z: Tensor, z_hat_d: Tensor
    ) -> dict[str, Tensor]:
        base_loss = super().compute_loss(x, g, z, z_hat_d)

        D_x = self.discr_forward(x)
        D_g = self.discr_forward(g)

        feat_loss = self.feat_loss(x, g, D_x, D_g)
        gen_adv_loss = self.gen_adv_loss(x, g, D_x, D_g)

        D_x_detached = self.discr_forward(x.detach())
        D_g_detached = self.discr_forward(g.detach())

        discr_loss = self.discr_compute_loss(
            x.detach(), g.detach(), D_x_detached, D_g_detached
        )

        base_loss["gen_loss"] = (
            base_loss["gen_loss"]
            + self.feat_loss_lambda * feat_loss
            + self.adv_loss_lambda * gen_adv_loss
        )

        return {
            "feat_loss": feat_loss,
            "gen_adv_loss": gen_adv_loss,
            **discr_loss,
            **base_loss,
        }
