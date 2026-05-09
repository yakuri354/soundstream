from pathlib import Path

import pandas as pd
import torch

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        params = {"compute_loss": True}

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            for opt in self.optimizers:
                self.optimizers[opt].zero_grad()
            params["update_codebook"] = True

        if not self.model.is_init:
            # assert self.is_train
            self.model.initialize(batch["input"])

        if self.autocast and self.is_train:
            with torch.autocast(device_type=self.device_type):
                outputs = self.model(**batch, **params)
        else:
            outputs = self.model(**batch, **params)

        batch.update(outputs)

        if self.is_train:
            batch["gen_loss"].backward()
            self._clip_grad_norm()
            self.optimizers["generator"].step()
            if "generator" in self.lr_schedulers:
                self.lr_schedulers["generator"].step()

            self.optimizers["discriminator"].zero_grad()
            batch["discr_loss"].backward()
            self._clip_grad_norm()
            self.optimizers["discriminator"].step()
            if "discriminator" in self.lr_schedulers:
                self.lr_schedulers["discriminator"].step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            for key, value in met(batch).items():
                metrics.update(key, value)
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        self.writer.add_audio(
            "input", batch["input"][:1, 0, :], sample_rate=self.model.sample_rate
        )
        self.writer.add_audio(
            "decoded", batch["decoded"][:1, 0, :], sample_rate=self.model.sample_rate
        )
