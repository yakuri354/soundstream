from abc import abstractmethod

from torch import nn


class BaseMetric(nn.Module):
    """
    Base class for all metrics
    """

    def __init__(self, name=None, *args, each_epoch: int = 1, **kwargs):
        """
        Args:
            name (str | None): metric name to use in logger and writer.
        """
        super().__init__()
        self.each_epoch = each_epoch
        self.name = name if name is not None else type(self).__name__

    def should_run(self, epoch: int, **kwargs):
        return epoch % self.each_epoch == 0
