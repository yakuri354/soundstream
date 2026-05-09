from abc import abstractmethod

from torch import nn


class BaseMetric(nn.Module):
    """
    Base class for all metrics
    """

    def __init__(self, name=None, *args, **kwargs):
        """
        Args:
            name (str | None): metric name to use in logger and writer.
        """
        super().__init__()
        self.name = name if name is not None else type(self).__name__
