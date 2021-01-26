"""Visualization utilities for tensorboard

For online logging we provide tensorboard logging API, e.g.::

    set_tensorboard_logging("test_directory/to/log/to")
    log_tensorboard("accuracy", 10, step=0)
    log_tensorboard("accuracy", 5, step=1)

"""

from typing import Optional, Union

import numpy as np
import torch.utils.tensorboard as tboard

_TENSORBOARD_WRITER: Optional[tboard.writer.SummaryWriter] = None


def set_tensorboard_logging(log_dir: str) -> None:
    """set what directory to write tensorboard results to

    :param log_dir: what, if any, directory to write tensorboard results to
    """
    global _TENSORBOARD_WRITER
    _TENSORBOARD_WRITER = tboard.SummaryWriter(log_dir=f".tensorboard/{log_dir}")


def log_tensorboard(tag: str, val: Union[float, np.ndarray], step: int) -> None:
    """logs a scalar or histogram to tensorboard

    :param tag: the 'topic' to write results to
    :param val: either a scalar or a histogram
    :param step: where on the x-axis the result should be written to
    """

    assert tensorboard_logging(), "please first verify logging is on"
    assert _TENSORBOARD_WRITER

    if np.isscalar(val):
        _TENSORBOARD_WRITER.add_scalar(tag, val, step)
    else:
        _TENSORBOARD_WRITER.add_histogram(tag, val, step)


def tensorboard_logging() -> bool:
    """returns whether we are logging to tensorboard

    :return: true if tensorboard logging is on
    """
    return _TENSORBOARD_WRITER is not None
