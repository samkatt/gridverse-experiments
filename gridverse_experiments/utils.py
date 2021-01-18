"""Basic recurring functionality"""

import logging
from typing import List


def set_logging_options(log_level: str):
    """Some basic options to make logs more readable

    Sets debug level, provide "WARNING" for no additional information, "DEBUG"
    for extra

    :param log_level: in ["DEBUG", "INFO", "WARNING"
    """
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s:%(name)s %(asctime)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def discounted_return(rewards: List[float], discount: float) -> float:
    """Computes the discounted return of the ``rewards`` given the ``discount``

    Args:
        rewards (`List[float]'):
        discount (`float`):

    Returns:
        `float`: the discounted returns
    """
    assert 0 <= discount <= 1, f"discount can really not be {discount}"

    # sum ( discount^t r_t )
    return sum(pow(discount, i) * r for i, r in enumerate(rewards))
