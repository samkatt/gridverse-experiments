""" Basic recurring functionality

An ensemble of functions that is considered useful to most experiments in this
package. This includes:

    - :func:`set_logging_options` to set basic logging config
    - computing :func:`discounted_return` from rewards
    - handling creating experiments directory
      :func:`create_experiments_directory` and saving data in them
      :func:`save_experiments_data`
"""

import logging
import os
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


def create_directory_or_exit(exp_dir_path: str) -> None:
    """Creates a directory at ``exp_dir_path``

    NOTE: exits the program when exists, this is considered an unresolvable
    error.

    :param exp_dir_path:
    :return: None
    """
    try:
        os.makedirs(exp_dir_path)
    except OSError as e:
        logging.warning("Exiting! Path '%s' produced error: %s", exp_dir_path, str(e))
        exit()
