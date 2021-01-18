#!/usr/bin/env python
"""Tests for `gba_pomdp` module."""


import pytest

from gridverse_experiments.utils import discounted_return


@pytest.mark.parametrize(
    "rewards,discount,ret",
    [
        ([0], 0, 0),
        ([0], 0.1, 0),
        ([5, 10], 0, 5),
        ([5, 20], 0.5, 15),
        (
            [
                5,
                20,
                0,
            ],
            0.5,
            15,
        ),
        ([1, 2, 3, 4, 5], 0.99, 14.604476049999999),
    ],
)
def test_episode_result(rewards, discount, ret):
    """Tests computing :meth:`EpisodeResult.discounted_return`"""
    assert discounted_return(rewards, discount) == ret


@pytest.mark.parametrize(
    "rewards,discount",
    [
        ([0], -0.1),
        ([0], -0.1),
        ([5, 10], -10),
        ([5, 20], -0.5),
        (
            [
                5,
                20,
                0,
            ],
            -1.1,
        ),
    ],
)
def test_episode_result_asserts(rewards, discount):
    """Tests :meth:`EpisodeResult.discounted_return` asserts"""
    with pytest.raises(AssertionError):
        discounted_return(rewards, discount)
