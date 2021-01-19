#!/usr/bin/env python
"""Tests for `gba_pomdp` module."""


import pytest

from gridverse_experiments.conf import expand_conf


@pytest.mark.parametrize(
    "conf,expansion",
    [
        ({"some_setting": -3}, {"": {"some_setting": -3}}),
        ({"some_setting": [-3]}, {"some_setting--3": {"some_setting": -3}}),
        (
            {"some_setting": [-3, "hah"]},
            {
                "some_setting--3": {"some_setting": -3},
                "some_setting-hah": {"some_setting": "hah"},
            },
        ),
        (
            {"some_setting": [-3, "ha", 100], "runs": [2, 10], "flag": True},
            {
                "some_setting--3~~~runs-2": {
                    "some_setting": -3,
                    "runs": 2,
                    "flag": True,
                },
                "some_setting--3~~~runs-10": {
                    "some_setting": -3,
                    "runs": 10,
                    "flag": True,
                },
                "some_setting-ha~~~runs-2": {
                    "some_setting": "ha",
                    "runs": 2,
                    "flag": True,
                },
                "some_setting-ha~~~runs-10": {
                    "some_setting": "ha",
                    "runs": 10,
                    "flag": True,
                },
                "some_setting-100~~~runs-2": {
                    "some_setting": 100,
                    "runs": 2,
                    "flag": True,
                },
                "some_setting-100~~~runs-10": {
                    "some_setting": 100,
                    "runs": 10,
                    "flag": True,
                },
            },
        ),
    ],
)
def test_episode_result_asserts(conf, expansion):
    """Tests :meth:`EpisodeResult.discounted_return` asserts"""
    print("input:", conf)
    assert expand_conf(conf) == expansion
