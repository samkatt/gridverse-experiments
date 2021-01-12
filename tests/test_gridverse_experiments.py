#!/usr/bin/env python
"""Tests for `gridverse_experiments` package."""


# import pytest  # type: ignore

from gridverse_experiments.online_planning import plan_online


def test_online_planning_runs():
    """Tests whether a basic run of :mod:`~gridverse_experiments.online_planning`"""
    num_parts = 48
    num_sims = 48
    expl_const = 5
    num_runs = 3
    verbose = False
    plan_online(
        "configs/gv_empty.4x4.yaml", num_parts, num_sims, expl_const, num_runs, verbose
    )
