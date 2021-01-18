#!/usr/bin/env python
"""Tests for `gridverse_experiments` package."""


from gridverse_experiments.gba_pomdp import run_from_yaml as run_gba_pomdp
from gridverse_experiments.online_planning import run_from_dict as online_planning


def test_online_planning_runs():
    """Tests a basic run of :mod:`gridverse_experiments.online_planning`"""
    args = {
        "env": "configs/gv_empty.4x4.deterministic_agent.yaml",
        "runs": 3,
        "discount_factor": 0.95,
        "ucb_constant": 5,
        "simulations": 32,
        "rollout_depth": 5,
        "particles": 32,
        "logging": "WARNING",
    }
    online_planning(args)


def test_gba_pomdp():
    """Tests a basic run of :mod:`~gridverse_experiments.gba_pomdp`"""
    run_gba_pomdp("configs/example_gba_pomdp.yaml")
