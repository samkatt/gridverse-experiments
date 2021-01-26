#!/usr/bin/env python
"""Tests for `gridverse_experiments` package."""


from gridverse_experiments.builtin_gbapomdp import (
    run_from_yaml as run_builtin_gba_pomdp,
)
from gridverse_experiments.gba_pomdp import run_from_yaml as run_gba_pomdp
from gridverse_experiments.online_planning import (
    run_from_dict as online_planning_from_dict,
)
from gridverse_experiments.online_planning import (
    run_from_yaml as online_planning_from_yaml,
)


def test_online_planning_runs():
    """Tests a basic run of :mod:`gridverse_experiments.online_planning`"""
    args = {
        "env": "tests/configs/gv_empty.4x4.deterministic_agent.yaml",
        "runs": 2,
        "discount_factor": 0.95,
        "ucb_constant": 1,
        "simulations": 32,
        "rollout_depth": 5,
        "particles": 16,
        "logging": "WARNING",
        "pouct_evaluation": "",
    }
    online_planning_from_dict(args)
    online_planning_from_yaml(
        "tests/configs/gv_empty.4x4.deterministic_agent.yaml",
        "tests/configs/example_online_planning.yaml",
    )


def test_gba_pomdp():
    """Tests a basic run of :mod:`~gridverse_experiments.gba_pomdp`"""
    run_gba_pomdp(
        "tests/configs/gv_empty.4x4.deterministic_agent.yaml",
        "tests/configs/example_gba_pomdp.yaml",
    )


def test_builtin_gba_pomdp():
    """Tests a basic run of :mod:`~gridverse_experiments.gba_pomdp`"""
    run_builtin_gba_pomdp("tests/configs/example_builtin_gba_pomdp.yaml")
