"""Tests for `gridverse_experiments.heuristics` module"""

import pytest
from gym_gridverse.envs.reset_functions import reset_empty

from gridverse_experiments.heuristics import inverted_goal_distance


@pytest.mark.parametrize(
    "agent_pos,multiplier,val",
    [
        ((3, 4), 0, 0),
        ((5, 5), 234, 2340),
        ((5, 4), 3, 3),
        ((5, 2), 3, 1),
    ],
)
def test_heuristics_on_empty(agent_pos, multiplier, val):
    s = reset_empty(7, 7)
    s.agent.position = agent_pos

    assert inverted_goal_distance(s, multiplier=multiplier) == val
    assert inverted_goal_distance(s, multiplier=multiplier, goal_pos=(5, 5)) == val
