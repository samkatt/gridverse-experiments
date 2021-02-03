"""Tests for `gridverse_experiments.heuristics` module"""

import pytest
from gym_gridverse.envs.reset_functions import reset_empty

from gridverse_experiments.heuristics import goal_distance, inverted_goal_distance


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


@pytest.mark.parametrize(
    "agent_pos,goal_pos,dist",
    [
        ((5, 5), None, 0),
        ((5, 5), (5, 5), 0),
        ((2, 3), (4, 5), 4),
        ((1, 1), None, 8),
    ],
)
def test_goal_dist(agent_pos, goal_pos, dist):
    s = reset_empty(7, 7)
    s.agent.position = agent_pos
    assert goal_distance(s, goal_pos) == dist
