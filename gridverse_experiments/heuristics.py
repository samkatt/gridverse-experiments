"""Domain knowledge and heuristics of gridverse

Contains:

    - Approximate state evaluation methods

"""

from typing import Optional

from gym_gridverse.geometry import Position, PositionOrTuple
from gym_gridverse.grid_object import Goal
from gym_gridverse.state import State


def goal_distance(s: State, goal_pos: Optional[PositionOrTuple] = None):
    """Returns distance (manhatten) from agent to goal

    :param s:
    :param goal_pos: optional give goal pos, otherwise computed
    :return:
    """

    if not goal_pos:
        goals = list(filter(lambda p: isinstance(s.grid[p], Goal), s.grid.positions()))
        assert len(goals) > 0

        goal_pos = goals[0]

    return Position.manhattan_distance(s.agent.position, goal_pos)


def inverted_goal_distance(
    s: State, goal_pos: Optional[PositionOrTuple] = None, multiplier=10
) -> float:
    """Returns a number inverse to the distance to the goal

    :param s: state to evaluate
    :param goal_pos: the position of the goal, searched for in ``s`` when not given
    :param multiplier: what to multiply the inversion with
    :return: ``multiplier``/``d``, where ``d`` is the distance to the goal
    """
    # make sure that dist is not too small for instabilities
    # if we are on the goal, for example, we'd get division by 0
    dist = max(goal_distance(s, goal_pos), 0.1)

    return multiplier / dist
