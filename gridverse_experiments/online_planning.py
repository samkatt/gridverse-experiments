"""Experiments for online planning"""

from copy import deepcopy
from typing import Tuple

from gym_gridverse.actions import Actions as GVerseAction
from gym_gridverse.envs.inner_env import InnerEnv
from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
from gym_gridverse.observation import Observation as GVerseObs
from gym_gridverse.state import State as GVerseState
from online_pomdp_planning import types as planning_types
from online_pomdp_planning.mcts import create_POUCT
from pomdp_belief_tracking import types as belief_types
from pomdp_belief_tracking.pf import create_rejection_sampling


def belief_sim_from(gridverse_inner_env: InnerEnv) -> belief_types.Simulator:
    """Transforms ``gridverse_inner_env`` into a sim usable for the belief

    :param gridverse_inner_env: a domain from ``gym_gridverse``
    :type gridverse_inner_env: InnerEnv
    :return: a simulator suitable for ``pomdp_belief_tracking``
    :rtype: belief_types.Simulator
    """

    def sim(s: GVerseState, a: GVerseAction) -> Tuple[GVerseState, GVerseObs]:
        """A ``online_pomdp_planning`` simulator through ``gym-gridverse`` elements

        :param s: current state
        :type s: GVerseState
        :param a: current action
        :type a: GVerseAction
        :return: next state-observation-reward pair
        :rtype: Tuple[GVerseState, GVerseObs]
        """
        next_s, _, _ = gridverse_inner_env.functional_step(s, a)
        o = gridverse_inner_env.functional_observation(next_s)
        return next_s, o

    # TODO: understand and fix mypy error
    return sim  # type: ignore


def planner_sim_from(gridverse_inner_env: InnerEnv) -> planning_types.Simulator:
    """Transforms ``gridverse_inner_env`` into a sim usable for the planner

    :param gridverse_inner_env: a domain from ``gym_gridverse``
    :type gridverse_inner_env: InnerEnv
    :return: a simulator suitable for ``online_pomdp_planning``
    :rtype: planning_types.Simulator
    """

    def sim(
        s: GVerseState, a: GVerseAction
    ) -> Tuple[GVerseState, GVerseObs, float, bool]:
        """A ``online_pomdp_planning`` simulator through ``gym-gridverse`` elements

        :param s: current state
        :type s: GVerseState
        :param a: current action
        :type a: GVerseAction
        :return: next state-observation-reward pair and whether transition is terminal
        :rtype: planning_types.Simulator
        """
        next_s, r, t = gridverse_inner_env.functional_step(s, a)
        o = gridverse_inner_env.functional_observation(next_s)
        return next_s, o, r, t

    # TODO: understand and fix mypy error
    return sim  # type: ignore


def main() -> None:
    """TODO"""
    domain = factory_env_from_yaml("configs/gv_empty.4x4.yaml")

    pouct = create_POUCT(domain.action_space.actions, planner_sim_from(domain))
    rs = create_rejection_sampling(belief_sim_from(domain), 128)

    domain.reset()
    belief = domain.functional_reset
    while True:

        a = pouct(lambda: deepcopy(belief()), 100)

        print(f"Taken {a}, in {domain.state.agent.position}")

        domain.step(a)

        o = domain.observation
        # TODO: understand and fix mypy error
        belief = rs(lambda: deepcopy(belief()), a, o)  # type: ignore

        print(f"New position is {domain.state.agent.position}")


if __name__ == "__main__":
    main()
