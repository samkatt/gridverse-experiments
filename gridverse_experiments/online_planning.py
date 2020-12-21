"""Experiments for online planning

This module solves environments from the Gridverse_ package with
online-planners_ and belief-tracking_. It contains some 'glue' code, to connect
the two, and provides a straightforward interface to run an :func:`episode`
with different configurations.

Call this script with a path to the environment YAML file.

.. _Gridverse: https://github.com/abaisero/gym-gridverse
.. _online-planners:  https://github.com/samkatt/online-pomdp-planners
.. _belief-tracking:  https://github.com/samkatt/pomdp-belief-tracking

"""

import argparse
from typing import List, Tuple

from gym_gridverse.action import Action as GVerseAction
from gym_gridverse.envs.inner_env import InnerEnv
from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
from gym_gridverse.observation import Observation as GVerseObs
from gym_gridverse.state import State as GVerseState
from online_pomdp_planning import types as planning_types
from online_pomdp_planning.mcts import create_POUCT
from pomdp_belief_tracking import types as belief_types
from pomdp_belief_tracking.pf import AcceptionProgressBar, create_rejection_sampling


def belief_sim_from(
    gridverse_inner_env: InnerEnv,
) -> belief_types.Simulator:
    """Transforms ``gridverse_inner_env`` into a sim usable for the belief

    :param gridverse_inner_env: a domain from ``gym_gridverse``
    :type gridverse_inner_env: InnerEnv
    :return: a simulator suitable for ``pomdp_belief_tracking``
    :rtype: belief_types.Simulator
    """

    class Sim(belief_types.Simulator):
        """A simulator for the belief created from grid-verse domain"""

        def __call__(  # type: ignore
            self, s: GVerseState, a: GVerseAction
        ) -> Tuple[GVerseState, GVerseObs]:
            """A ``online_pomdp_planning`` simulator through ``gym-gridverse`` elements

            TODO: mypy does not accept this signature, see ignored type
            statement above, not sure how to solve the type inference problem.

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

    return Sim()  # should be acceptable for mypy?!


def planner_sim_from(
    gridverse_inner_env: InnerEnv,
) -> planning_types.Simulator:
    """Transforms ``gridverse_inner_env`` into a sim usable for the planner

    :param gridverse_inner_env: a domain from ``gym_gridverse``
    :type gridverse_inner_env: InnerEnv
    :return: a simulator suitable for ``online_pomdp_planning``
    :rtype: planning_types.Simulator
    """

    class Sim(planning_types.Simulator):
        """A simulator for the belief created from grid-verse domain"""

        def __call__(  # type: ignore
            self, s: GVerseState, a: GVerseAction
        ) -> Tuple[GVerseState, GVerseObs, float, bool]:
            """A ``online_pomdp_planning`` simulator through ``gym-gridverse`` elements

            TODO: mypy does not accept this signature, see ignored type
            statement above, not sure how to solve the type inference problem.

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

    return Sim()


def episode(
    planner: planning_types.Planner,
    belief_update: belief_types.BeliefUpdate,
    belief: belief_types.StateDistribution,
    domain: InnerEnv,
) -> List[float]:
    """Runs a single episode

    Acts in ``domain`` according to actions picked by ``planner``, using
    beliefs updated by the ``belief_update`` starting from ``belief``.

    :param planner: the belief-based policy
    :type planner: planning_types.Planner
    :param belief_update: updates belief in between taking actions
    :type belief_update: belief_types.BeliefUpdate
    :param belief: the initial belief
    :type belief: belief_types.StateDistribution
    :param domain: the actual environment in which actions are taken
    :type domain: InnerEnv
    :return: a list of rewards
    :rtype: List[float]
    """

    domain.reset()
    rewards: List[float] = []

    t = False
    while not t:

        print(f"{domain.state.agent}, planning action...")

        a, planner_info = planner(belief)
        assert isinstance(a, GVerseAction)  # XXX: should not be necessary for mypy?!

        r, t = domain.step(a)
        rewards.append(r)

        print(
            f"Planner evaluated actions: {planner_info['max_q_action_selector-values']}"
            f"Taken {a} for reward={r}, updating belief..."
        )

        belief, _ = belief_update(belief, a, domain.observation)

    return rewards


def main() -> None:
    """Main function of online planning

    Call the script with ``-h`` for the accepted arguments. The bare minimum is
    the first argument, a path to a `YAML` file describing the grid-verse
    environment.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("env")

    # initiate arguments
    parser.add_argument("--runs", "-n", type=int, default=1)

    parser.add_argument("--exploration-constant", "-u", type=float, default=1)
    parser.add_argument("--simulations", "-s", type=int, default=48)

    parser.add_argument("--particles", "-b", type=int, default=48)

    # process arguments
    args = parser.parse_args()

    num_particles = args.particles
    num_sims = args.simulations
    u = args.exploration_constant

    # initiate agent and enviornment
    domain = factory_env_from_yaml(args.env)
    planner = create_POUCT(
        domain.action_space.actions,
        planner_sim_from(domain),
        num_sims,
        progress_bar=True,
        ucb_constant=u,
    )
    belief_update = create_rejection_sampling(
        belief_sim_from(domain),
        num_particles,
        process_acpt=AcceptionProgressBar(num_particles),
    )

    # run experiment
    rewards: List[List[float]] = []
    for _ in range(args.runs):
        rewards.append(
            episode(
                planner=planner,
                belief_update=belief_update,
                belief=domain.functional_reset,
                domain=domain,
            )
        )
        print("Episode terminated")

    # TODO: process returns
    print(rewards)


if __name__ == "__main__":
    main()
