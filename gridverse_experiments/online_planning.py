"""Experiments for online planning

This module solves environments from the Gridverse_ package with
online-planners_ and belief-tracking_. It contains some 'glue' code, to connect
the two, and provides a straightforward interface to run an :func:`episode`
with different configurations.

Call this script with a path to the environment YAML file and parameters::

    python gridverse_experiments/online_planning.py -h

    python gridverse_experiments/online_planning.py \
            configs/gv_empty.4x4.deterministic_agent.yaml \
            --logging DEBUG --runs 5 --ucb 5 --sim 128 --part 16 \
            --pouct_evaluation inverted_goal_distance

Otherwise use as a library and provide YAML files to

.. autofunction:: run_from_yaml
   :noindex:

It is also possible to provide a template for YAML files and have them
automatically expanded into separate configurations, using all possible
combinations::

    # in python
    from gridverse_experiments.online_planning import generate_config_expansions
    generate_config_expansions(path/to/template.yaml)


.. _Gridverse: https://github.com/abaisero/gym-gridverse
.. _online-planners:  https://github.com/samkatt/online-pomdp-planners
.. _belief-tracking:  https://github.com/samkatt/pomdp-belief-tracking

"""

import argparse
import logging
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml
from gym_gridverse.action import Action as GVerseAction
from gym_gridverse.envs.inner_env import InnerEnv
from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
from gym_gridverse.observation import Observation as GVerseObs
from gym_gridverse.state import State as GVerseState
from online_pomdp_planning import types as planning_types
from online_pomdp_planning.mcts import Evaluation as MCTSEval
from online_pomdp_planning.mcts import create_POUCT
from pomdp_belief_tracking import types as belief_types
from pomdp_belief_tracking.pf.rejection_sampling import (
    AcceptionProgressBar,
    accept_noop,
    create_rejection_sampling,
)

from gridverse_experiments import conf, utils
from gridverse_experiments.heuristics import inverted_goal_distance


def belief_sim_from(
    gridverse_inner_env: InnerEnv,
) -> belief_types.Simulator:
    """Transforms ``gridverse_inner_env`` into a sim usable for the belief

    :param gridverse_inner_env: a domain from ``gym_gridverse``
    :return: a simulator suitable for ``pomdp_belief_tracking``
    """

    class Sim(belief_types.Simulator):
        """A simulator for the belief created from grid-verse domain"""

        def __call__(
            self, s: GVerseState, a: GVerseAction
        ) -> Tuple[GVerseState, GVerseObs]:
            """A ``online_pomdp_planning`` simulator through ``gym-gridverse`` elements

            :param s: current state
            :param a: current action
            :return: next state-observation-reward pair
            """
            next_s, _, _ = gridverse_inner_env.functional_step(s, a)
            o = gridverse_inner_env.functional_observation(next_s)
            return next_s, o

    return Sim()


def planner_sim_from(
    gridverse_inner_env: InnerEnv,
) -> planning_types.Simulator:
    """Transforms ``gridverse_inner_env`` into a sim usable for the planner

    :param gridverse_inner_env: a domain from ``gym_gridverse``
    :return: a simulator suitable for ``online_pomdp_planning``
    """

    class Sim(planning_types.Simulator):
        """A simulator for the belief created from grid-verse domain"""

        def __call__(
            self, s: GVerseState, a: GVerseAction
        ) -> Tuple[GVerseState, GVerseObs, float, bool]:
            """A ``online_pomdp_planning`` simulator through ``gym-gridverse`` elements

            :param s: current state
            :param a: current action
            :return: next state-observation-reward pair and whether transition is terminal
            """
            next_s, r, t = gridverse_inner_env.functional_step(s, a)
            o = gridverse_inner_env.functional_observation(next_s)
            return next_s, o, r, t

    return Sim()


def episode(
    planner: planning_types.Planner,
    belief: belief_types.Belief,
    domain: InnerEnv,
) -> List[Dict[str, Any]]:
    """Runs a single episode

    Acts in ``domain`` according to actions picked by ``planner``, using
    beliefs updated by the ``belief_update`` starting from ``belief``.

    Returns a list of dictionaries, one for each timestep. The dictionary includes things as:

        - "reward": the reward give to the agent at the time step
        - "terminal": whether the step was terminal (should really only be last, if any)
        - "timestep": the time step (should be equal to the index)
        - information from the planner info
        - information from belief info

    TODO: include horizon

    :param planner: the belief-based policy
    :param belief: updates belief in between taking actions
    :param domain: the actual environment in which actions are taken
    :return: a list of episode results (rewards and info dictionaries)
    """
    logger = logging.getLogger("episode")

    domain.reset()

    info: List[Dict[str, Any]] = []

    terminal = False
    timestep = 0
    while not terminal:

        logger.debug("%s, planning action...", domain.state.agent)

        a, planner_info = planner(belief.sample)

        assert isinstance(a, GVerseAction)

        r, terminal = domain.step(a)

        logger.debug(
            "Planner evaluated actions %s\nTaken %s for reward %s, updating belief...",
            planner_info["max_q_action_selector-values"],
            a,
            r,
        )

        belief_info = belief.update(a, domain.observation)

        info.append(
            {
                "timestep": timestep,
                "reward": r,
                "terminal": terminal,
                **planner_info,
                **belief_info,
            }
        )

        timestep += 1

    return info


def main(
    domain: InnerEnv,
    planner: planning_types.Planner,
    belief: belief_types.Belief,
    runs: int,
    logging_level: str,
) -> List[Dict[str, Any]]:
    """plan online function of online planning

    Handles calling :func:`episode` ::

        for r in runs:
            rewards = episode(domain, planner, belief_updat)

    In the episode actions are taken in ``domain`` according to the
    ``planner``, which uses a belief maintained by ``belief``.

    Returns a one-dimensional list of dictionaries. This is a flat
    concatenation of the results returned by :func:`episode`. Each entry (dict)
    has a key "run" that indicates the number o fthe run it was produced.

    :param domain:
    :param planner:
    :param belief:
    :param runs:
    :param logging_level:
    :return: flat concatenation of the results of each episode
    """

    utils.set_logging_options(logging_level)

    logger = logging.getLogger("plan-online")
    logger.info("starting %s run(s)", runs)

    output: List[Dict[str, Any]] = []
    for run in range(runs):

        belief.distribution = domain.functional_reset

        episode_output = episode(
            planner=planner,
            belief=belief,
            domain=domain,
        )

        # here we explicitly add the information of which run the result was
        # generated to each entry in the results
        for o in episode_output:
            o["run"] = run

        # extend -- flat concatenation -- of our results
        output.extend(episode_output)

        logger.info(
            "run %s/%s terminated: r(%s)",
            run + 1,
            runs,
            utils.discounted_return([t["reward"] for t in episode_output], 0.95),
        )

    return output


def run_from_yaml(env_yaml_file: str, solution_params_yaml: str):
    """Calls :func:`plan_online` with arguments described in YAML

    Under the hood calls :func:`_load_from_dict` to initiate the planner,
    domain and belief update.

    :param env_yaml_file: YAML path describing the gridverse domain scheme
    :param solution_params_yaml: YAML path with parameters (`configs/example_online_planning.yaml`)
    """
    with open(solution_params_yaml) as f:
        args = yaml.safe_load(f)

    args["env"] = env_yaml_file
    run_from_dict(args)


def run_from_dict(args: Dict[str, Any]):
    """Runs the program given arguments in ``args``

    Messy function that loads the domain, planner and belief given
    config in ``args``.

    Will print out an error if the incorrect keys are given (or mising)

    Then calls main

    :param args:
    """

    try:
        domain = factory_env_from_yaml(args["env"])

        state_evaluation = create_state_evaluation(args["pouct_evaluation"])
        planner = create_POUCT(
            actions=domain.action_space.actions,
            sim=planner_sim_from(domain),
            num_sims=args["simulations"],
            init_stats=None,
            leaf_eval=state_evaluation,
            ucb_constant=args["ucb_constant"],
            rollout_depth=args["rollout_depth"],
            discount_factor=args["discount_factor"],
            progress_bar=args["logging"] == "DEBUG",
        )

        process_acpt = (
            AcceptionProgressBar(args["particles"])
            if args["logging"] == "DEBUG"
            else accept_noop
        )

        belief_update = create_rejection_sampling(
            sim=belief_sim_from(domain),
            n=args["particles"],
            process_acpt=process_acpt,
        )

        belief = belief_types.Belief(domain.functional_reset, belief_update)

    except KeyError as e:
        logging.warning("YAML file error: %s", str(e))
        exit()

    if "save_path" in args and args["save_path"]:
        utils.create_experiments_directory_or_exit(args["save_path"])

    result = main(domain, planner, belief, args["runs"], args["logging"])

    if "save_path" in args and args["save_path"]:
        with open(os.path.join(args["save_path"], "params.yaml"), "w") as outfile:
            yaml.dump(args, outfile, default_flow_style=False)
        shutil.copyfile(args["env"], os.path.join(args["save_path"], "env.yaml"))
        pd.DataFrame(result).to_pickle(
            os.path.join(args["save_path"], "timestep_data.pkl")
        )


def create_state_evaluation(strategy: str) -> Optional[MCTSEval]:
    """Creates a MCTS leaf evaluation strategy

    Mapping from configuration to online-planning leaf evaluation method

    :param strategy: description of the evalutation (in ["", "inverted_goal_distance"])
    :return: an evaluation strategy for POUCT
    """
    # base case
    if not strategy:
        return None

    if strategy == "inverted_goal_distance":

        def evaluation(s: GVerseState, o, t: bool, info) -> float:
            """State evalution, calls ``inverted_goal_distance`` if not terminal


            Follows the protocol of ``Evaluation`` in ``MCTS`` of ``online_pomdp_planning``
            """
            if t:
                return 0.0
            return inverted_goal_distance(s, multiplier=1)

        return evaluation

    raise ValueError(f"pouct_evaluation {strategy} is not supported")


def generate_config_expansions(yaml_template_path: str) -> None:
    """Tool to generate ``config.yaml`` files from ``yaml_template_path``

    Assumes ``yaml_template_path`` is a ``yaml`` file that contains list
    entries that are supposed to be 'expanded', i.e., generate a combinatorial
    set. This function will do so, and write them to disk under a sane name

    :param yaml_template_path: path to template config file with list entries
    :returns: None, writes to disk
    """

    with open(yaml_template_path, "r") as input_file:
        config = yaml.safe_load(input_file)
    expansions = conf.expand_conf(config)

    assert (
        len(expansions) != 0
    ), f"Somehow {yaml_template_path} generated zero expansions"

    for n, c in expansions.items():
        c["save_path"] = n
        expansions_name = f"{n}.yaml"
        with open(expansions_name, "w") as output_file:
            yaml.dump(c, output_file, default_flow_style=False)


if __name__ == "__main__":
    """Runs GBA-POMDP

    Assumes arguments are passed in per command line, calls main with them
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("env")
    parser.add_argument("--runs", "-n", type=int, default=1)
    parser.add_argument("--discount_factor", "-y", type=float, default=0.95)

    parser.add_argument("--ucb_constant", "-u", type=float, default=1)
    parser.add_argument("--simulations", "-s", type=int, default=32)
    parser.add_argument("--rollout_depth", "-d", type=int, default=100)
    parser.add_argument(
        "--pouct_evaluation", choices=["", "inverted_goal_distance"], default=""
    )

    parser.add_argument("--particles", "-b", type=int, default=32)
    parser.add_argument("--save_path", type=str)

    parser.add_argument(
        "--logging",
        choices=["INFO", "DEBUG", "WARNING"],
        default="INFO",
        help="Logging level, set to `WARNING` for no output, `DEBUG` for additional info",
    )

    run_from_dict(vars(parser.parse_args()))
