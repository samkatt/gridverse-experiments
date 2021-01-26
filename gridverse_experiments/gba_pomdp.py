"""Solving grid verse with GBA-POMDP

GBA-POMDP is a Bayesian reinforcement learning framework. In this approach the
ssumption is that the dynamics of the environment are not known, but learned
while interacting in the environment.

The GBA-POMDP casts the _learning problem_ into a _planning problem_, where the
dynamics are known. These dynamics govern how to maintain the belief over both
the current state and dynamics of the environment. The result is solved by a
combination of belief-tracking_ and online-planning_.

.. _online-planners:  https://github.com/samkatt/online-pomdp-planners
.. _belief-tracking:  https://github.com/samkatt/pomdp-belief-tracking

Example usage::

    python gridverse_experiments/gba_pomdp.py \
            configs/gv_empty.8x8.yaml --logging DEBUG \
            --episodes 2 --pouct_evaluation inverted_goal_distance \
            -B rejection_sampling --learning_rate .05 --dropout_rate .5

Otherwise use as a library and provide YAML files to

.. autofunction:: run_from_yaml
   :noindex:

"""

import logging
import os
import shutil
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import deque
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import general_bayes_adaptive_pomdps.pytorch_api
import numpy as np
import online_pomdp_planning.types as planner_types
import pandas as pd
import pomdp_belief_tracking.types as belief_types
import yaml
from general_bayes_adaptive_pomdps.misc import set_random_seed
from general_bayes_adaptive_pomdps.models.partial.domain.gridverse_gbapomdps import (
    GridversePositionAugmentedState,
    create_gbapomdp,
    gverse_obs2array,
)
from general_bayes_adaptive_pomdps.models.partial.partial_gbapomdp import (
    GBAPOMDPThroughAugmentedState,
)
from gym_gridverse.action import Action as GVerseAction
from gym_gridverse.envs.inner_env import InnerEnv
from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
from gym_gridverse.representations.observation_representations import (
    DefaultObservationRepresentation,
)
from online_pomdp_planning.mcts import Evaluation as MCTSEval
from online_pomdp_planning.mcts import create_POUCT as lib_create_POUCT
from pomdp_belief_tracking.pf import importance_sampling as IS
from pomdp_belief_tracking.pf import particle_filter as PF
from pomdp_belief_tracking.pf import rejection_sampling as RS

from gridverse_experiments import conf, utils
from gridverse_experiments.heuristics import inverted_goal_distance


def main(conf: Dict[str, Any]) -> None:
    """runs PO-UCT planner with a belief on given configurations

    :param args: optional list of arguments
    """

    # post process arguments
    if not conf["search_depth"]:
        conf["search_depth"] = conf["horizon"]
    if not conf["belief_minimal_sample_size"]:
        conf["belief_minimal_sample_size"] = conf["num_particles"]

    if "save_path" in conf and conf["save_path"]:
        utils.create_experiments_directory_or_exit(conf["save_path"])

    utils.set_logging_options(conf["logging"])
    logger = logging.getLogger("GBA-POMDP")

    if conf["tensorboard_logdir"]:
        general_bayes_adaptive_pomdps.pytorch_api.set_tensorboard_logging(
            conf["tensorboard_logdir"]
        )

    # TODO: improve API: give device to `BADDr`
    general_bayes_adaptive_pomdps.pytorch_api.set_device(conf["use_gpu"])

    # TODO: allow seeds in `pomdp_belief_tracking` and `online_pomdp_planning`
    if conf["random_seed"]:
        set_random_seed(conf["random_seed"])

    env = factory_env_from_yaml(conf["env"])

    baddr = create_gbapomdp(
        env,  # type: ignore
        conf["optimizer"],
        conf["learning_rate"],
        conf["network_size"],
        conf["dropout_rate"],
        conf["num_pretrain_epochs"],
        conf["batch_size"],
    )

    planner = create_planner(
        baddr,
        conf["num_sims"],
        conf["exploration"],
        conf["search_depth"],
        conf["gamma"],
        conf["pouct_evaluation"],
        conf["logging"],
    )
    belief = create_belief(
        baddr,
        conf["belief"],
        conf["num_particles"],
        conf["belief_minimal_sample_size"],
        conf["logging"],
    )

    def set_domain_state(augmented_state: GridversePositionAugmentedState):
        """sets domain state in ``s`` to sampled initial state """
        augmented_state.domain_state = env.functional_reset()
        return augmented_state

    output: List[Dict[str, Any]] = []

    for run in range(conf["runs"]):

        avg_recent_return = deque([], 50)

        for episode in range(conf["episodes"]):

            # TODO: there has to be a better way
            if episode > 0:
                belief.distribution = PF.apply(
                    set_domain_state,  # type: ignore
                    belief.distribution,
                )

            episode_output = run_episode(env, planner, belief, conf["horizon"])

            # here we explicitly add the information of which run the result
            # was generated to each entry in the results
            for o in episode_output:
                o["episode"] = episode
                o["run"] = run

            # extend -- flat concatenation -- of our results
            output.extend(episode_output)

            discounted_return = utils.discounted_return(
                [t["reward"] for t in episode_output], conf["gamma"]
            )
            avg_recent_return.append(discounted_return)

            logger.warning(
                "Episode %s/%s return: %s",
                episode + 1,
                conf["episodes"],
                discounted_return,
            )
            logger.info(
                f"run {run+1}/{conf['runs']} episode {episode+1}/{conf['episodes']}: "
                f"avg return: {np.mean(avg_recent_return)}"
            )

    if "save_path" in conf and conf["save_path"]:
        with open(os.path.join(conf["save_path"], "params.yaml"), "w") as outfile:
            yaml.dump(conf, outfile, default_flow_style=False)
        shutil.copyfile(conf["env"], os.path.join(conf["save_path"], "env.yaml"))
        pd.DataFrame(output).to_pickle(
            os.path.join(conf["save_path"], "timestep_data.pkl")
        )


def run_episode(
    env: InnerEnv,
    planner: planner_types.Planner,
    belief: belief_types.Belief,
    horizon: int,
) -> List[Dict[str, Any]]:
    """runs a single episode

    Returns information returned by the planner and belief in a list, where the
    nth element is the info of the nth episode step.

    Returns a list of dictionaries, one for each timestep. The dictionary includes things as:

        - "reward": the reward give to the agent at the time step
        - "terminal": whether the step was terminal (should really only be last, if any)
        - "timestep": the time step (should be equal to the index)
        - information from the planner info
        - information from belief info

    :param env:
    :param planner:
    :param belief:
    :param horizon: length of episode
    :return: a list of episode results (rewards and info dictionaries)
    """

    logger = logging.getLogger("episode")
    env.reset()

    info: List[Dict[str, Any]] = []

    # TODO: generalize to whatever obs2array is used in the GBA-POMDP
    obs2array = partial(
        gverse_obs2array,
        env,
        DefaultObservationRepresentation(env.observation_space),
    )

    for timestep in range(horizon):

        # actual step
        action, planning_info = planner(belief.sample)
        r, t = env.step(GVerseAction(action))

        logger.debug(
            "A(%s) -> (%s %s) --- r(%s)",
            GVerseAction(action),
            env.state.agent.position,
            env.state.agent.orientation,
            r,
        )

        belief_info = belief.update(action, obs2array(env.observation))

        logger.debug(
            "Planner output: %s \nBelief output: %s", planning_info, belief_info
        )

        info.append(
            {
                "timestep": timestep,
                "reward": r,
                "terminal": t,
                **planning_info,
                **belief_info,
            }
        )

        if t:
            break

    return info


def create_planner(
    baddr: GBAPOMDPThroughAugmentedState,
    num_sims: int = 500,
    exploration_constant: float = 1.0,
    planning_horizon: int = 10,
    discount: float = 0.95,
    pouct_evaluation: str = "",
    log_level: str = "INFO",
) -> planner_types.Planner:
    """The factory function for planners

    Currently just returns PO-UCT with given parameters, but allows for future generalization

    Real `env` is used for the rollout policy

    :param baddr:
    :param num_sims: number of simulations to run
    :param exploration_constant: the UCB-constant for UCB
    :param planning_horizon: how far into the future to plan for
    :param discount: the discount factor to plan for
    :param pouct_evaluation: leaf evaluation strategy (in ["", "pouct_evaluation"])
    :param log_level: in ["DEBUG", "INFO", "WARNING"]
    """

    actions = list(np.int64(i) for i in range(baddr.action_space.n))
    online_planning_sim = SimForPlanning(baddr)
    state_evaluation = create_state_evaluation(pouct_evaluation)

    return lib_create_POUCT(
        actions,
        online_planning_sim,
        num_sims,
        discount_factor=discount,
        rollout_depth=planning_horizon,
        leaf_eval=state_evaluation,
        ucb_constant=exploration_constant,
        progress_bar=log_level == "DEBUG",
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

        def evaluation(s: GridversePositionAugmentedState, o, t: bool, info) -> float:
            """State evalution, calls ``inverted_goal_distance`` if not terminal


            Follows the protocol of ``Evaluation`` in ``MCTS`` of ``online_pomdp_planning``
            """
            if t:
                return 0.0
            return inverted_goal_distance(s.domain_state)

        return evaluation

    raise ValueError(f"pouct_evaluation {strategy} is not supported")


def create_belief(
    gbapomdp: GBAPOMDPThroughAugmentedState,
    belief: str,
    num_particles: int,
    minimal_sample_size: float,
    log_level: str,
) -> belief_types.Belief:
    """Creates the belief update

    Dispatches to :func:`create_rejection_sampling` or
    :func:`create_importance_sampling` to create the actual belief update

    :param gbapomdp: the GBA-POMDP to update the belief for
    :param belief: configuration name ("importance_sampling" or "rejection_sampling")
    :param num_particles: number of particles
    :param minimal_sample_size: threshold before resampling
    :param log_level: in ["DEBUG", "INFO", "WARNING"]
    """
    if belief == "rejection_sampling":
        bu = create_rejection_sampling(num_particles, log_level)
    elif belief == "importance_sampling":
        bu = create_importance_sampling(gbapomdp, num_particles, minimal_sample_size)
    else:
        raise ValueError(f"{belief} not accepted belief configuration")

    return belief_types.Belief(gbapomdp.sample_start_state, bu)


def create_rejection_sampling(
    num_samples: int, log_level: str
) -> belief_types.BeliefUpdate:
    """Creates a rejection-sampling belief update

    Returns a rejection sampling belief update that tracks ``num_samples``
    particles in the ``baddr``. Basically glue between
    ``general_bayes_adaptive_pomdps`` and ``pomdp_belief_tracking``.

    :param num_samples: number of particles to main
    :param log_level: in ["DEBUG", "INFO", "WARNING"]
    """

    progress_bar = (
        RS.AcceptionProgressBar(num_samples) if log_level == "DEBUG" else RS.accept_noop
    )

    def process_acpt(ss: GridversePositionAugmentedState, ctx, info):
        # update the parameters of the augmented state
        # without modifying original state
        progress_bar(ss, ctx, info)
        return ss.update_model_distribution(
            ctx["state"], ctx["action"], ss, ctx["observation"]
        )

    def belief_sim(s: GridversePositionAugmentedState, a: int):
        return s.domain_step(a)

    return RS.create_rejection_sampling(
        belief_sim, num_samples, np.array_equal, process_acpt  # type: ignore
    )


def create_importance_sampling(
    baddr: GBAPOMDPThroughAugmentedState, num_samples: int, minimal_sample_size: float
) -> belief_types.BeliefUpdate:
    """Creates importance sampling

    Returns a rejection sampling belief update that tracks ``num_samples``
    particles in the ``baddr``. Basically glue between
    ``general_bayes_adaptive_pomdps`` and ``pomdp_belief_tracking``.

    Uses ``baddr`` to simulate and reject steps.

    :param baddr: the GBA-POMDP to track belief for
    :param num_samples: number of particles to main
    :param minimal_sample_size: threshold before resampling
    """

    def transition_func(s, a):
        return baddr.simulation_step(s, a, optimize=True).state

    def obs_model(s, a, ss: GridversePositionAugmentedState, o) -> float:
        # XXX: we know the observation function is deterministic. We also know
        # exaclty how it is called under the hood. So here we call it, and see
        # if it produces the observation perceived by the agent
        return float(np.array_equal(o, ss.observation))

    resample_condition = partial(IS.ineffective_sample_size, minimal_sample_size)

    return IS.create_sequential_importance_sampling(
        resample_condition, transition_func, obs_model, num_samples  # type: ignore
    )


class SimForPlanning(planner_types.Simulator):
    """A simulator for ``online_pomdp_planning`` from ``general_bayes_adaptive_pomdps``"""

    def __init__(self, bnrl_simulator: GBAPOMDPThroughAugmentedState):
        """Wraps and calls ``bnrl_simulator`` with imposed signature

        :param bnrl_simulator:
        """
        super().__init__()
        self.gbapomdp = bnrl_simulator

    def __call__(
        self, s: np.ndarray, a: int
    ) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        """The signature for the simulator for online planning

        Upon calling, produces a transition (state, observation, reward, terminal)

        :param s: input state
        :param a: input action
        """
        next_s, obs = self.gbapomdp.simulation_step(s, a)  # TODO: no copy!
        reward = self.gbapomdp.reward(s, a, next_s)
        terminal = self.gbapomdp.terminal(s, a, next_s)

        return next_s, obs.data.tobytes(), reward, terminal


def run_from_yaml(env_yaml_file: str, solution_params_yaml: str):
    """Calls :func:`plan_online` with arguments described in YAML

    :param env_yaml_file: YAML path to env (`configs/gv_empty.4x4.deterministic_agent.yaml`)
    :param solution_params_yaml: YAML path with parameters (`configs/example_gbapomdp.yaml`)
    """
    with open(solution_params_yaml) as f:
        args = yaml.safe_load(f)

    args["env"] = env_yaml_file
    main(args)


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
        # hard-coded info: this script uses "save_path" to store results into
        # here we set this variable, knowing ``n`` is unique
        c["save_path"] = n
        expansions_name = f"{n}.yaml"
        with open(expansions_name, "w") as output_file:
            yaml.dump(c, output_file, default_flow_style=False)


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("env", help="Path to YAML description of Gridverse enviornment")

    parser.add_argument(
        "--logging",
        choices=["INFO", "DEBUG", "WARNING"],
        default="INFO",
        help="Logging level, set to `WARNING` for no output, `DEBUG` for additional info",
    )

    parser.add_argument(
        "--runs", default=1, type=int, help="number of runs to average returns over"
    )

    parser.add_argument(
        "--horizon", "-H", default=1000, type=int, help="length of the problem"
    )

    parser.add_argument(
        "--episodes", default=1000, type=int, help="number of episodes to run"
    )

    parser.add_argument(
        "--gamma", default=0.95, type=float, help="discount factor to be used"
    )

    parser.add_argument(
        "--num_sims",
        default=512,
        type=int,
        help="number of simulations/iterations to run per step",
    )

    parser.add_argument(
        "--exploration", type=float, default=1, help="PO-UCT (UCB) exploration constant"
    )

    parser.add_argument(
        "--search_depth",
        "-d",
        type=int,
        default=0,
        help="The max depth of the MCTS search tree, if not set will be horizon",
    )

    parser.add_argument(
        "--pouct_evaluation",
        choices=["", "inverted_goal_distance"],
        default="",
        help="Heuristic used during MCTS. Defaults to random rollouts",
    )

    parser.add_argument(
        "--belief",
        "-B",
        help="type of belief update",
        choices=["rejection_sampling", "importance_sampling"],
        required=True,
    )

    parser.add_argument(
        "--num_particles", default=512, help="number of particles in belief", type=int
    )

    parser.add_argument(
        "--belief_minimal_sample_size",
        default=0,
        help="Threshold before resampling during importance sampling, \
                default is resampling every step",
        type=float,
    )

    # parser.add_argument(
    #     "--train_offline",
    #     choices=["on_true", "on_prior"],
    #     default="on_true",
    #     help="which, if applicable, type of learning to use",
    # )

    parser.add_argument(
        "--optimizer",
        "-O",
        default="SGD",
        type=str,
        choices=["SGD", "Adam"],
        help="The optimizer used to learn the networks",
    )

    parser.add_argument(
        "--learning_rate",
        "--alpha",
        default=1e-4,
        type=float,
        help="learning rate of the policy gradient descent",
    )

    parser.add_argument(
        "--online_learning_rate",
        "--online_alpha",
        default=1e-4,
        type=float,
        help="learning rate of the policy gradient descent",
    )

    parser.add_argument(
        "--network_size",
        help="the number of hidden nodes in the q-network",
        default=32,
        type=int,
    )

    parser.add_argument(
        "--num_nets", default=1, type=int, help="number of learned dynamic models"
    )

    parser.add_argument(
        "--batch_size", default=32, type=int, help="size of learning batch"
    )

    parser.add_argument(
        "--num_pretrain_epochs",
        default=4096,
        type=int,
        help="number of batch training offline",
    )

    parser.add_argument("--use_gpu", action="store_true", help="enables gpu usage")

    parser.add_argument(
        "--random_seed", "--seed", default=0, type=int, help="set random seed"
    )

    parser.add_argument(
        "--tensorboard_logdir",
        default="",
        type=str,
        help="the log directory for tensorboard",
    )

    # parser.add_argument(
    #     "--perturb_stdev",
    #     default=0,
    #     type=float,
    #     help="the amount of parameter pertubation applies during belief updates",
    # )

    # parser.add_argument(
    #     "--backprop",
    #     action="store_true",
    #     help="whether to apply backprop during belief updates",
    # )

    parser.add_argument("--dropout_rate", type=float, help="dropout rate", default=0)

    # parser.add_argument(
    #     "--replay_update",
    #     action="store_true",
    #     help="whether to do updates from the replay buffer during belief updates",
    # )

    parser.add_argument(
        "--save_path",
        type=str,
        default="",
        help="When given, will save results of experiments in provided folder path",
    )

    main(vars(parser.parse_args()))
