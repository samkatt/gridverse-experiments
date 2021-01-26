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

    python gridverse_experiments/gba_pomdp.py -D tiger --episodes 10 -H 30 \
            --expl 100 --num_sims 4096 --num_part 1024 -B rejection_sampling \
            --num_pretrain 4096 --alpha .1 --train on_true --num_nets 1 \
            --logging DEBUG

    # from experiments -- checking tensorboard logging
    python ../gridverse_experiments/gba_pomdp.py -D tiger --episodes 100 -H 30 \
            --expl 100 --num_sims 4096 --num_part 1024 -B importance_sampling \
            --num_pretrain 4096 --alpha .1 --train on_prior --prior_certainty 10 \
            --num_nets 20 --prior_correct 0 --online_learning_rate .01 --backprop \
            --belief_minimal 32 --tensor test_learning_unique_32resample --logging DEBUG

Otherwise use as a library and provide YAML files to

.. autofunction:: run_from_yaml
   :noindex:

"""

import logging
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from collections import deque
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, List, Tuple

import general_bayes_adaptive_pomdps.pytorch_api
import numpy as np
import online_pomdp_planning.types as planner_types
import pandas as pd
import pomdp_belief_tracking.types as belief_types
import yaml
from general_bayes_adaptive_pomdps.agents.neural_networks.neural_pomdps import (
    DynamicsModel,
)
from general_bayes_adaptive_pomdps.agents.planning.pouct import (
    RolloutPolicy,
    random_policy,
)
from general_bayes_adaptive_pomdps.analysis.augmented_beliefs import analyzer_factory
from general_bayes_adaptive_pomdps.domains import (
    EncodeType,
    create_environment,
    create_prior,
    gridverse_domain,
)
from general_bayes_adaptive_pomdps.environments import Environment, Simulator
from general_bayes_adaptive_pomdps.misc import set_random_seed
from general_bayes_adaptive_pomdps.models.baddr import (
    BADDr,
    create_transition_sampler,
    train_from_samples,
)
from online_pomdp_planning.mcts import Policy
from online_pomdp_planning.mcts import create_POUCT as lib_create_POUCT
from pomdp_belief_tracking.pf import importance_sampling as IS
from pomdp_belief_tracking.pf import particle_filter as PF
from pomdp_belief_tracking.pf import rejection_sampling as RS

from gridverse_experiments import conf, utils


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

    # TODO: improve API: give device to `BADDr`
    general_bayes_adaptive_pomdps.pytorch_api.set_device(conf["use_gpu"])

    # TODO: allow seeds in `pomdp_belief_tracking` and `online_pomdp_planning`
    if conf["random_seed"]:
        set_random_seed(conf["random_seed"])

    # setup
    env = create_environment(
        conf["domain"],
        conf["domain_size"],
        EncodeType.DEFAULT,
        conf["domain_description"],
    )
    assert isinstance(env, Simulator)

    baddr = BADDr(env, conf=Namespace(**conf))

    planner = create_planner(
        baddr,
        create_rollout_policy(env, conf["rollout_policy"]),
        conf["num_sims"],
        conf["exploration"],
        conf["search_depth"],
        conf["gamma"],
    )
    belief = create_belief(
        baddr, conf["belief"], conf["num_particles"], conf["belief_minimal_sample_size"]
    )
    belief_analyzer = analyzer_factory(conf["domain"], conf["domain_size"])
    train_method = create_train_method(env, conf)

    def set_domain_state(s: BADDr.AugmentedState):
        """sets domain state in ``s`` to sampled initial state """
        return BADDr.AugmentedState(baddr.sample_domain_start_state(), s.model)

    output: List[Dict[str, Any]] = []

    for run in range(conf["runs"]):

        if conf["tensorboard_logdir"]:
            general_bayes_adaptive_pomdps.pytorch_api.set_tensorboard_logging(
                f"{conf['tensorboard_logdir']}-{run}"
            )

        # TODO: refactor at some point
        baddr.reset(train_method, conf["learning_rate"], conf["online_learning_rate"])

        avg_recent_return = deque([], 50)

        for episode in range(conf["episodes"]):

            env.reset()

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

            if general_bayes_adaptive_pomdps.pytorch_api.tensorboard_logging():
                for tag, val in belief_analyzer(belief.distribution):  # type: ignore
                    general_bayes_adaptive_pomdps.pytorch_api.log_tensorboard(
                        tag, val, episode
                    )

            if general_bayes_adaptive_pomdps.pytorch_api.tensorboard_logging():
                general_bayes_adaptive_pomdps.pytorch_api.log_tensorboard(
                    "return", discounted_return, episode
                )

    if "save_path" in conf and conf["save_path"]:
        with open(os.path.join(conf["save_path"], "params.yaml"), "w") as outfile:
            yaml.dump(conf, outfile, default_flow_style=False)
        # shutil.copyfile(conf["env"], os.path.join(conf["save_path"], "env.yaml"))
        pd.DataFrame(output).to_pickle(
            os.path.join(conf["save_path"], "timestep_data.pkl")
        )


def create_train_method(
    env: Simulator, conf: Dict[str, Any]
) -> Callable[[DynamicsModel], None]:
    """creates a model training method

    This returns a function that can be called on any
    `general_bayes_adaptive_pomdps.agents.neural_networks.neural_pomdps.DynamicsModel` net to be
    trained

    :param env:
    :param conf:
    """
    logger = logging.getLogger("train method")

    # select train_on_true versus train_on_prior
    if conf["train_offline"] == "on_true":

        def sim_sampler() -> Simulator:
            return env

    elif conf["train_offline"] == "on_prior":
        sim_sampler = create_prior(
            conf["domain"],
            conf["domain_size"],
            conf["prior_certainty"],
            conf["prior_correctness"],
            EncodeType.DEFAULT,
        ).sample

    def train_method(net: DynamicsModel):
        sim = sim_sampler()
        logger.debug("Training network on %s", sim)
        sampler = create_transition_sampler(sim)
        train_from_samples(
            net, sampler, conf["num_pretrain_epochs"], conf["batch_size"]
        )

    return train_method


def run_episode(
    env: Environment,
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

    info: List[Dict[str, Any]] = []

    for timestep in range(horizon):

        # actual step
        action, planning_info = planner(belief.sample)
        step = env.step(action)

        logger.debug("A(%s) -> O(%s) --- r(%s)", action, step.observation, step.reward)

        belief_info = belief.update(action, step.observation)

        info.append(
            {
                "timestep": timestep,
                "reward": step.reward,
                "terminal": step.terminal,
                **planning_info,
                **belief_info,
            }
        )

        if step.terminal:
            break

    return info


def create_planner(
    baddr: BADDr,
    rollout_policy: Policy,
    num_sims: int = 500,
    exploration_constant: float = 1.0,
    planning_horizon: int = 10,
    discount: float = 0.95,
) -> planner_types.Planner:
    """The factory function for planners

    Currently just returns PO-UCT with given parameters, but allows for future generalization

    Real `env` is used for the rollout policy

    :param baddr:
    :param rollout_policy: the rollout policy
    :param num_sims: number of simulations to run
    :param exploration_constant: the UCB-constant for UCB
    :param planning_horizon: how far into the future to plan for
    :param discount: the discount factor to plan for
    """

    actions = list(np.int64(i) for i in range(baddr.action_space.n))
    online_planning_sim = SimForPlanning(baddr)

    return lib_create_POUCT(
        actions,
        online_planning_sim,
        num_sims,
        policy=rollout_policy,
        discount_factor=discount,
        rollout_depth=planning_horizon,
        ucb_constant=exploration_constant,
    )


def create_belief(
    gbapomdp: BADDr,
    belief: str,
    num_particles: int,
    minimal_sample_size: float,
) -> belief_types.Belief:
    """Creates the belief update

    Dispatches to :func:`create_rejection_sampling` or
    :func:`create_importance_sampling` to create the actual belief update

    :param gbapomdp: the GBA-POMDP to update the belief for
    :param belief: configuration name ("importance_sampling" or "rejection_sampling")
    :param num_particles: number of particles
    :param minimal_sample_size: threshold before resampling
    """
    if belief == "rejection_sampling":
        bu = create_rejection_sampling(gbapomdp, num_particles)
    elif belief == "importance_sampling":
        bu = create_importance_sampling(gbapomdp, num_particles, minimal_sample_size)
    else:
        raise ValueError(f"{belief} not accepted belief configuration")

    return belief_types.Belief(gbapomdp.sample_start_state, bu)


def create_rejection_sampling(
    baddr: BADDr, num_samples: int
) -> belief_types.BeliefUpdate:
    """Creates a rejection-sampling belief update

    Returns a rejection sampling belief update that tracks ``num_samples``
    particles in the ``baddr``. Basically glue between
    ``general_bayes_adaptive_pomdps`` and ``pomdp_belief_tracking``.

    Uses ``baddr`` to simulate and reject steps.

    :param baddr: the GBA-POMDP to track belief for
    :param num_samples: number of particles to main
    """

    def process_acpt(ss, ctx, _):
        # update the parameters of the augmented state
        copy = deepcopy(ss)
        baddr.update_theta(
            copy.model,
            ctx["state"].domain_state,
            ctx["action"],
            copy.domain_state,
            ctx["observation"],
        )
        return copy

    def belief_sim(s: np.ndarray, a: int) -> Tuple[np.ndarray, np.ndarray]:
        out = baddr.simulation_step_without_updating_theta(s, a)
        return out.state, out.observation

    return RS.create_rejection_sampling(
        belief_sim, num_samples, np.array_equal, process_acpt
    )


def create_importance_sampling(
    baddr: BADDr, num_samples: int, minimal_sample_size: float
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
        return baddr.simulation_step(s, a).state

    def obs_model(s, a, ss, o) -> float:
        o_model = ss.model.observation_model(s.domain_state, a, ss.domain_state)
        return np.prod([distr[feature] for distr, feature in zip(o_model, o)])

    resample_condition = partial(IS.ineffective_sample_size, minimal_sample_size)

    return IS.create_sequential_importance_sampling(
        resample_condition, transition_func, obs_model, num_samples
    )


class RolloutPolicyForPlanning(Policy):
    """A policy for ``online_pomdp_planning`` from ``general_bayes_adaptive_pomdps`` policies"""

    def __init__(self, pol: RolloutPolicy):
        """Wraps and calls ``pol`` with imposed signature

        :param pol:
        """
        super().__init__()
        self._rollout_pol = pol

    def __call__(self, s: np.ndarray, _: np.ndarray) -> int:
        """The signature for the policy for online planning

        A stochastic mapping from state and observation to action

        :param s: the state
        :param _: the observation, ignored
        """
        return self._rollout_pol(s)


class SimForPlanning(planner_types.Simulator):
    """A simulator for ``online_pomdp_planning`` from ``general_bayes_adaptive_pomdps``"""

    def __init__(self, bnrl_simulator: BADDr):
        """Wraps and calls ``bnrl_simulator`` with imposed signature

        :param bnrl_simulator:
        :type bnrl_simulator: BADDr
        """
        super().__init__()
        self._bnrl_sim = bnrl_simulator

    def __call__(
        self, s: np.ndarray, a: int
    ) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        """The signature for the simulator for online planning

        Upon calling, produces a transition (state, observation, reward, terminal)

        :param s: input state
        :param a: input action
        """
        next_s, obs = self._bnrl_sim.simulation_step_without_updating_theta(s, a)
        reward = self._bnrl_sim.reward(s, a, next_s)
        terminal = self._bnrl_sim.terminal(s, a, next_s)

        return next_s, obs.data.tobytes(), reward, terminal


def create_rollout_policy(domain: Simulator, rollout_descr: str) -> Policy:
    """returns, if available, a domain specific rollout policy

    Currently only supported by grid-verse environment:
        - "default" -- default "informed" rollout policy
        - "gridverse-extra" -- straight if possible, otherwise turn

    :param domain: environment
    :param rollout_descr: "default" or "gridverse-extra"
    """

    if isinstance(domain, gridverse_domain.GridverseDomain):
        if rollout_descr == "default":
            pol = partial(
                gridverse_domain.default_rollout_policy,
                encoding=domain._state_encoding,  # pylint: disable=protected-access
            )
        elif rollout_descr == "gridverse-extra":
            pol = partial(
                gridverse_domain.straight_or_turn_policy,
                encoding=domain._state_encoding,  # pylint: disable=protected-access
            )
    else:

        if rollout_descr:
            raise ValueError(
                f"{rollout_descr} not accepted as rollout policy for domain {domain}"
            )

        pol = partial(random_policy, action_space=domain.action_space)

    def rollout(augmented_state: BADDr.AugmentedState) -> int:
        """
        So normally PO-UCT expects states to be numpy arrays and everything is
        dandy, but we are planning in augmented space here in secret. So the
        typical rollout policy of the environment will not work: it does not
        expect an `AugmentedState`. So here we gently provide it the underlying
        state and all is well

        :param augmented_state:
        """
        return pol(augmented_state.domain_state)

    return RolloutPolicyForPlanning(rollout)


def run_from_yaml(solution_params_yaml: str):
    """Calls :func:`plan_online` with arguments described in YAML

    :param config_yaml: YAML path with parameters (`configs/example_gbapomdp.yaml`)
    """
    with open(solution_params_yaml) as f:
        args = yaml.safe_load(f)

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

    parser.add_argument(
        "--logging",
        choices=["INFO", "DEBUG", "WARNING"],
        default="INFO",
        help="Logging level, set to `WARNING` for no output, `DEBUG` for additional info",
    )

    parser.add_argument(
        "--domain",
        "-D",
        help="which domain to use method on",
        required=True,
        choices=[
            "tiger",
            "gridworld",
            "collision_avoidance",
            "chain",
            "road_racer",
            "gridverse",
        ],
    )

    parser.add_argument(
        "--domain_size",
        type=int,
        default=0,
        help="size of domain (gridworld is size of grid)",
    )

    parser.add_argument(
        "--domain_description",
        help="domain description, depends on domain used (currently only used by GridverseDomain)",
        default="",
        type=str,
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
        "--rollout_policy",
        type=str,
        choices=["", "default", "gridverse-extra"],
        default="",
        help="Rollout policy description; currently only applicable to gridverse,\
                which accepts 'gridverse-extra' for the extra-good rollout",
    )

    parser.add_argument(
        "--search_depth",
        "-d",
        type=int,
        default=0,
        help="The max depth of the MCTS search tree, if not set will be horizon",
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

    parser.add_argument(
        "--train_offline",
        choices=["on_true", "on_prior"],
        default="on_true",
        help="which, if applicable, type of learning to use",
    )

    parser.add_argument(
        "--prior_certainty",
        type=float,
        default=10,
        help='How "strong" the prior is, currently implemented for \
                Tiger, CollisionAvoidance & RoadRacer as number of the total counts',
    )

    parser.add_argument(
        "--prior_correctness",
        type=float,
        default=0,
        help='How "correct" the prior is, currently implemented for \
                Tiger: [0,1] -> [62.5,85] observation probability',
        metavar="[0, 1]",
    )

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

    parser.add_argument(
        "--perturb_stdev",
        default=0,
        type=float,
        help="the amount of parameter pertubation applies during belief updates",
    )

    parser.add_argument(
        "--backprop",
        action="store_true",
        help="whether to apply backprop during belief updates",
    )

    parser.add_argument("--dropout_rate", type=float, help="dropout rate", default=0)

    parser.add_argument(
        "--replay_update",
        action="store_true",
        help="whether to do updates from the replay buffer during belief updates",
    )

    parser.add_argument(
        "--freeze_model",
        type=str,
        help="What parts of the models to freeze after prior learning",
        choices=["", "T", "O"],
    )

    parser.add_argument(
        "--known_model",
        type=str,
        help="What parts of the models is known prior learning (and thus need not be learned)",
        choices=["", "T", "O"],
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="",
        help="When given, will save results of experiments in provided folder path",
    )

    main(vars(parser.parse_args()))
