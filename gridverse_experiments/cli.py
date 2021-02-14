"""Console script for gridverse_experiments.

Contains entry points for::

    - running experiments,
    - visualizing results, and
    - utility functions

There are two main types of experiments, `planning` and `learning`. In
`planning` the domain dynamics are given. Example usage online planning::

    gridverse_experiments online_planning \
            configs/gv_empty.4x4.deterministic_agent.yaml \
            configs/example_online_planning.yaml

In `learning` only some prior knowledge (e.g. in the form of data) is known.
Example usage general Bayes-adaptive POMDP::


    gridverse_experiments gba_pomdp \
            configs/gv_empty.4x4.deterministic_agent.yaml \
            configs/example_gba_pomdp.yaml

To visualize, indicate whether you would like to compare `planning` or
`learning` experiments. Example usage visualization ::

    gridverse_viz online_planning/gba_pomdp file1 file2 file3

This package provides some additional utility functionality to make either
running or analyzing experiments easier. To expand a `yaml` file of parameters
(with list) into one of each combination (all combinations of the parameters
        specified as lists)::

    gridverse_utils expand_parameter_file <path/to/file> gba_pomdp --tensorboard

Alternatively, to merge experiments (`episodic_data.pkl`) results into episodic data::

    gridverse_utils merge_experiments <output_file> <input_file1> <input_file2>...
"""

import argparse
import sys
from typing import Dict, List

from gridverse_experiments.gba_pomdp import (
    generate_config_expansions as gba_pomdp_generate_config_expansions,
)
from gridverse_experiments.gba_pomdp import merge_experiments
from gridverse_experiments.gba_pomdp import run_from_yaml as run_gbapomdp
from gridverse_experiments.online_planning import run_from_yaml as run_online_planning
from gridverse_experiments.visualization.utils import import_experiment_dirs
from gridverse_experiments.visualization.viz_gba_pomdp import compare_gba_pomdp_return
from gridverse_experiments.visualization.viz_online_planning import (
    compare_online_planning_return,
)


def parse_overwrites(overwrites: List[str]) -> Dict[str, str]:
    """Parse list 'key=value' into dictionary of str -> str

    :param overwrites:
    :returns: A dictionary of key -> value pairs extracted from ``overwrites``
    """
    try:
        return dict(str.split("=") for str in overwrites)
    except ValueError:
        raise ValueError(f"Unable to parse extra arguments {overwrites}")


def main():
    """Console script for gridverse_experiments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=["online_planning", "gba_pomdp"])
    parser.add_argument("env", help="YAML file describing gridverse environment")
    parser.add_argument("method_config", help="YAML file describing method parameters")

    args, overwrites = parser.parse_known_args()

    if args.method == "online_planning":
        # TODO: allow for overwrites
        run_online_planning(args.env, args.method_config)
    if args.method == "gba_pomdp":
        run_gbapomdp(args.env, args.method_config, parse_overwrites(overwrites))


def viz():
    """Console script for gridverse_viz."""
    print("vizzing..!")
    parser = argparse.ArgumentParser()
    parser.add_argument("option", choices=["online_planning", "gba_pomdp"])
    parser.add_argument(
        "files",
        nargs="+",
        help="Panda frame files to compare",
    )

    args = parser.parse_args()

    if args.option == "online_planning":
        compare_online_planning_return(
            import_experiment_dirs(args.files, "params.yaml", "timestep_data.pkl")
        )
    if args.option == "gba_pomdp":
        compare_gba_pomdp_return(
            import_experiment_dirs(args.files, "params.yaml", "episodic_data.pkl")
        )


def utils():
    """Console script for gridverze_utils"""

    parser = argparse.ArgumentParser()

    cmd_subparsers = parser.add_subparsers(dest="cmd")

    # condensing time-step data into episodic data
    merge_experiments_parser = cmd_subparsers.add_parser("merge_experiments")
    merge_experiments_parser.add_argument("save_path")
    merge_experiments_parser.add_argument("experiment_dirs", nargs="+")

    # expand params parser
    # TODO: add online_pomdp
    expand_params_parser = cmd_subparsers.add_parser("expand_parameter_file")
    expand_params_parser.add_argument("file", help="Param yaml file to expand")

    expand_method_param_subparser = expand_params_parser.add_subparsers(dest="method")
    expand_gba_pomdp_param_parser = expand_method_param_subparser.add_parser(
        "gba_pomdp"
    )

    expand_gba_pomdp_param_parser.add_argument(
        "--tensorboard",
        action="store_true",
        default=False,
        help="Enable to populate tensorboard log directory",
    )

    args = parser.parse_args()

    if args.cmd == "expand_parameter_file":
        if args.method == "gba_pomdp":
            gba_pomdp_generate_config_expansions(args.file, args.tensorboard)
    elif args.cmd == "merge_experiments":
        merge_experiments(
            args.save_path,
            import_experiment_dirs(
                args.experiment_dirs, "params.yaml", "episodic_data.pkl"
            ),
        )


if __name__ == "__main__":
    sys.exit(main())
