"""Console script for gridverse_experiments.

Example usage online planning::

    gridverse_experiments online_planning \
            configs/gv_empty.4x4.deterministic_agent.yaml \
            configs/example_online_planning.yaml

Example usage general Bayes-adaptive POMDP::


    gridverse_experiments gba_pomdp \
            configs/gv_empty.4x4.deterministic_agent.yaml \
            configs/example_gba_pomdp.yaml

Example usage visualization ::

    gridverse_viz online_planning file1 file2 file3

Example usage utility::

    gridverse_utils expand_parameter_file <path/to/file> gba_pomdp --tensorboard

"""
import argparse
import sys

from gridverse_experiments.gba_pomdp import (
    generate_config_expansions as gba_pomdp_generate_config_expansions,
)
from gridverse_experiments.gba_pomdp import run_from_yaml as run_gbapomdp
from gridverse_experiments.online_planning import run_from_yaml as run_online_planning
from gridverse_experiments.visualization.utils import import_experiment_dirs
from gridverse_experiments.visualization.viz_online_planning import (
    compare_online_planning_return,
)


def main():
    """Console script for gridverse_experiments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=["online_planning", "gba_pomdp"])
    parser.add_argument("env", help="YAML file describing gridverse environment")
    parser.add_argument("method_config", help="YAML file describing method parameters")

    args = parser.parse_args()

    if args.method == "online_planning":
        run_online_planning(args.env, args.method_config)
    if args.method == "gba_pomdp":
        run_gbapomdp(args.env, args.method_config)


def viz():
    """Console script for gridverse_viz."""
    print("vizzing..!")
    parser = argparse.ArgumentParser()
    parser.add_argument("option", choices=["online_planning"])
    parser.add_argument(
        "files",
        nargs="+",
        help="Panda frame files to compare",
    )

    args = parser.parse_args()

    if args.option == "online_planning":
        compare_online_planning_return(import_experiment_dirs(args.files))


def utils():
    """Console script for gridverze_utils"""

    parser = argparse.ArgumentParser()

    cmd_subparsers = parser.add_subparsers(dest="cmd")

    expand_params_parser = cmd_subparsers.add_parser("expand_parameter_file")
    expand_params_parser.add_argument("file", help="Param yaml file to expand")

    expand_method_param_subparser = expand_params_parser.add_subparsers(dest="method")
    expand_gba_pomdp_param_parser = expand_method_param_subparser.add_parser(
        "gba_pomdp"
    )
    # TODO: add online_pomdp

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


if __name__ == "__main__":
    sys.exit(main())
