"""Console script for gridverse_experiments.

Example usage online planning::

    gridverse_experiments online_planning \
            configs/gv_empty.4x4.deterministic_agent.yaml \
            configs/example_online_planning.yaml

    gridverse_experiments gba_pomdp \
            place_holder \
            configs/example_gba_pomdp.yaml


Example usage GBA-POMDP::



"""
import argparse
import sys

from gridverse_experiments.gba_pomdp import run_from_yaml as run_gbapomdp
from gridverse_experiments.online_planning import run_from_yaml as run_online_planning


def main():
    """Console script for gridverse_experiments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=["online_planning", "gba_pomdp"])
    parser.add_argument("env", help="YAML file describing gridverse environment")
    parser.add_argument("method_config", help="YAML file describing method parameters")

    args = parser.parse_args()

    if args.method == "online_planning":
        run_online_planning(args.env, args.method_config)
    elif args.method == "gba_pomdp":
        run_gbapomdp(args.method_config)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
