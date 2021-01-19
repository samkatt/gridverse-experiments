"""Utility functions for configuration """

from itertools import product
from typing import Any, Dict


def expand_conf(
    conf: Dict[str, Any], sep="~~~", pair_sep="-"
) -> Dict[str, Dict[str, Any]]:
    """Expands ``conf``, with list elements, in all possible sets of configs

    Takes the configurations (keys in ``conf``), assumes entries that are
    ``List`` type contain multiple possible values, and generates a config for
    each possible assignment. The assignments are returned as a dictionary,
    where the key describes the unique name of the configuration.

    For example::

        $ expand_conf( {"conf_a": 5, "setting_b": 2} )
        { "": {"conf_a": 5, "setting_b": 2} }

        $ expand_conf( {"conf_a": 5, "setting_b": [2, 3]} )
        {
                "setting_b-2": {"conf_a": 5, "setting_b": 2},
                "setting_b-3": {"conf_a": 5, "setting_b": 3}
        }

        $ expand_conf( {"conf_a": [5, "bla"], "setting_b": [2, 3]} )
        [
                "conf_a-5~~~setting_b-2": {"conf_a": 5, "setting_b": 2},
                "conf_a-5~~~setting_b-3": {"conf_a": 5, "setting_b": 3},
                "conf_a-bla~~~setting_b-2": {"conf_a": "bla", "setting_b": 2},
                "conf_a-bla~~~setting_b-3": {"conf_a": "bla", "setting_b": 3},
        ]

    This can easily be used to expand existing YAML files into multiple,
    assuming a YAML file with list inputs at ``yaml_path``::

        $ with open(yaml_path, "r") as input_file:
        $   conf = yaml.safe_load(input_file)
        $ expansions = expand_conf(conf)
        $ for n, c in expansions.items():
        $   expansions_name = "expanded_config--" + n + ".yaml"
        $   with open(expansions_name, "w") as output_file:
        $       yaml.dump(c, output_file, default_flow_style=False)

    This will result in a bunch of YAML files called "expanded_config--..."

    :conf: configurations where list values are considered multiple assignments
    :param sep: the string to concatenate each setting with in the names
    :param pair_sep: the string to seperate key-values with in the names
    :returns: A dictionary of name -> config mapping, expanded from incoming ``config``

    """
    static_config = {k: v for k, v in conf.items() if not isinstance(v, list)}
    expanding_config = {k: v for k, v in conf.items() if isinstance(v, list)}
    expanding_labels = expanding_config.keys()
    expanding_values = expanding_config.values()

    # special case - nothing to expand - not sure why you would call this
    if len(expanding_config) == 0:
        return {"": static_config}

    output = {}
    for assignment in product(*expanding_values):
        assigned_conf = dict(zip(expanding_labels, assignment))
        name = sep.join([k + pair_sep + str(v) for k, v in assigned_conf.items()])
        output[name] = {**static_config, **assigned_conf}

    return output
