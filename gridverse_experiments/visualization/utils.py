"""Utility for visualizations"""

import os
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import yaml


def import_experiment_dirs(
    dirs: List[str],
) -> Iterable[Tuple[str, Dict[str, Any], pd.DataFrame]]:
    """Returns a generator of (file-name, args, dataframe) tuples

    :param dirs:
    :return: (file-name, args, dfs) tuples generator
    """
    for f in dirs:
        with open(os.path.join(f, "params.yaml")) as param_file:
            args = yaml.safe_load(param_file)

        yield (f, args, pd.read_pickle(os.path.join(f, "timestep_data.pkl")))
