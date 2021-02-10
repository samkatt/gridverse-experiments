"""Utility for visualizations"""

import os
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import yaml


def import_experiment_dirs(
    dirs: List[str],
    param_file_name: str,
    data_file_name: str,
) -> Iterable[Tuple[str, Dict[str, Any], pd.DataFrame]]:
    """Returns a generator of (file-name, args, dataframe) tuples

    :param dirs: list of directories to import from
    :param data_file_name: the name of the file (in ``dirs``) where parameters are stored
    :param data_file_name: the name of the file (in ``dirs``) where data is stored
    :return: (file-name, args, dfs) tuples generator
    """
    for f in dirs:
        with open(os.path.join(f, param_file_name)) as param_file:
            args = yaml.safe_load(param_file)

        yield (f, args, pd.read_pickle(os.path.join(f, data_file_name)))
