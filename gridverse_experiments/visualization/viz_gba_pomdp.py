"""Visualizing and comparing results from ``gba_pomdp``

Can be called via the command line through::

    gridverse_viz online_planning/gba_pomdp file1 file2 file3

"""

from typing import Any, Dict, Iterable, Tuple

import pandas as pd
import seaborn as sns


def plot_dataframe(df: pd.DataFrame) -> None:
    sns.lineplot(
        data=df,
        y="discounted_return",
        x="episode",
        hue="label",
    )


def gba_pomdp_returns_to_pd(
    experiments: Iterable[Tuple[str, Dict[str, Any], pd.DataFrame]]
) -> pd.DataFrame:
    timestep_dfs = list(experiments)

    for f, _, df in timestep_dfs:
        df["label"] = f

    return pd.concat(df[2] for df in timestep_dfs)
