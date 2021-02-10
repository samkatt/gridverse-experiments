"""Visualizing and comparing results from ``gba_pomdp``

Can be called via the command line through::

    gridverse_viz online_planning/gba_pomdp file1 file2 file3

"""

from typing import Any, Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def compare_gba_pomdp_return(
    experiments: Iterable[Tuple[str, Dict[str, Any], pd.DataFrame]]
):
    # prepare dataframe
    timestep_dfs = list(experiments)

    for f, _, df in timestep_dfs:
        df["label"] = f

    df = pd.concat(df[2] for df in timestep_dfs)

    # compute sums over episodes, grouped by parameters
    episodic_mean = df.groupby(["episode", "label"]).mean().reset_index()

    # plot
    sns.lineplot(
        data=episodic_mean,
        y="discounted_return",
        x="episode",
        # style="rollout_depth",
        # size="ucb_constant",
        hue="label",
    )

    plt.show()
