"""Visualizing and comparing results from ``online_planning``"""

from typing import Any, Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

params = [
    "ucb_constant",
    "simulations",
    "rollout_depth",
    "particles",
    "pouct_evaluation",
    "discount_factor",
]


def compare_online_planning_return(
    experiments: Iterable[Tuple[str, Dict[str, Any], pd.DataFrame]]
):

    # prepare dataframe
    timestep_dfs = list(experiments)

    for f, args, df in timestep_dfs:
        df["label"] = f

        for o in params:
            df[o] = args[o]

    df = pd.concat(df[2] for df in timestep_dfs)

    # compute sums over episodes, grouped by parameters
    episodic_sums = df.groupby(["run"] + params).sum().reset_index()

    # plot
    sns.catplot(
        data=episodic_sums,
        kind="boxen",
        y="reward",
        x="simulations",
        # style="rollout_depth",
        # size="particles",
        hue="ucb_constant",
        col="rollout_depth",
        row="particles",
    )

    plt.show()
