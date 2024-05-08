import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.ioff()
# TODO make 3D plot on experiment with z - metric, x,y epoch and argumet
# on stats models epoch is always one, so drop that dim. but the general case has to be able to do this
# if arg is linear then its planes, if not, maybe a 2D is good enough


def get_ncols_nrows_figure(n_plots, max_n_cols=2):
    sqrt_n_plots = math.ceil(math.sqrt(n_plots))

    if sqrt_n_plots > max_n_cols:
        ncols = max_n_cols
    else:
        ncols = sqrt_n_plots
    nrows = math.ceil(n_plots / ncols)
    return nrows, ncols


def make_all_metric_plot(
    scores_df,
    metrics_to_check,
    benchmark_score=None,
    column_to_index="epoch",
    column_to_group="name",
    x_label_name=None,
    max_n_cols=2,
    figsize=(10, 10),
    folder_figures="",
    figure_name="experiment_results.png",
    limit_by="",
    shadow_plot=False,
):
    shadow_plot = scores_df[column_to_index].max() <= 1 or shadow_plot
    do_the_shadow = False

    scores_df_metrics = [
        f for f in scores_df.keys() if f not in [column_to_group]
    ]
    scores_df_metrics = [
        f for f in scores_df_metrics if f not in [column_to_index]
    ]

    ylimit_down = 0

    if isinstance(scores_df, dict):
        scores_df = pd.DataFrame(scores_df)
    non_numeric_columns = scores_df.select_dtypes(
        exclude=["int64", "float64"]
    ).columns
    scores_df_metrics = [
        f for f in scores_df_metrics if f not in non_numeric_columns
    ]

    metrics_to_check = metrics_to_check or scores_df_metrics

    n_plots = len(metrics_to_check)

    nrows, ncols = get_ncols_nrows_figure(n_plots, max_n_cols=max_n_cols)
    # Create a grid of subplots with the desired number of rows and columns
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    list_of_labels = scores_df[column_to_group].unique().tolist()
    # Get the number of unique labels
    num_labels = len(list_of_labels)

    # Create a colormap with the desired number of colors
    colormap = plt.cm.get_cmap("viridis", num_labels)

    # Get a list of colors from the colormap
    colors = [colormap(i) for i in range(num_labels)]
    handles = []
    labels = []
    label_color_mapping = dict(zip(list_of_labels, colors))

    # Iterate over each metric and plot it on a separate subplot
    for i, met in enumerate(metrics_to_check):
        ylimit = None
        ax = np.array(axes).flatten()[i]
        # # Save the list of lines before plotting
        # lines_before = ax.get_lines()

        # Get the label for the metric
        label = scores_df.loc[scores_df[met].idxmax(), column_to_group]

        # Get the color for the label
        # color = label_color_mapping[label]
        if shadow_plot:
            if met in [
                "rmse",
                "abs error",
                "alloc missing",
                "alloc surplus",
                "benchmark abs error",
                "benchmark rmse",
                "benchmark alloc missing",
                "benchmark alloc surplus",
            ]:
                scores_df.groupby(column_to_group).min()[met].plot.bar(ax=ax)
            else:
                scores_df.groupby(column_to_group).max()[met].plot.bar(ax=ax)
        else:
            scores_df.set_index(column_to_index).sort_index().groupby(
                column_to_group
            )[met].plot(ax=ax)
            do_the_shadow = True

        # Does not work for
        # scores_df.set_index(column_to_index).sort_index().groupby(column_to_group)[met].plot(ax=ax, color=color, label=label)
        # scores_df.set_index(column_to_index).sort_index().groupby(column_to_group)[met].plot(ax=ax, color=colors[i % len(colors)], label=list_of_labels[i % len(list_of_labels)])

        # Get the new lines by comparing the list of lines before and after plotting
        lines_after = ax.get_lines()
        new_lines = [
            line for line in lines_after if line.get_label() not in labels
        ]
        new_labels = [
            f.get_label() for f in lines_after if f.get_label() not in labels
        ]
        handles += new_lines
        labels += new_labels

        if limit_by == "outliers":
            # Calculate mean and standard deviation
            mean = scores_df.groupby("name")[met].mean().mean()
            std = scores_df.groupby("name")[met].std().mean()
            # Set y-axis limits
            ylimit = mean + 3 * std
            if "outliers" not in figure_name:
                figure_name = figure_name.replace(".png", "_no_outliers.png")
        if not isinstance(limit_by, str):
            ylimit = limit_by
            if "limit_by" not in figure_name:
                figure_name = figure_name.replace(
                    ".png", f"_limit_by_{limit_by}.png"
                )

        x_label_name = x_label_name or column_to_index.capitalize()
        np.array(axes).flatten()[i].set_xlabel(x_label_name)

        # TODO: future generalize this to get from some datestructure with the units and stuff
        ylabel = "MWh"
        if "perce" in met:
            ylabel = "%"
        if "mape" in met:
            ylabel = "%"
        if ylabel == "%":
            # Set y-axis limits
            ylimit = 100
            ylimit_down = 0

        np.array(axes).flatten()[i].set_ylabel(ylabel)
        np.array(axes).flatten()[i].set_title(f"{met}")

        # Add a horizontal line at the benchmark score for this metric
        if benchmark_score:
            if met in benchmark_score:
                line = (
                    np.array(axes)
                    .flatten()[i]
                    .axhline(y=benchmark_score[met], color="r", linestyle="--")
                )
                label = "Benchmark"
                if label not in labels:
                    handles.append(line)
                    labels.append(label)
                if limit_by == "benchmark":
                    ylimit = benchmark_score[met] / 0.8
                    if "benchmark" not in figure_name:
                        figure_name = figure_name.replace(
                            ".png", "_near_benchmark.png"
                        )
            elif "EPEA" in met:
                ylimit = 100
                ylimit_down = -20

        if ylimit:
            np.array(axes).flatten()[i].set_ylim([ylimit_down, ylimit])

    # a = np.array(axes).flatten()[i].legend()
    # Create the legend
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(
        os.path.join(folder_figures, figure_name),
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close()
    if do_the_shadow:
        make_all_metric_plot(
            scores_df,
            metrics_to_check,
            benchmark_score=benchmark_score,
            column_to_index=column_to_index,
            column_to_group=column_to_group,
            x_label_name=x_label_name,
            max_n_cols=max_n_cols,
            figsize=figsize,
            folder_figures=folder_figures,
            figure_name=figure_name.replace(".png", "_shadow.png"),
            limit_by=limit_by,
            shadow_plot=True,
        )


def make_metric_plot(
    scores_df,
    metric,
    benchmark_metric=None,
    figsize=(10, 10),
    column_to_index="epoch",
    column_to_group="name",
    ylimit=None,
    ylimit_down=None,
    figure_path="experiment_figure.png",
    shadow_plot=False,
):
    do_the_shadow = False
    fig, ax = plt.subplots(figsize=figsize)
    shadow_plot = scores_df[column_to_index].max() <= 1 or shadow_plot
    if (
        shadow_plot
    ):  # If epochs is not a thing, or if you want the "shadow" on the epoch axis
        # scores_df.set_index(column_to_group)[metric].plot.bar(ax=ax)
        if metric in [
            "rmse",
            "abs error",
            "alloc missing",
            "alloc surplus",
            "benchmark abs error",
            "benchmark rmse",
            "benchmark alloc missing",
            "benchmark alloc surplus",
        ]:
            scores_df.groupby(column_to_group).min()[metric].plot.bar(ax=ax)
        else:
            scores_df.groupby(column_to_group).max()[metric].plot.bar(ax=ax)
    else:
        scores_df.set_index(column_to_index).sort_index().groupby(
            column_to_group
        )[metric].plot(ax=ax)
        do_the_shadow = True
    if benchmark_metric:
        ax.axhline(
            y=scores_df[benchmark_metric].iloc[0], color="r", linestyle="--"
        )

    if ylimit:
        ax.set_ylim([ylimit_down, ylimit])

    plt.tight_layout()
    plt.savefig(
        figure_path,
        bbox_inches="tight",
    )
    plt.close()
    if do_the_shadow:
        make_metric_plot(
            scores_df,
            metric,
            benchmark_metric,
            figsize,
            column_to_index,
            column_to_group,
            ylimit,
            ylimit_down,
            figure_path.replace(".png", "_shadow.png"),
            shadow_plot=True,
        )


def make_experiment_plot(
    scores_df, folder_figures=None, metrics_to_sort=None, **kwargs
):
    metrics_to_sort = metrics_to_sort or ["name", "epoch"]
    scores_df = scores_df.sort_values(metrics_to_sort)

    all_metrics = [f for f in scores_df.columns if f not in ["name", "epoch"]]
    prediction_metrics = [f for f in all_metrics if "benchmark" not in f]
    benchmark_metrics = [f for f in all_metrics if "benchmark" in f]

    benchmark_score = {}

    for metric in prediction_metrics:

        # Get the corresponding benchmark metric
        bench_metric = None
        bench_metric = [
            f for f in benchmark_metrics if metric in f
        ]  # TODO: this might give the wrong benchmark metric
        if len(bench_metric) == 0:
            bench_metric = None
        else:
            bench_metric = bench_metric[0]
            benchmark_score[metric] = scores_df[bench_metric].iloc[0]

        figure_path = os.path.join(folder_figures, f"{metric}_results.png")

        make_metric_plot(
            scores_df, metric, bench_metric, figure_path=figure_path, **kwargs
        )

    make_all_metric_plot(
        scores_df,
        prediction_metrics,
        benchmark_score=benchmark_score,
        column_to_index="epoch",
        column_to_group="name",
        max_n_cols=2,
        figsize=(10, 10),
        folder_figures=folder_figures,
        figure_name="experiment_results.png",
        limit_by="",
    )
    make_all_metric_plot(
        scores_df,
        ["alloc missing", "alloc surplus"],
        benchmark_score=benchmark_score,
        column_to_index="epoch",
        column_to_group="name",
        max_n_cols=2,
        figsize=(10, 10),
        folder_figures=folder_figures,
        figure_name="allocs_results.png",
        limit_by="",
    )
    make_all_metric_plot(
        scores_df,
        [
            "EPEA F",
            "EPEA D",
            "EPEA",
            "EPEA norm",
            "EPEA Bench",
            "EPEA Bench norm",
        ],
        benchmark_score=benchmark_score,
        column_to_index="epoch",
        column_to_group="name",
        max_n_cols=2,
        figsize=(10, 10),
        folder_figures=folder_figures,
        figure_name="EPEAS.png",
        limit_by="",
    )
