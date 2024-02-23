import math
import os

import keras
import numpy as np
import pandas as pd

from muaddib.shaihulud_utils import write_dict_to_file


def prediction_score(test_dataset_Y, predictions, test_allocation, model_name):
    erro = test_dataset_Y - predictions
    erro_abs = np.abs(erro)
    erro_abs_sum = np.nansum(erro_abs)

    mse = np.nanmean(np.square(erro))
    rmse = np.sqrt(mse)

    erro_spain = test_dataset_Y - test_allocation
    erro_spain_abs = np.abs(erro_spain)
    erro_spain_abs_sum = np.nansum(erro_spain_abs)

    # Alocacao em falta, e alocaçao a mais
    alloc_missing = np.where(
        predictions >= test_dataset_Y, 0, test_dataset_Y - predictions
    )
    alloc_surplus = np.where(
        predictions < test_dataset_Y, 0, predictions - test_dataset_Y
    )

    spain_alloc_missing = np.where(
        test_allocation >= test_dataset_Y, 0, test_dataset_Y - test_allocation
    )
    spain_alloc_surplus = np.where(
        test_allocation < test_dataset_Y, 0, test_allocation - test_dataset_Y
    )

    # Percentagem das vezes que o modelo é melhor que o espanhol
    # Cenario optimo
    # maior ou igual a test, e menor que allocation

    # Melhor
    # optimo + aqueles que estao mais perto (erro mais pequeno)
    smaller_error = erro_abs < erro_spain_abs

    # mais que test (so quando tambem menos que alocado)
    mask_great_or_equal = predictions >= test_dataset_Y
    mask_smaller_or_equal_spain = predictions <= test_allocation

    better_allocation_mask = mask_great_or_equal & mask_smaller_or_equal_spain
    better_allocation_mask = (
        np.sum(better_allocation_mask) / test_dataset_Y.size
    )

    optimal_mask = mask_great_or_equal & smaller_error
    optimal_percentage = (np.sum(optimal_mask) / test_dataset_Y.size) * 100

    # Assumir que é prioridade ter alocado, meljor que espanha é erro menor e ter alocado,
    # better_than_spain = smaller_error & mask_great_or_equal # assim teriamos de assumir que se eu alocasse 100000000 para 100 e espanha 95 que o meu era melhor..
    beter_percentage = (np.sum(smaller_error) / test_dataset_Y.size) * 100

    benchmark_alloc_missing = np.sum(spain_alloc_missing)
    benchmark_alloc_surplus = np.sum(spain_alloc_surplus)

    alloc_missing_sum = np.sum(alloc_missing)
    alloc_surplus_sum = np.sum(alloc_surplus)

    missing_smaller = alloc_missing_sum < benchmark_alloc_missing
    surplus_smaller = alloc_surplus_sum < benchmark_alloc_surplus

    better_than_benchmark = missing_smaller & surplus_smaller

    bscore_m = (
        (benchmark_alloc_missing - np.sum(alloc_missing))
        / benchmark_alloc_missing
    ) * 100
    bscore_s = (
        (benchmark_alloc_surplus - np.sum(alloc_surplus))
        / benchmark_alloc_surplus
    ) * 100

    EPEA = ((erro_spain_abs_sum - erro_abs_sum) / erro_spain_abs_sum) * 100
    EPEA_norm = np.mean([bscore_m, bscore_s])

    bscore_m2 = bscore_m
    bscore_s2 = bscore_s
    if bscore_s < 0:
        bscore_s2 = bscore_s2 * bscore_s2 * -1
    if bscore_m < 0:
        bscore_m2 = bscore_m2 * bscore_m2 * -1
    EPEA_norm2 = np.mean([bscore_m2, bscore_s2])

    if better_than_benchmark:
        EPEA_Bench = EPEA
        EPEA_Bench_norm = EPEA_norm
    else:
        EPEA_Bench = 0
        EPEA_Bench_norm = 0

    predict_score = {
        "name": [model_name],
        "rmse": [rmse],
        "abs error": [erro_abs_sum],
        "alloc missing": [alloc_missing_sum],
        "alloc surplus": [alloc_surplus_sum],
        "EPEA_F": [bscore_m],
        "EPEA_D": [bscore_s],
        "EPEA": [EPEA],
        "EPEA_norm": [EPEA_norm],
        "EPEA_Bench": [EPEA_Bench],
        "EPEA_Bench_norm": [EPEA_Bench_norm],
        "EPEA_norm2": [EPEA_norm2],
        "optimal percentage": [optimal_percentage],
        "better percentage": [beter_percentage],
    }

    return predict_score


def save_scores(
    test_dataset_Y,
    predictions,
    test_allocation,
    model_test_filename,
    predict_score,
    model_score_filename,
    epoch=None,
):
    pred_dict = {
        "test": test_dataset_Y,
        "prediction": predictions,
        "benchmark": test_allocation,
    }
    os.makedirs(os.path.dirname(model_test_filename), exist_ok=True)
    np.savez_compressed(model_test_filename, **pred_dict)
    if epoch is not None:
        predict_score.update({"epoch": epoch})
    os.makedirs(os.path.dirname(model_score_filename), exist_ok=True)
    write_dict_to_file(predict_score, model_score_filename)

    return


def validate_model(
    model_path,
    datamanager=None,
    model_name=None,
    **kwargs,
):
    X, Y, _, _ = datamanager.validation_data(skiping_step=24)
    benchmark = datamanager.benchmark_data()

    # TODO: add a workdir as a optional arg, and also see how to do not freq saves.
    prediction_path = model_path.replace(
        "freq_saves", "freq_predictions"
    ).replace(".keras", ".npz")
    score_path = model_path.replace("freq_saves", "freq_predictions").replace(
        ".keras", ".json"
    )

    os.makedirs(os.path.dirname(score_path), exist_ok=True)
    os.makedirs(os.path.dirname(prediction_path), exist_ok=True)

    if not os.path.exists(prediction_path):
        keras.backend.clear_session()
        # Compile is false because we just need to predict.
        trained_model = keras.models.load_model(model_path, compile=False)
        predictions = trained_model.predict(X)
    else:
        predictions = np.load(prediction_path)["prediction"]

    predict_score = prediction_score(Y, predictions, benchmark, model_name)

    save_scores(
        Y,
        predictions,
        benchmark,
        prediction_path,
        predict_score,
        score_path,
        **kwargs,
    )

    return predict_score


def train_model(
    model_obj=None,
    epochs=None,
    datamanager=None,
    optimizer=None,
    loss=None,
    batch_size=None,
    weights=None,
    callbacks=None,
    training_args=None,
    **kwargs,
):
    training_args = training_args or {}
    X, Y, _, _ = datamanager.training_data(**training_args)
    keras.backend.clear_session()

    fit_args = {
        "epochs": epochs,
        "callbacks": callbacks,
        "batch_size": batch_size,
        "shuffle": True,
    }

    compile_args = {
        "optimizer": optimizer,
        "loss": loss,
    }

    if weights:
        sample_weights = np.abs(np.array(Y) - datamanager.y_mean)
        fit_args.update({"sample_weight": sample_weights})

    model_obj.compile(**compile_args)

    history_new = model_obj.fit(X, Y, **fit_args, **kwargs)
    return history_new


def result_validation(exp_results, validation_target,validate_mode="highest_stable", **kwargs):
    exp_results = exp_results.drop_duplicates(["name", "epoch"])
    exp_results = exp_results.sort_values(["name", "epoch"])
    if validate_mode=="highest_stable":
        group = exp_results.groupby("name")[validation_target].sum()
        if validation_target == "rmse":
            best_name = group.index[group.argmin()]
        else:
            best_name = group.index[group.argmax()]
        
        best_value_case = exp_results[exp_results["name"] == best_name]
        if validation_target == "rmse":
            best_value = min(best_value_case[validation_target])
        else:
            best_value = max(best_value_case[validation_target])
        best_value_case = best_value_case[best_value_case[validation_target] == best_value]

    elif validate_mode=="highest":
        if validation_target == "rmse":
            best_value = min(exp_results[validation_target])
        else:
            best_value = max(exp_results[validation_target])
        best_value_case = exp_results[exp_results[validation_target] == best_value]
    # unique_values_list = best_value_case["name"].unique().tolist()

    # best_value_case = best_value_case[best_value_case["name"].isin(unique_values_list)]
    return best_value_case, best_value


def get_ncols_nrows_figure(n_plots, max_n_cols=2):
    sqrt_n_plots = math.ceil(math.sqrt(n_plots))

    if sqrt_n_plots > max_n_cols:
        ncols = max_n_cols
    else:
        ncols = sqrt_n_plots
    nrows = math.ceil(n_plots / ncols)
    return nrows, ncols


def make_metric_plot(
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
):
    import matplotlib.pyplot as plt

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
        # Save the list of lines before plotting
        lines_before = ax.get_lines()

        # Get the label for the metric
        label = scores_df.loc[scores_df[met].idxmax(), column_to_group]

        # Get the color for the label
        color = label_color_mapping[label]

        scores_df.set_index(column_to_index).sort_index().groupby(
            column_to_group
        )[met].plot(ax=ax)

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
