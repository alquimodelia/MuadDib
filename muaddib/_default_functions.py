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
        (benchmark_alloc_missing - alloc_missing_sum)
        / benchmark_alloc_missing
    ) * 100
    bscore_s = (
        (benchmark_alloc_surplus - alloc_surplus_sum)
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

def simple_prediction_score(test_dataset_Y, predictions, test_allocation, model_name, **kwargs):
    erro = test_dataset_Y - predictions
    erro_abs = np.abs(erro)
    erro_abs_sum = np.nansum(erro_abs)
    mse = np.nanmean(np.square(erro))
    rmse = np.sqrt(mse)
    predict_score = {
        "name": [model_name],
        "rmse": [rmse],
        "mean abs error": [np.nanmean(erro_abs)],}

    return predict_score

def prediction_score(test_dataset_Y, predictions, test_allocation, model_name, **kwargs):
    erro = test_dataset_Y - predictions
    erro_abs = np.abs(erro)
    erro_abs_sum = np.nansum(erro_abs)

    mse = np.nanmean(np.square(erro))
    rmse = np.sqrt(mse)

    truth_derivative = np.diff(test_dataset_Y.ravel())
    pred_derivative = np.diff(predictions.ravel())
    erro_derivative = truth_derivative-pred_derivative
    erro_abs_derivative = np.abs(erro_derivative)
    erro_abs_sum_derivative = np.nansum(erro_abs_derivative)
    mse_derivative = np.nanmean(np.square(erro_derivative))
    rmse_derivative = np.sqrt(mse_derivative)


    path_new_data = "/home/joao/Documentos/repos/currency_scraper/data/benchmark/new_data.csv"
    new_data = pd.read_csv(path_new_data)
    if test_allocation:
        benchmark_final, benchmark_edge = test_allocation

        benchmark_final_truth = new_data[new_data["status"]=="new"]["EUR"].values
        benchmark_final = benchmark_final.ravel()
        min_final_len = min([len(benchmark_final_truth), len(benchmark_final)])

        erro_final = benchmark_final_truth[:min_final_len] - benchmark_final[:min_final_len]
        
        slope_final_pred = benchmark_final[:min_final_len][-1] - benchmark_final[:min_final_len][0]
        slope_final_pred = slope_final_pred/len(benchmark_final[:min_final_len])

        slope_final_truth = benchmark_final_truth[:min_final_len][-1] - benchmark_final_truth[:min_final_len][0]
        slope_final_truth = slope_final_truth/len(benchmark_final_truth[:min_final_len])
        slope_error_final = slope_final_truth - slope_final_pred

        erro_abs_final = np.abs(erro_final)
        erro_abs_sum_final = np.nansum(erro_abs_final)
        mse_final = np.nanmean(np.square(erro_final))
        rmse_final = np.sqrt(mse_final)

        final_truth_derivative = np.diff(benchmark_final_truth[:min_final_len])
        final_pred_derivative = np.diff(benchmark_final[:min_final_len])
        final_erro_derivative = final_truth_derivative-final_pred_derivative
        final_erro_abs_derivative = np.abs(final_erro_derivative)
        final_erro_abs_sum_derivative = np.nansum(final_erro_abs_derivative)
        final_mse_derivative = np.nanmean(np.square(final_erro_derivative))
        final_rmse_derivative = np.sqrt(final_mse_derivative)
        slope_rmse_derivative_final = slope_error_final*final_rmse_derivative



    else:
        benchmark_edge = None
        rmse_final= None
        erro_abs_sum_final=None
        final_erro_abs_sum_derivative = None
        final_rmse_derivative = None
        slope_error_final=None
        slope_rmse_derivative_final=None

    if benchmark_edge is not None:
        benchmark_edge_truth = new_data["EUR"].values
        benchmark_edge=benchmark_edge.ravel()
        erro_edge = benchmark_edge_truth[-len(benchmark_edge):] - benchmark_edge
        erro_abs_edge = np.abs(erro_edge)
        erro_abs_sum_edge = np.nansum(erro_abs_edge)
        mse_edge = np.nanmean(np.square(erro_edge))
        rmse_edge = np.sqrt(mse_edge)


        edge_truth_derivative = np.diff(benchmark_edge_truth[-len(benchmark_edge):])
        edge_pred_derivative = np.diff(benchmark_edge)
        edge_erro_derivative = edge_truth_derivative-edge_pred_derivative
        edge_erro_abs_derivative = np.abs(edge_erro_derivative)
        edge_erro_abs_sum_derivative = np.nansum(edge_erro_abs_derivative)
        edge_mse_derivative = np.nanmean(np.square(edge_erro_derivative))
        edge_rmse_derivative = np.sqrt(edge_mse_derivative)

        slope_edge_truth = benchmark_edge_truth[-len(benchmark_edge):][-1] - benchmark_edge_truth[-len(benchmark_edge):][0]
        slope_edge_truth = slope_edge_truth/len(benchmark_edge_truth[-len(benchmark_edge):])

        slope_edge_pred = benchmark_edge[-1] - benchmark_edge[0]
        slope_edge_pred = slope_edge_pred/len(benchmark_edge)

        slope_error_edge = slope_edge_truth - slope_edge_pred
        slope_rmse_derivative_edge=slope_error_edge*edge_rmse_derivative

    else:
        rmse_edge = rmse_final
        erro_abs_sum_edge = erro_abs_sum_final
        edge_erro_abs_sum_derivative = final_erro_abs_sum_derivative
        edge_rmse_derivative = final_rmse_derivative
        slope_error_edge=slope_error_final
        slope_rmse_derivative_edge=slope_rmse_derivative_final

    # commun_steps_erros = erro[:commun_steps]

    predict_score = {
        "name": [model_name],
        "rmse": [rmse],
        "abs error": [erro_abs_sum],
        "erro abs sum derivative":[erro_abs_sum_derivative],
        "rmse derivative":[rmse_derivative],


        "rmse final": [rmse_final],
        "erro abs final": [erro_abs_sum_final],
        "final erro abs sum derivative":[final_erro_abs_sum_derivative],
        "final rmse derivative":[final_rmse_derivative],
        "slope error final":[slope_error_final],
        "slope rmse derivative final":[slope_rmse_derivative_final],
        
        
        "rmse edge": [rmse_edge],
        "erro abs sum edge": [erro_abs_sum_edge],
        "edge erro abs sum derivative":[edge_erro_abs_sum_derivative],
        "edge rmse derivative":[edge_rmse_derivative],
        "slope error edge":[slope_error_edge],
        "slope rmse derivative edge":[slope_rmse_derivative_edge],



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
    skiping_step=24,
    bench_args = None,
    valdiation_args=None,
    benchmark=True,
    prediction_score_fn=None,
    num_common=0,
    **kwargs,
):
    valdiation_args = valdiation_args or {}
    prediction_score_fn = prediction_score_fn or prediction_score

    X, Y, _, _ = datamanager.validation_data(skiping_step=skiping_step, **valdiation_args)

    if benchmark:
        bench_args = bench_args or {}
        benchmark = datamanager.benchmark_data(**bench_args)


    data_for_btc_validation_final = datamanager.read_data().drop(["date"], axis=1).iloc[-datamanager.X_timeseries:].values
    data_for_btc_validation_final = data_for_btc_validation_final.reshape((1, *data_for_btc_validation_final.shape))
    data_for_btc_validation_edge=None
    edge_value = datamanager.Y_timeseries - 12
    if edge_value>0:
        data_for_btc_validation_edge = datamanager.read_data().drop(["date"], axis=1).iloc[-(datamanager.X_timeseries+edge_value):-edge_value].values
        data_for_btc_validation_edge = data_for_btc_validation_edge.reshape((1, *data_for_btc_validation_edge.shape))


    # TODO: add a workdir as a optional arg, and also see how to do not freq saves.
    prediction_path = model_path.replace(
        "freq_saves", "freq_predictions"
    ).replace(".keras", ".npz")
    score_path = model_path.replace("freq_saves", "freq_predictions").replace(
        ".keras", ".json"
    )

    os.makedirs(os.path.dirname(score_path), exist_ok=True)
    os.makedirs(os.path.dirname(prediction_path), exist_ok=True)
    trained_model = keras.models.load_model(model_path, compile=False)
    if not os.path.exists(prediction_path):
        keras.backend.clear_session()
        # Compile is false because we just need to predict.
        trained_model = keras.models.load_model(model_path, compile=False)
        predictions = trained_model.predict(X)
    else:
            predictions = np.load(prediction_path)["prediction"]
    print(predictions)
    # THis is cannot be merge, its btc to damn shit
    benchmark_final = trained_model.predict(data_for_btc_validation_final)
    benchmark_edge = None
    if data_for_btc_validation_edge is not None:
        benchmark_edge = trained_model.predict(data_for_btc_validation_edge)
    benchmark = benchmark_final, benchmark_edge
    predict_score = prediction_score_fn(Y, predictions, benchmark, model_name, num_common=num_common)
    benchmark = False
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
        if weights =="mean":
            sample_weights = np.abs(np.array(Y) - datamanager.y_mean)
        else:
            sample_weights = np.arange(len(Y))+1
        if isinstance(weights, str):
            if "squared" in weights.lower():
                sample_weights = sample_weights **2
            if "cubic" in weights.lower():
                sample_weights = sample_weights **3
            if "inv" in weights.lower():
                sample_weights = np.max(sample_weights)/sample_weights
            if "norm" in weights.lower():
                sample_weights = sample_weights / np.max(sample_weights)

                
        sam = sample_weights.reshape(Y.shape[0], 1, 1)
        sample_weights = np.broadcast_to(sam, Y.shape)
        # sample_weights = sample_weights.flatten()
        fit_args.update({"sample_weight": sample_weights})

    model_obj.compile(**compile_args)
    print(model_obj.summary())
    print(X.shape)
    print(Y.shape)
    history_new = model_obj.fit(X, Y, **fit_args, **kwargs)
    return history_new


def result_validation(exp_results, validation_target,validate_mode="highest",cut_off_epoch=False, **kwargs):
    exp_results = exp_results.drop_duplicates(["name", "epoch"])
    exp_results = exp_results.sort_values(["name", "epoch"])
    min_error_var = "rmse" in validation_target
    min_error_var = min_error_var or "erro" in validation_target
    print("min_error_var", min_error_var)
    if cut_off_epoch:
        if not isinstance(cut_off_epoch, int):
            cut_off_epoch =exp_results["epoch"].max()/4
        exp_results = exp_results[exp_results["epoch"]>=cut_off_epoch]

    if validate_mode=="highest_stable":
        group = exp_results.groupby("name")[validation_target].sum()
        if min_error_var:
            best_name = group.index[group.argmin()]
        else:
            best_name = group.index[group.argmax()]
        
        best_value_case = exp_results[exp_results["name"] == best_name]
        if min_error_var:
            best_value = min(best_value_case[validation_target].abs()).item()
        else:
            best_value = max(best_value_case[validation_target].abs()).item()
        best_value_case = best_value_case[best_value_case[validation_target] == best_value]

    elif validate_mode=="highest":
        if min_error_var:
            best_index = exp_results[validation_target].abs().nsmallest(1).index
        else:
            best_index = exp_results[validation_target].nlargest(1).index
        best_value = exp_results.loc[best_index][validation_target].item()
        best_value_case = exp_results.loc[best_index]
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
        if not isinstance(limit_by, str):
            ylimit = limit_by
            if "limit_by" not in figure_name:
                figure_name = figure_name.replace(".png", f"_limit_by_{limit_by}.png")


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


class MetricsSave():
    def __init__(self,metrics_to_save=None):
        self.metrics_to_save=metrics_to_save

    def metrics_to_save_fn(self, experiment, exp_results=None,name_col="name", metrics_to_save=None, **kwargs):
        if len(exp_results)==0:
            exp_results = experiment.validate_experiment()
        unique_name = exp_results["name"].unique()

        metrics_to_save = metrics_to_save or self.metrics_to_save
        if metrics_to_save is None:
            metrics_to_save = [f for f in exp_results.columns if f not in ["name", "epoch"]]
        
        exp_metrics ={}
        for name in unique_name:
            best_case, best_result = experiment.result_validation_fn(
                exp_results[exp_results["name"]==name], experiment.validation_target, **kwargs
            )
            exp_metrics[name]={}
            for met in metrics_to_save:
                exp_metrics[name][met] = best_case.iloc[0].to_dict()[met]

        
        return exp_metrics