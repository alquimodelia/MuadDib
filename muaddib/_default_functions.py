import json
import os

import keras
import numpy as np


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
    with open(model_score_filename, "w") as mfile:
        json.dump(predict_score, mfile)

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
    **kwargs,
):
    X, Y, _, _ = datamanager.training_data()
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


def result_validation(exp_results, validation_target, **kwargs):
    exp_results = exp_results.drop_duplicates(["name", "epoch"])
    exp_results = exp_results.sort_values(["name", "epoch"])

    best_value = max(exp_results[validation_target])
    best_value_case = exp_results[exp_results[validation_target] == best_value]
    # unique_values_list = best_value_case["name"].unique().tolist()

    # best_value_case = best_value_case[best_value_case["name"].isin(unique_values_list)]
    return best_value_case, best_value
