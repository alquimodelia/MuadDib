import math

import numpy as np
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import acf, adfuller, pacf

# TODO: study change to pmdarima


def get_suggestion(
    data,
    function_suggestion,
    n_returns=None,
    n_lags=None,
    number_maximas_to_study=20,
):
    n_lags = n_lags or len(data)
    acf_values_all = function_suggestion(data, nlags=n_lags)

    local_minimas = find_peaks(-acf_values_all)
    local_maximas = find_peaks(acf_values_all)
    first_minima = local_minimas[0][0]

    first_4_maxima = local_maximas[0][:4]

    maxima_to_use = first_4_maxima[np.argmax(acf_values_all[first_4_maxima])]
    maxima_to_use_on_dataset = acf_values_all[
        first_4_maxima[np.argmin(acf_values_all[first_4_maxima])]
    ]

    maximas_to_study = np.argsort(acf_values_all)[::-1][
        :number_maximas_to_study
    ]
    maxima_to_use_on_dataset = acf_values_all[
        maximas_to_study[np.argmin(acf_values_all[maximas_to_study])]
    ]
    suggested_p = np.where(acf_values_all >= maxima_to_use_on_dataset)[0]
    suggested_p = suggested_p[suggested_p > first_minima]
    sugested_ps = set(list(suggested_p))

    intial_slide = (
        acf_values_all[:first_minima] > acf_values_all[maxima_to_use]
    )
    for i in np.where(intial_slide)[0]:
        if i != 0:
            sugested_ps.add(i)
    sugested_ps = list(sugested_ps)
    if n_returns:
        best_sorted_indexes = np.argsort(acf_values_all[sugested_ps])[::-1]
        sugested_ps = np.array(sugested_ps)[best_sorted_indexes][:n_returns]

    sugested_ps = list(sugested_ps)
    sugested_ps = [int(f) for f in sugested_ps]
    return sugested_ps


def get_p_suggestion(
    data,
    n_returns=None,
    number_maximas_to_study=20,
):

    return get_suggestion(
        data,
        acf,
        n_returns=n_returns,
        number_maximas_to_study=number_maximas_to_study,
    )


def get_q_suggestion(
    data, n_returns=None, number_maximas_to_study=20, n_lags=None
):
    n_lags = n_lags or min(
        int(math.ceil(len(data) / 100)), 170
    )  # should be one month
    return get_suggestion(
        data,
        pacf,
        n_returns=n_returns,
        n_lags=n_lags,
        number_maximas_to_study=number_maximas_to_study,
    )


def is_stationary(series):
    """
    Check if the series is stationary using the Augmented Dickey-Fuller test.

    Parameters:
    - series: Pandas Series representing the time series data.

    Returns:
    - Boolean indicating whether the series is stationary.
    """
    result = adfuller(series.dropna(), autolag="AIC")
    p_value = result[1]
    return p_value < 0.05  # Assuming a significance level of 0.05


def get_d_suggestion(data):
    """
    Recursively determine the number of derivatives needed to make a time series stationary.

    Parameters:
    - data: Pandas Series representing the time series data.

    Returns:
    - Integer representing the number of derivatives needed.
    """

    d = 0
    while not is_stationary(data):
        data = data.diff().dropna()  # Apply differencing
        d += 1

    return int(d)
