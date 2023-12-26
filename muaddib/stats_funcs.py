import numpy as np
from scipy.optimize import curve_fit
from typing import Iterable

def get_mean_from_rolling(s, win):
    roll_mean = s.dropna().rolling(win).mean().dropna()
    if len(roll_mean)<win:
        return roll_mean.mean()
    else:
        return get_mean_from_rolling(roll_mean, win)



def cutoff(y, y_max):
    if isinstance(y, Iterable):
        r = []
        for i in y:
            r.append(np.min([i, y_max]))
    else:
        r = min(y, y_max)
    return r

def linear_function(x, m, b):
    y = m*x + b
    return cutoff(y, 100)

def exponential_function(x, a, b, c):
    y= a * np.exp(-b * x) + c
    return cutoff(y, 100)

def log_function(x, a, b):
    y= a + b * np.log(x)
    return cutoff(y, 100)


def create_polynomial_function(degree):
    def polynomial_function(x, *coeffs):
        y= np.polynomial.polynomial.Polynomial(coeffs)(x)
        return cutoff(y, 100)

    return polynomial_function

def sinusoidal_function(x, A,B,C,D):
    y= A*np.sin(B*x+C)+D
    return cutoff(y, 100)


def generalized_additive_function(x,a,b):
    y= a*x+b*np.exp(-x)
    return cutoff(y, 100)




ALL_MODELS_DICT_FUNCTION = {
    "linear_function":{"model":linear_function,
                        "po":[1,1]
                            }, 
    "exponential_function":{"model":exponential_function,
                        "po":[1,1, 1]
                            }, 
    "log_function":{"model":log_function,
                        "po":[1,1]
                            }, 
    "poly2":{"model":create_polynomial_function(2),
                        "po":[1, 1, 1]
                            },
    "poly3":{"model":create_polynomial_function(3),
                        "po":[1, 1, 1, 1]
                            },
    "poly4":{"model":create_polynomial_function(4),
                        "po":[1, 1, 1, 1, 1]
                            }, 
    "sinusoidal_function":{"model":sinusoidal_function,
                        "po":[1,1, 0, 1]
                            }, 
    "generalized_additive_function":{"model":generalized_additive_function,
                        "po":[1,1]
                            }, 

}