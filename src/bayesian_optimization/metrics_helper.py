import numpy as np

def get_nlpd(mu, sig2, y_true):
    """ calculate the negative log predictive density
    :param mu: mean of the predictions
    :param sig2: variance of the predictions
    :param y_true: true values
    :return nlpd: negative log predictive density"""
    nlpd = - (-0.5 * np.log(2 * np.pi) - 0.5 * np.log(sig2)
                - 0.5 * (np.square(y_true.ravel().reshape(len(y_true.ravel()), ) - mu)) / sig2)

    return nlpd

def get_squared_error(mu, y_true):
    """calcualte the squared error
    :param mu: mean of the predictions
    :param y_true: true values"""
    rmse = np.square(mu- y_true)
    return rmse

def get_regret(y, y_best_dist, target):
    """calcualate regret. Because we are trying to get as close to the target we first calculate the distance from the target. 
    :param y: y values
    :param y_best: the best (closest) y value so far
    :param target: the target for the current surface"""

    y_dist = np.abs(y - target)
    regret = y_dist - y_best_dist

    return regret
