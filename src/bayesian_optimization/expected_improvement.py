import copy
import numpy as np
from scipy.stats import ncx2, norm


class ExpectedImprovement:
    """This expected improvement class creates an acquisition function based on the target vector optimisation, except
    the objective values are scaled by constants alpha. This means we can alter which objectives are most important. If
    alphas = [1]*len(params) then we recover the original target vector optimisation"""

    def __init__(self, alphas=None):

        """:param params: list of parameters. should be either or both of 'm' and 'r'
        :param alphas: list of floats. If None, then alphas = [1]*len(params)"""

        params = ['r']

        self.params = ['r']

        if alphas is None:
            self.alpha =  1 
        else:
            self.alpha = alphas

    def filter_ys(self, ys, targets, params):
        """filter ys so all drifts (m) below the target drift are equal to the target
        :param ys: candas parameter array of the data
        :param targets: candas parameter array of the targets
        :param params: list of parameters. should be either or both of 'm' and 'r'
        :return: candas parameter array of the data with the drifts (m) below the target drift equal to the target
        """
        ys = copy.deepcopy(ys)
        if 'm' in params:
            targ_m = targets['m'].values()
            ms = ys['m'].values()
            ms[ms < targ_m] = targ_m
            ys['m'] = ms

        return ys

    def Chi_EI(self, mu, sig2, target, best_yet, k=1):
        """Expected improvement function taken from https://github.com/akuhren/target_vector_estimation
        :param mu: array of means
        :param sig2: array of variances
        :param target: array of targets
        :param best_yet: float of the best value so far
        :param k: int of the number of objectives
        :return: array of the expected improvement"""

        gamma2 = sig2.mean(axis=1)

        nc = ((target - mu) ** 2).sum(axis=1) / gamma2

        h1_nx = ncx2.cdf((best_yet / gamma2), k, nc)
        h2_nx = ncx2.cdf((best_yet / gamma2), (k + 2), nc)
        h3_nx = ncx2.cdf((best_yet / gamma2), (k + 4), nc)

        t1 = best_yet * h1_nx
        t2 = gamma2 * (k * h2_nx + nc * h3_nx)

        return t1 - t2

    def BestYet(self, ys, target):
        """Function to calculate the closest distance from the target we have observed so far
        :param ys: a series containing the values in the train set
        :param target_parrays: a 2x1 array containing the desired values for the rate and drift parameters
        :return best: the smallest distance between an observation and the target, a float"""

        ys_ = np.array(self.alpha * ys)
        target = target['Target Rate']
        target = np.vstack([self.alpha * np.atleast_2d(target)]*len(ys_)).reshape(-1, )

        assert (ys_.shape[0] == target.shape[0])
        best = ((ys_ - target) ** 2).min()
        return best

    def EI(self, preds, target, best_yet, params):
        """Expected improvement using target vector optimisation
        :param preds: a candas uncertain parameter array containing the mean and variance of the predictions
        :param target_parrays: a 2x1 array containing the desired values for the rate and drift parameters
        :param best_yet: the smallest distance between an observation and the target, a float
        :param params: list of parameters. should be either or both of 'm' and 'r'
        :return ei: the expected improvements of each prediction"""

        mu = np.array([self.alpha * preds[f'mu'].ravel()]).T
        sig2 = np.array([self.alpha ** 2 * preds[f'sig2'].ravel()]).T
        target = np.hstack([self.alpha * np.atleast_2d(target['Target Rate'])])

        k = len(params)
        ei = self.Chi_EI(mu, sig2, target, best_yet, k=k)

        return ei

    def get_error_from_optimization_target(self, df):
        """Get the error of the points so far from the optimization target.
        :param df: a pandas dataframe containing the predictions and the optimization target
        :return df: the same dataframe with an extra column containing the error from the optimization target"""
        df[f'error from optimization target z'] = np.sqrt(np.sum([(alpha * df[f'target {param} z']
                                                                   - alpha * df[f'stzd {param}']) ** 2 for param, alpha
                                                                  in
                                                                  self.alphas.items()], axis=0))
        return df

