import numpy as np
from scipy import stats


def bootstrap(x, confidence=0.95):
    # sample = np.random.choice(x, size=30, replace=False)
    bs = np.random.choice(x, (len(x), 1000), replace=True)
    bs_means = bs.mean(axis=0)
    bs_means_mean = bs_means.mean()
    minquant = (1 - confidence) / 2
    maxquant = minquant + confidence
    lower_ci = np.quantile(bs_means, minquant)
    upper_ci = np.quantile(bs_means, maxquant)
    return bs_means_mean, lower_ci, upper_ci


def t_test(x, y):
    t, p = stats.ttest_ind(x, y, equal_var=False, permutations=1000)
    return t, p
