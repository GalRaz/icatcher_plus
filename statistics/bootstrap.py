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


def bootstrap_ttest(x, y, confidence=0.95):
    xs = np.random.choice(x, (len(x), 1000), replace=True)
    ys = np.random.choice(y, (len(y), 1000), replace=True)
    t, p = stats.ttest_ind(xs, ys, equal_var=False, axis=0)
    t_mean = np.mean(t)
    minquant = (1 - confidence) / 2
    maxquant = minquant + confidence
    lower_ci = np.quantile(t, minquant)
    upper_ci = np.quantile(t, maxquant)
    return t_mean, lower_ci, upper_ci
