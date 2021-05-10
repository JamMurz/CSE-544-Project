import numpy as np
import matplotlib.pyplot as plt
import pandas
from scipy.stats import binom
from scipy.stats import geom
from scipy.stats import poisson
from scipy.stats import kstwobign
from scipy.stats import kstest


def plot_ecdf(S, label):
    x = np.sort(S)
    y = np.arange(len(x))/float(len(x))
    plt.step(x, y, label=label)
    return x, y

df = pandas.read_csv('../CSE-544-Datasets/States Data/23.csv')
df = df[(df['Date'] >= '2020-10-01') & (df['Date'] <= '2020-12-31')]

UT_confirmed = np.array(df['UT confirmed'].tolist(), dtype=np.float64)
VA_confirmed = np.array(df['VA confirmed'].tolist(), dtype=np.float64)
UT_deaths = np.array(df['UT deaths'].tolist(), dtype=np.float64)
VA_deaths = np.array(df['VA deaths'].tolist(), dtype=np.float64)

for label, UT, VA in [('Confirmed', UT_confirmed, VA_confirmed), ('Deaths', UT_deaths, VA_deaths)]:
    # Plot VA eCDF
    plt.figure(1 if label == 'Confirmed' else 4)
    plt.title(label)
    plot_ecdf(VA, '')

    poisson_cdf = lambda k: poisson.cdf(k, np.mean(UT))
    poisson_cdf = np.vectorize(poisson_cdf)
    y = poisson_cdf(VA)
    diff1 = max(np.abs(y - np.arange(len(VA))/float(len(VA))))
    diff2 = max(np.abs(y - (np.arange(len(VA)) + 1)/float(len(VA))))
    print('Manually computed KS statistic: ' + str(max(diff1, diff2)))
    print('Scipy KS test result: ' + str(kstest(VA.tolist(), poisson_cdf)))
    print()
    plt.step(VA, y)

    geom_cdf = lambda k: geom.cdf(k, 1.0 / np.mean(UT))
    geom_cdf = np.vectorize(geom_cdf)
    y = geom_cdf(VA)
    plt.step(VA, y)
    diff1 = max(np.abs(y - np.arange(len(VA))/float(len(VA))))
    diff2 = max(np.abs(y - (np.arange(len(VA)) + 1)/float(len(VA))))
    print('Manually computed KS statistic: ' + str(max(diff1, diff2)))
    print('Scipy KS test result: ' + str(kstest(VA.tolist(), geom_cdf)))
    print()
    plt.step(VA, y)

    n = 1 + np.mean(UT) - np.mean(UT ** 2) / np.mean(UT)
    p = np.mean(UT) / n
    binom_cdf = lambda k: binom.cdf(k, n, p)
    binom_cdf = np.vectorize(binom_cdf)
    y = binom_cdf(VA)
    plt.step(VA, y)
    diff1 = max(np.abs(y - np.arange(len(VA))/float(len(VA))))
    diff2 = max(np.abs(y - (np.arange(len(VA)) + 1)/float(len(VA))))
    print('Manually computed KS statistic: ' + str(max(diff1, diff2)))
    print('Scipy KS test result: ' + str(kstest(VA.tolist(), binom_cdf)))
    print()
    plt.step(VA, y)

plt.show()
