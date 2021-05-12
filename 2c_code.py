import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bisect import bisect_left, bisect_right
from scipy import stats

def run_permutation_test(X, Y, NUM_PERM):
    X_mean = np.mean(X)
    Y_mean = np.mean(Y)
    observed_diff = abs(X_mean - Y_mean)

    count = 0
    all_levels = X + Y
    for _ in range(NUM_PERM):
        np.random.shuffle(all_levels)
        g1 = all_levels[:len(X)]
        g2 = all_levels[len(X):]
        diff = abs(np.mean(g1) - np.mean(g2))
        if diff > observed_diff:
            count += 1

    return observed_diff, count, NUM_PERM, count / NUM_PERM

def plot_ecdf(S, label):
    x = np.sort(S)
    y = np.arange(len(x))/float(len(x))
    plt.step(x, y, label=label)
    return x, y

def ks_2_sample(X, Y):
    x1, y1 = plot_ecdf(X, 'UT')
    x2, y2 = plot_ecdf(Y, 'VA')

    data_all = np.concatenate([x1, x2])
    # using searchsorted solves equal data problem
    n1, n2 = len(x1), len(x2)
    idx1 = np.searchsorted(x1, data_all, side='right')
    idx2 = np.searchsorted(x2, data_all, side='right')
    cdf1 = idx1 / n1
    cdf2 = idx2 / n2
    cddiffs = cdf1 - cdf2
    minS = np.clip(-np.min(cddiffs), 0, 1)
    maxS = np.max(cddiffs)
    if minS > maxS:
        val = minS
        idx = np.argmin(cddiffs)
    else:
        val = maxS
        idx = np.argmax(cddiffs)
    # print(val)
    # print(cdf1)'
    # print(idx)
    # print(x1[idx1[idx]])
    # print(x2[idx2[idx]])
    maxdiff_x = x1[idx1[idx]]
    plt.plot((maxdiff_x,maxdiff_x),(y1[idx1[idx]],y2[idx2[idx]]), 'b--', label='Max Diff = {:.4f}'.format(val))
    plt.legend()
    plt.xlabel('Num Cases')
    plt.ylabel('eCDF')
    plt.show()

def run_part_c(X, Y):
    print(run_permutation_test(X, Y, 1000))
    print(stats.ks_2samp(X, Y))
    ks_2_sample(X, Y)

def main():
    df = pd.read_csv('23.csv',header=0, names=['date','UT_conf','VA_conf', 'UT_death', 'VA_death'])
    start_idx = df[df['date'] == '2020-09-30'].index[0]
    end_idx = df[df['date'] == '2020-12-31'].index[0] + 1
    filtered_df = df[start_idx:end_idx]
    filtered_df[['UT_conf','VA_conf', 'UT_death', 'VA_death']] = filtered_df[['UT_conf','VA_conf', 'UT_death', 'VA_death']].diff()
    filtered_df = filtered_df[1:]
    print(filtered_df)

    # print("Outliers")
    # labels = ['UT_conf','VA_conf', 'UT_death', 'VA_death']
    # for label in labels:
    #     print(label)
    #     Q1 = filtered_df[label].quantile(0.25)
    #     Q3 = filtered_df[label].quantile(0.75)
    #     IQR = Q3 - Q1
    #     small_outlier = filtered_df.query('{} < (@Q1 - 1.5 * @IQR)'.format(label))
    #     large_outlier = filtered_df.query('{} > (@Q3 + 1.5 * @IQR)'.format(label))
    #     if len(small_outlier) > 0:
    #         print('BELOW Q1')
    #         print(small_outlier[['date', label]])
    #     if len(large_outlier) > 0:
    #         print('ABOVE Q3')
    #         print(large_outlier[['date', label]])
    #     print()

    # for confirmed
    print("Confirmed:")
    X, Y = map(list, zip(*filtered_df[['UT_conf', 'VA_conf']].values))
    run_part_c(X, Y)

    # for deaths
    print("Deaths:")
    X, Y = map(list, zip(*filtered_df[['UT_death', 'VA_death']].values))
    run_part_c(X, Y)

if __name__=="__main__":
    main()

# print("Q4 Part C")
