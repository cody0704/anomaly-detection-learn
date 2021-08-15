import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tsmoothie.utils_func import sim_randomwalk
from tsmoothie.smoother import *

def read_data(f):
    df = pd.read_csv(f)
    df = df.set_index('date')
    df.index = pd.to_datetime(df.index)
    ts = df['count']

    return ts

if __name__ == '__main__':
    period = 1440
    filename = '/Users/cody/Documents/github/cody/ml-anomaly-detection/data/api_access_fix.csv'
    ts = read_data(filename)
    train = ts[period:]


   # generate randomwalks
    np.random.seed(33)

    # operate smoothing
    smoother = ExponentialSmoother(window_len=period, alpha=0.3)
    smoother.smooth(ts)
    tsmooth = smoother.smooth_data[0]
    tsmooth = pd.Series(tsmooth, index=ts.index[period:], name='tsmooth')
    tsmooth_data = smoother.data[0]
    tsmooth_data = pd.Series(tsmooth_data, index=ts.index[period:], name='tsmooth')

    # generate intervals
    deviation = 1.1
    low, up = smoother.get_intervals('sigma_interval')
    low = pd.Series(low[0], index=ts.index[period:], name='low')
    up = pd.Series(up[0], index=ts.index[period:], name='up')
    up = up * deviation
    low = low / deviation

    # Alarm
    alarm = train[(low > train) | (up < train)]

    # plot the first smoothed timeseries with intervals
    plt.plot(ts, linewidth=0.5, label='original')
    plt.plot(alarm, '.k', color='red', linewidth=0.8, label='alarm')
    # plt.plot(tsmooth, linewidth=1, color='blue', label='tsmmoth')
    # plt.plot(tsmooth_data, '.k', label='original')
    plt.xlabel('time')


    plt.legend(loc='best')
    plt.tight_layout()
    plt.fill_between(low.index,low, up, alpha=0.5)
    plt.show()

    # print(smoother.smooth_data)