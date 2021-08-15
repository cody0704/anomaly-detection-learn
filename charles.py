import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    test = ts[-period:]

    train = ts[:-period]
    up = train.max()
    low = train.min()

    up_bound = []
    low_bound = []

    up_bound.append(up)
    low_bound.append(low)

    for i in range(len(test)):
        if up < test[i]:
            up = test[i]
        elif low > test[i]:
            low = test[i]
        
        if len(test) != i+1:
            train = ts[i:-period+i]
            up = train.max()
            low = train.min()
            up_bound.append(up)
            low_bound.append(low)

    up_bound = pd.Series(up_bound, index=test.index, name='up')
    low_bound = pd.Series(low_bound, index=test.index, name='low')

    plt.plot(ts, linewidth=0.5, label='original')
    plt.fill_between(test.index,low_bound, up_bound, alpha=0.5)
    plt.xlabel('time')


    plt.legend(loc='best')
    plt.tight_layout()
    # plt.fill_between(low.index,low, up, alpha=0.5)
    plt.show()