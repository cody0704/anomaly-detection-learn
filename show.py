#!/usr/bin/python
#-*- coding:utf-8 -*-

import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.chdir(os.getcwd()+'/data')

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

    ### Start: Plotclear
    plt.subplot(211)
    # orignal data
    plt.plot(ts)

    # plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    ### End: Plot
    ##### END: ARIMA
        