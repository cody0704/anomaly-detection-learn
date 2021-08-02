#!/usr/bin/python
#-*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from datetime import timedelta

def draw_ts(timeseries):
    timeseries.plot()
    plt.show()

def read_data(f):
        data = pd.read_csv(f)
        data = data.set_index('date')
        data.index = pd.to_datetime(data.index)
        ts = data['count']
        # draw_ts(ts)
        return ts

def diff_smooth(ts):
    dif = ts.diff().dropna()
    td = dif.describe()
    high = td['75%'] + 1.5 * (td['75%'] - td['25%'])
    low = td['25%'] - 1.5 * (td['75%'] - td['25%'])

    forbid_index = dif[(dif > high) | (dif < low)].index 
    i = 0
    while i < len(forbid_index) - 1:
        n = 1
        start = forbid_index[i]
        while forbid_index[i+n] == start + timedelta(minutes=n):
            n += 1
        i += n - 1

        end = forbid_index[i]
        value = np.linspace(ts[start - timedelta(minutes=1)], ts[end + timedelta(minutes=1)], n)
        ts[start: end] = value
        i += 1
    
    return ts

def diff_smooth(ts):
    dif = ts.diff().dropna()
    td = dif.describe()
    high = td['75%'] + 1.5 * (td['75%'] - td['25%'])
    low = td['25%'] - 1.5 * (td['75%'] - td['25%'])

    forbid_index = dif[(dif > high) | (dif < low)].index 
    i = 0
    while i < len(forbid_index) - 1:
        n = 1
        start = forbid_index[i]
        while forbid_index[i+n] == start + timedelta(minutes=n):
            n += 1
        i += n - 1

        end = forbid_index[i]
        value = np.linspace(ts[start - timedelta(minutes=1)], ts[end + timedelta(minutes=1)], n)
        ts[start: end] = value
        i += 1
    
    return ts

def add_season(seasonal,residual, train_size, pred_time_index, trend_pred):
    train_season = seasonal[:train_size]
    d = residual.describe()
    delta = d['75%'] - d['25%']
    deviaction = 1.2
    low_error, high_error = (d['75%'] - 1 * delta * deviaction, d['25%'] + 1 * delta * deviaction)
    
    values = []
    low_conf_values = []
    high_conf_values = []
    

    for i, t in enumerate(pred_time_index):
        trend_part = trend_pred[i]

        # 相同时间点的周期数据均值
        season_part = train_season[
            train_season.index.time == t.time()
            ].mean()

        # 趋势 + 周期 + 误差界限
        predict = trend_part + season_part
        low_bound = trend_part + season_part + low_error
        high_bound = trend_part + season_part + high_error

        values.append(predict)
        low_conf_values.append(low_bound)
        high_conf_values.append(high_bound)

    # 得到预测值，误差上界和下界
    final_pred = pd.Series(values, index=pred_time_index, name='predict')
    low_conf = pd.Series(low_conf_values, index=pred_time_index, name='low_conf')
    high_conf = pd.Series(high_conf_values, index=pred_time_index, name='high_conf')
    
    return final_pred, low_conf, high_conf

if __name__ == '__main__':
    filename = '/Users/cody/Documents/github/cody/ml-anomaly-detection/data/api_access_fix.csv'
    ts = read_data(filename)
    period = 1440
    order = (2, 1, 3)
    
    # Draw ts plt
    # draw_ts(ts)
    smooth = ts.copy()
    smooth = diff_smooth(smooth)
    test = ts[-period:]
    train = smooth[:len(smooth)-period]

    # STL
    decomposition = seasonal_decompose(train, period=period, two_sided=False)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    # print(seasonal)
    residual = decomposition.resid
    # decomposition.plot()
    # plt.show() 

    trend.dropna(inplace=True)
    trend_model = ARIMA(train, order).fit(disp=-1, method='css')

    # predict
    pred_time_index = pd.date_range(start=train.index[-1], periods=period+1, freq='1min')[1:]
    trend_pred = trend_model.forecast(period)[0]
    final_pred, low_conf, high_conf = add_season(seasonal, residual, period, pred_time_index, trend_pred)

    plt.subplot(211)
    # orignal data
    plt.plot(ts)
    plt.title(filename.split('.')[0])

    final_pred.plot(color='blue', label='Predict') # predicts
    test.plot(color='red', label='Original') # real data
    low_conf.plot(color='grey', label='low') # low
    high_conf.plot(color='grey', label='high') # high

    plt.legend(loc='best')
    plt.title('RMSE: %.4f' % np.sqrt(sum((final_pred.values - test.values) ** 2) / test.size))
    plt.tight_layout()
    plt.show()