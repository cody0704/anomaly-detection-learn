#!/usr/bin/python
#-*- coding:utf-8 -*-

import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.metrics import mean_squared_error

from module import periodic_check

os.chdir(os.getcwd()+'/data')

def read_data(f):
    df = pd.read_csv(f)
    df = df.set_index('date')
    df.index = pd.to_datetime(df.index)
    ts = df['count']

    return ts

def add_season(seasonal,residual, train_size, pred_time_index, trend_pred):
    train_season = seasonal[:train_size]
    d = residual.describe()
    # deviation = 0.5
    delta = d['75%'] - d['25%']
    deviation = 0.5
    low_error, high_error = ((d['25%'] - delta) * deviation, (d['75%'] + delta) * deviation)
    
    values = []
    low_conf_values = []
    high_conf_values = []
    
    for i, t in enumerate(pred_time_index):
        trend_part = trend_pred[i]
        # same period value mean
        season_part = train_season[train_season.index.time == t.time()].mean()

        # Trend + Pred + limit
        predict = round(trend_part + season_part, 2)
        low_bound = round(trend_part + season_part + low_error, 2)
        high_bound = round(trend_part + season_part + high_error, 2)

        values.append(predict)
        low_conf_values.append(low_bound)
        high_conf_values.append(high_bound)

    # Get Pred, low, high
    # print(values)
    final_pred = pd.Series(values, index=pred_time_index, name='predict')
    low_conf = pd.Series(low_conf_values, index=pred_time_index, name='low_conf')
    high_conf = pd.Series(high_conf_values, index=pred_time_index, name='high_conf')
    
    return final_pred, low_conf, high_conf

if __name__ == '__main__':
    period = 1440
    filename = '/Users/cody/Documents/github/cody/ml-anomaly-detection/data/api_access_fix.csv'
    ts = read_data(filename)

    # Times Series Smooth
    # Check Periodic
    print("Check Periodic")
    middline = list()
    X = [i%period for i in range(0, len(ts))]
    y = ts.values
    degree = 4
    coef = np.polyfit(X, y, degree)
    for i in range(len(X)):
        value = coef[-1]
        for d in range(degree):
            value += X[i]**(degree-d) * coef[d]
        middline.append(value)

    check = periodic_check.period_check(middline, period)
    middline = pd.DataFrame(middline)
    print("Periodic:", check)
    
    smooth = ts.copy()
    if (check):
        print("Times Series Smooth")
        smooth = periodic_check.diff_smooth(smooth)
    
    # Check Stationarity
    print("Check Stationarity")
    result = adfuller(smooth.values, autolag='AIC')
    print("T-Value:", result[0])
    if (result[0]<=result[4]['1%'] or result[0]<=result[4]['5%']):
        print("Stationarity")
    else:
        print("No Stationarity")

    if (check):
        test = ts[-period:]
        train = smooth[:-period]
        # STL
        decomposition = seasonal_decompose(train, period=period, two_sided=False)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        # STL PLT
        # decomposition.plot()
        # plt.show()
        ##### START: ARIMA

        # trend data
        trend.dropna(inplace=True)

        # get good p d q
        best_order = None
        if best_order == None:
            best_score, best_order = float("inf"), None
            for p in [0, 1, 2, 4, 6 ,8]:
                for q in range(0, 3):
                    for d in range(0, 3):
                        order = (p, d, q)
                        try:
                            trend_model = ARIMA(train.values, order=order).fit(method_kwargs={"warn_convergence": False})
                            # predict
                            forecast = trend_model.forecast(period)
                            mse = mean_squared_error(test, forecast)
                            if mse < best_score:
                                best_score, best_order = mse, order
                            print('ARIMA%s MSE=%.3f' % (order,mse))
                        except:
                            continue
            print('Best ARIMA%s MSE=%.3f' % (best_order, best_score))

        trend_model = ARIMA(train.values, order=best_order).fit(method_kwargs={"warn_convergence": False})
        # predict
        pred_time_index = pd.date_range(start=train.index[-1], periods=period+1, freq='1min')[1:]
        trend_pred = trend_model.forecast(period)

        final_pred, low_conf, high_conf = add_season(seasonal, residual, period, pred_time_index, trend_pred)
        # Alarm
        test_time = test.keys()
        # for index in range(len(final_pred)):
        #     if high_conf[index] < test[index]:
        #         print("high", test_time[index], final_pred[index])
        #     elif low_conf[index] > test[index]:
        #         print("low", test_time[index], final_pred[index])

        ### Start: Plotclear
        plt.subplot(211)
        # orignal data
        plt.plot(ts)

        plt.plot(final_pred,color='blue', label='Predict')
        plt.plot(test,color='red', label='Original')
        plt.plot(low_conf,color='yellow', label='low')
        plt.plot(high_conf,color='grey', label='high')

        plt.legend(loc='best')
        plt.title('RMSE: %.4f' % np.sqrt(sum((final_pred.values - test.values) ** 2) / test.size))
        plt.tight_layout()
        plt.show()
        ### End: Plot
        ##### END: ARIMA
        