#!/usr/bin/python
#-*- coding:utf-8 -*-

import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.arima.model as stats
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.metrics import mean_squared_error

from module import periodic_check

os.chdir(os.getcwd()+'/data')

def draw_ts(ts):
    ts = ts[ts["Location"] == "Finchley"]
    ts["PM10"].plot()
    plt.legend(loc='best')
    plt.show()

def get_pq(ts, nlags=24, alpha=.05):
    acf_x, confint = acf(ts, nlags=24, alpha=.05, fft=False)
    acf_px, confint2 = pacf(ts, nlags=24, alpha=.05)

    confint = confint - confint.mean(1)[:, None]
    confint2 = confint2 - confint2.mean(1)[:, None]
    
    p, q = 0, 0
    for key1, x, y, z in zip(range(nlags), acf_x, confint[:,0], confint[:,1]):
        if x > y and x < z:
            q = key1
            break

    for key2, x, y, z in zip(range(nlags), acf_px, confint2[:,0], confint[:,1]):
        if x > y and x < z:
            p = key2
            break

    return p, q

def read_data(f):
        df = pd.read_csv(f)
        df = df.set_index('date')
        df.index = pd.to_datetime(df.index)
        ts = df['count']

        return ts


def add_season(seasonal,residual, train_size, pred_time_index, trend_pred):
    train_season = seasonal[:train_size]
    d = residual.describe()
    deviation = 0.5
    delta = d['50%'] - d['25%'] * deviation
    low_error, high_error = (d['25%'] - delta, d['75%'] + delta)
    
    values = []
    low_conf_values = []
    high_conf_values = []
    

    for i, t in enumerate(pred_time_index):
        trend_part = trend_pred[i]

        # same period value mean
        season_part = train_season[
            train_season.index.time == t.time()
            ].mean()

        # Trend + Pred + limit
        predict = trend_part + season_part
        low_bound = trend_part + season_part + low_error
        high_bound = trend_part + season_part + high_error
        print(predict, trend_part, season_part)
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
    print("ts len:",len(ts))

    # Times Series Smooth
    print("Step1 - Times Series Smooth")
    # smooth = periodic_check.diff_smooth(ts)
    smooth = ts
    print("smooth len:",len(smooth))
    
    # Check Stationarity
    print("Step2 - Check Stationarity")
    result = adfuller(smooth.values, autolag='AIC')
    check_stat = 1
    if (result[0]<=result[4]['1%'] or result[0]<=result[4]['5%']):
        check_stat = 0
    print("T-Value:", result[0])

    if (check_stat):
        print("Not Good, Bye")
        exit(0)


    # Check Periodic
    print("Step3 - Check Periodic")
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

    # check = periodic_check.period_check(middline, period)
    # middline = pd.DataFrame(middline)
    # print("Periodic:", check)
    check = True
    if (check):
        # p,q = get_pq(smooth)
        # order = (p, 0, q)
        # print(order)

        # STL
        decomposition = seasonal_decompose(smooth, period=period, two_sided=False)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        # STL PLT
        # decomposition.plot()
        # plt.show()

        ##### START: ARIMA
        test = ts[-period:]

        # trend data
        trend.dropna(inplace=True)
        train = trend[:len(trend)-period]

        # get good p d q
        best_score, best_cfg = float("inf"), None
        for p in [0, 1, 2, 4, 6, 8, 10]:
            for d in range(0, 3):
                for q in range(0, 3):
                    order = (p,d,q)
                    try:
                        trend_model = stats.ARIMA(train.values, order=order).fit(method_kwargs={"warn_convergence": False})
                        trend_model_fit = trend_model

                        # predict
                        pred_time_index = pd.date_range(start=train.index[-1], periods=period+1, freq='1min')[1:]
                        forecast = trend_model_fit.get_forecast(steps=period)
                        trend_pred = forecast.summary_frame(alpha=0.5)["mean"].values
                        mse = mean_squared_error(test, trend_pred)
                        if mse < best_score:
                            best_score, best_cfg = mse, order
                        print('ARIMA%s MSE=%.3f' % (order,mse))
                    except:
                        continue
        print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
        exit(0)
        final_pred, low_conf, high_conf = add_season(seasonal, residual, period, pred_time_index, trend_pred)

        # Alarm
        # test_time = test.keys()
        # for index in range(len(final_pred)):
        #     if high_conf[index] < test[index]:
        #         print("high", test_time[index], test[index])
        #     elif low_conf[index] > test[index]:
        #         print("low", test_time[index], test[index])

        ### Start: Plotclear
        # plt.subplot(211)
        # orignal data
        # plt.plot(ts)

        # plt.plot(final_pred,color='blue', label='Predict')
        # plt.plot(test,color='red', label='Original')
        # print(test)
        # plt.plot(low_conf,color='yellow', label='low')
        # plt.plot(high_conf,color='grey', label='high')

        # plt.legend(loc='best')
        # plt.title('RMSE: %.4f' % np.sqrt(sum((final_pred.values - test.values) ** 2) / test.size))
        # plt.tight_layout()
        # plt.show()
        ### End: Plot
        ##### END: ARIMA