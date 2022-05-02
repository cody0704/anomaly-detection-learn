import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as datetime
from fbprophet import Prophet
from sklearn import metrics


if __name__ == '__main__':
    filename = '../data/api_access_fix.csv'
    df = pd.read_csv(filename)
    # df = df.rename(columns={'date': 'ds', 'count': 'y'})
    df = df.drop("Unnamed: 0",axis=1)
    df = df.set_index('ds')
    df.index = pd.to_datetime(df.index)

    print(df.head())

    plt.style.use('ggplot')
    df.plot(figsize=(12, 8))
    # plt.show()

    df['y'] = np.log(df['y'])
    # 定義模型
    model = Prophet()

    # 訓練模型
    model.fit(df)

    # 建構預測集
    future = model.make_future_dataframe(periods=1440) #forecasting for 1 year from now.

    # 進行預測
    forecast = model.predict(future)

    figure=model.plot(forecast)
    plt.show()