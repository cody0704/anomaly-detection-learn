#!/usr/bin/python3

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed

np.random.seed(1)
tf.random.set_seed(1)

df = pd.read_csv('/Users/cody/Documents/github/cody/ml-anomaly-detection/data/api_access_fix.csv')
df = df[['date', 'count']]
df['date'] = pd.to_datetime(df['date'])

print(df['date'].min(), df['count'].max())

print(df.head())

# Show Line Chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['date'], y=df['count'], name='api count'))
fig.update_layout(showlegend=True, title='API Count 2017-11-11 ~ 2017-11-16')
# fig.show()

# Train
train, test = df.loc[df['date'] <= '2017-11-16'], df.loc[df['date'] > '2017-11-16']
print(train.tail())

print(test.head())
print(train.shape,test.shape)

scaler = StandardScaler()
scaler = scaler.fit(train[['count']])

train['count'] = scaler.transform(train[['count']])
test['count'] = scaler.transform(test[['count']])

TIME_STEPS=60

def create_sequences(X, y, time_steps=TIME_STEPS):
    Xs, ys = [], []
    for i in range(len(X)-time_steps):
        Xs.append(X.iloc[i:(i+time_steps)].values)
        ys.append(y.iloc[i+time_steps])
    
    return np.array(Xs), np.array(ys)

X_train, y_train = create_sequences(train[['count']], train['count'])
X_test, y_test = create_sequences(test[['count']], test['count'])

print(f'Training shape: {X_train.shape}')

print(f'Testing shape: {X_test.shape}')


model = Sequential()
if os.path.isfile('model.h5'):
    # Load Model
    model = load_model('model.h5')
else:
    # 第一層 LSTM, 特徵值 64
    model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
    # Dropout層：每次更新參數的時候隨機斷開一定百分比(b)的輸入神經元連接，用於防止過擬合
    model.add(Dropout(rate=0.2))
    # RepeatVector層：RepeatVector層將輸入重複n次
    model.add(RepeatVector(X_train.shape[1]))
    # 第二層 LSTM, 特徵值 64
    model.add(LSTM(32, return_sequences=True))
    # 丟棄不好的特徵
    model.add(Dropout(rate=0.2))
    # TimeDistributed 應用於不同時間段
    # Dense 用來對上一層的神經元進行全部連接，實現特徵的非線性組合。
    # TimeDistributed + Dense 將連續特徵應用於每個時間片段上
    model.add(TimeDistributed(Dense(X_train.shape[2])))
    model.compile(optimizer='adam', loss='mae')
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1,
                        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')], shuffle=False)

    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    plt.show()

    # Save Model
    model.save("model.h5")
print(model.summary())


model.evaluate(X_test, y_test)

X_train_pred = model.predict(X_train, verbose=0)
train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

plt.hist(train_mae_loss, bins=50)
plt.xlabel('Train MAE loss')
plt.ylabel('Number of Samples');

threshold = np.max(train_mae_loss)
print(threshold, train_mae_loss)
print(f'Reconstruction error threshold: {threshold}')

X_test_pred = model.predict(X_test, verbose=0)
test_mae_loss = np.mean(np.abs(X_test_pred-X_test), axis=1)

plt.hist(test_mae_loss, bins=50)
plt.xlabel('Test MAE loss')
plt.ylabel('Number of samples');

test_score_df = pd.DataFrame(test[TIME_STEPS:])
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = threshold
test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']
test_score_df['count'] = test[TIME_STEPS:]['count']

fig = go.Figure()
fig.add_trace(go.Scatter(x=test_score_df['date'], y=test_score_df['loss'], name='Test loss'))
fig.add_trace(go.Scatter(x=test_score_df['date'], y=test_score_df['threshold'], name='Threshold'))
fig.update_layout(showlegend=True, title='Test loss vs. Threshold')
# fig.show()

anomalies = test_score_df.loc[test_score_df['anomaly'] == True]
anomalies.shape

fig = go.Figure()
fig.add_trace(go.Scatter(x=test_score_df['date'], y=scaler.inverse_transform(test_score_df['count']), name='api count'))
fig.add_trace(go.Scatter(x=anomalies['date'], y=scaler.inverse_transform(anomalies['count']), mode='markers', name='Anomaly'))
fig.update_layout(showlegend=True, title='Detected anomalies')
# fig.show()