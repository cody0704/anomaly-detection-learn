import statsmodels.tsa.arima.model as stats

time_series = [2, 3.0, 5, 7, 9, 11, 13, 17, 19]
steps = 4
alpha = 0.05

model = stats.ARIMA(time_series, order=(0, 1, 0))
model_fit = model.fit()

forecast = model_fit.get_forecast(steps=steps)
forecasts_and_intervals = forecast.summary_frame(alpha=alpha)

print(forecast[0])