# Forecasting


```python
fc = ac.Forecaster("sales", "date", horizon=14, group="store").fit(history)
print(fc.backtest_)        # rolling-origin backtest: MAE, RMSE, sMAPE, MASE vs seasonal-naive
fc.predict()               # store, date, step, forecast, lower, upper
ac.reconcile_bottom_up(fc.predict(), {"s1": ["north", "total"], "s2": ["north", "total"]},
                       group="store", time_col="date")      # forecasts that add up
```

- **Inputs:** the frequency and seasonality are inferred. It can forecast many series at once,
  and takes `known_future` inputs such as planned promotions.
- **Models:** it picks between a seasonal-naive baseline and gradient boosting on lags,
  rolling means and calendar features, whichever scores better in the backtest.
- **Intervals:** these come from the backtest errors at each step ahead. In testing, the 90%
  intervals covered 88% of later actuals.
