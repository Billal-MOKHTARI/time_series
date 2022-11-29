# Model training

def time_series_train(df, growth='linear',
                          changepoints=None,
                          n_changepoints=25,
                          changepoint_range=0.8,
                          yearly_seasonality='auto',
                          weekly_seasonality='auto',
                          daily_seasonality='auto',
                          holidays=None,
                          seasonality_mode='additive',
                          seasonality_prior_scale=10.0,
                          holidays_prior_scale=10.0,
                          changepoint_prior_scale=0.05,
                          mcmc_samples=0,
                          interval_width=0.80,
                          uncertainty_samples=1000,
                          stan_backend=None):
  m = Prophet(interval_width=0.95, daily_seasonality=True)
  model = m.fit(df)

  return m
  
  
  # Making predictions
  
  """
m -> Prophet
plot -> boolean : that tells us if we'll plot tha forecasting graphic
plot_components -> boolean : that tells us if we'll plot each component graphic
periods -> integer : the number of the days we'll predict
freq -> char : forecast following Days, Months, Years, Hours, .... When we set periods to 100 with freq='M', that means we
wanna predict 100 months
"""

def time_series_predict(m, plot=True, plot_components=True, periods=100, freq='D', include_history=True):
  future = m.make_future_dataframe(periods=periods, freq=freq, include_history=include_history)
  forecast = m.predict(future)
  
  if plot :
    plot = m.plot(forecast)
  
  if plot_components :
    plot_decompose = m.plot_components(forecast)

  return forecast
