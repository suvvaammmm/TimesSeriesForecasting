import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox


def run_sarima(series, forecast_steps: int = 5):
    # Normalise input — works whether caller passes Series or ndarray
    series_arr = np.asarray(series, dtype=float)
    series_arr = series_arr[~np.isnan(series_arr)]   # drop NaNs

    if len(series_arr) < 20:
        raise ValueError("SARIMA needs at least 20 data points")

    log_prices  = np.log(series_arr)
    log_returns = np.diff(log_prices)

    use_seasonal   = len(log_returns) >= 30
    seasonal_order = (1, 0, 1, 5) if use_seasonal else (0, 0, 0, 0)

    aic           = 0.0
    residual_mean = 0.0
    lb_pvalue     = 0.5

    try:
        model = SARIMAX(
            log_returns,
            order=(1, 0, 1),
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False, maxiter=200)

        aic         = float(result.aic)
        fc_obj      = result.get_forecast(steps=forecast_steps)
        ret_forecast = fc_obj.predicted_mean.values
        ret_ci       = fc_obj.conf_int().values

        last_log         = log_prices[-1]
        cum_ret_forecast = np.cumsum(ret_forecast)
        cum_ret_lower    = np.cumsum(ret_ci[:, 0])
        cum_ret_upper    = np.cumsum(ret_ci[:, 1])

        forecast = np.exp(last_log + cum_ret_forecast)
        lower    = np.exp(last_log + cum_ret_lower)
        upper    = np.exp(last_log + cum_ret_upper)

        residuals     = result.resid
        residual_mean = float(np.mean(residuals))
        max_lag       = max(1, min(10, len(residuals) // 3))
        lb_test       = acorr_ljungbox(residuals, lags=[max_lag], return_df=True)
        lb_pvalue     = float(lb_test["lb_pvalue"].values[0])

    except Exception as e:
        print("SARIMA ERROR:", e)
        last_price = float(series_arr[-1])
        avg_return = float(np.mean(np.diff(series_arr)))
        forecast   = np.array([last_price + avg_return * (i + 1) for i in range(forecast_steps)])
        std_val    = float(np.std(series_arr))
        lower      = forecast - 1.96 * std_val
        upper      = forecast + 1.96 * std_val

    return (
        np.round(forecast, 2).tolist(),
        np.round(lower,    2).tolist(),
        np.round(upper,    2).tolist(),
        round(float(aic),           2),
        round(float(residual_mean), 4),
        round(float(lb_pvalue),     4),
    )