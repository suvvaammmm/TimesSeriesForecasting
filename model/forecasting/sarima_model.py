import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox


def run_sarima(series, forecast_steps=5):
    """
    Runs SARIMA on the log returns of a price series.

    Returns:
        forecast,
        lower_ci,
        upper_ci,
        aic,
        residual_mean,
        ljungbox_pvalue
    """

    # ---------------------------
    # Prepare data
    # ---------------------------
    series_arr = np.asarray(series, dtype=float)
    series_arr = series_arr[~np.isnan(series_arr)]

    if len(series_arr) < 20:
        raise ValueError("SARIMA requires at least 20 observations.")

    log_prices = np.log(series_arr)
    log_returns = np.diff(log_prices)

    use_seasonal = len(log_returns) >= 30
    seasonal_order = (1, 0, 1, 5) if use_seasonal else (0, 0, 0, 0)

    aic = 0.0
    residual_mean = 0.0
    lb_pvalue = 0.5

    try:

        model = SARIMAX(
            log_returns,
            order=(1, 0, 1),
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        result = model.fit(
            disp=False,
            maxiter=200
        )

        aic = float(result.aic)

        # ---------------------------
        # Forecast log returns
        # ---------------------------

        forecast_returns = np.asarray(
            result.forecast(steps=forecast_steps)
        )

        # Convert log-return forecast to price forecast

        last_log_price = log_prices[-1]

        cumulative_returns = np.cumsum(forecast_returns)

        forecast = np.exp(
            last_log_price + cumulative_returns
        )

        # ---------------------------
        # Confidence Interval
        # ---------------------------

        residuals = np.asarray(result.resid)

        residual_mean = float(np.mean(residuals))

        residual_std = float(np.std(residuals))

        # Keep CI visually reasonable (5%-15%)

        ci_percent = np.clip(
            residual_std,
            0.05,
            0.15
        )

        margin = forecast * ci_percent

        lower = forecast - margin
        upper = forecast + margin

        # ---------------------------
        # Ljung-Box
        # ---------------------------

        max_lag = max(
            1,
            min(10, len(residuals) // 3)
        )

        lb = acorr_ljungbox(
            residuals,
            lags=[max_lag],
            return_df=True
        )

        lb_pvalue = float(
            lb["lb_pvalue"].iloc[0]
        )

    except Exception as e:

        print("SARIMA ERROR:", e)

        last_price = float(series_arr[-1])

        avg_change = float(
            np.mean(np.diff(series_arr))
        )

        forecast = np.array([
            last_price + avg_change * (i + 1)
            for i in range(forecast_steps)
        ])

        margin = forecast * 0.08

        lower = forecast - margin
        upper = forecast + margin

    return (
        np.round(forecast, 2).tolist(),
        np.round(lower, 2).tolist(),
        np.round(upper, 2).tolist(),
        round(aic, 2),
        round(residual_mean, 4),
        round(lb_pvalue, 4),
    )