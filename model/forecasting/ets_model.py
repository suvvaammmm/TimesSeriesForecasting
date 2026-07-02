import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.stats.diagnostic import acorr_ljungbox


def run_ets(series: pd.Series, steps: int = 5):
    """
    ETS on raw price LEVELS — not log-transformed.
    Uses AIC-based selection across additive/multiplicative × damped/undamped.
    CI widens with horizon (√h scaling) to reflect compounding uncertainty.
    This is visually distinct from ARIMA (log-price) and SARIMA (log-returns).
    """
    series = pd.Series(series).astype(float).dropna().reset_index(drop=True)
    n = len(series)

    if n < 10:
        raise ValueError("ETS needs at least 10 data points")

    all_positive = bool((series > 0).all())

    # ── Candidate configs: (trend, damped, error_type)
    # Damped additive/multiplicative trend = hallmark of ETS vs ARIMA
    candidates = [
        ("add", True,  "add"),    # Holt damped additive  ← typically best for prices
        ("add", False, "add"),    # Holt linear additive
        (None,  False, "add"),    # Simple exponential smoothing (SES)
    ]
    if all_positive:
        candidates = [
            ("mul", True,  "mul"),  # Multiplicative damped  ← best for volatile prices
            ("add", True,  "add"),
            ("mul", False, "mul"),
            ("add", False, "add"),
            (None,  False, "add"),
        ]

    # ── AIC selection
    best_aic    = float("inf")
    best_fitted = None

    for trend, damped, error in candidates:
        try:
            mdl    = ExponentialSmoothing(
                series,
                trend=trend,
                damped_trend=damped if trend else False,
                error=error,
                initialization_method="estimated",
            )
            fitted = mdl.fit(optimized=True, remove_bias=True)

            if fitted.aic < best_aic:
                best_aic    = fitted.aic
                best_fitted = fitted

        except Exception:
            continue

    # ── Absolute fallback
    if best_fitted is None:
        mdl         = ExponentialSmoothing(series, initialization_method="heuristic")
        best_fitted = mdl.fit()

    # ── Forecast
    fc_arr   = np.array(best_fitted.forecast(steps), dtype=float)
    forecast = fc_arr.tolist()

    # ── Widening CI: uncertainty grows as √(horizon) — distinct from ARIMA's log-space CI
    residuals = best_fitted.resid.dropna()
    resid_std = float(residuals.std()) if len(residuals) > 1 else float(series.std())
    z = 1.96

    lower_ci = [float(fc_arr[i] - z * resid_std * np.sqrt(i + 1)) for i in range(steps)]
    upper_ci = [float(fc_arr[i] + z * resid_std * np.sqrt(i + 1)) for i in range(steps)]

    # ── AIC
    aic = float(best_fitted.aic) if hasattr(best_fitted, "aic") else 0.0

    # ── Residual diagnostics
    residual_mean = float(residuals.mean()) if len(residuals) > 0 else 0.0

    lb_pvalue = 0.5
    try:
        max_lag   = max(1, min(10, len(residuals) // 3))
        lb_result = acorr_ljungbox(residuals, lags=[max_lag], return_df=True)
        lb_pvalue = float(lb_result["lb_pvalue"].iloc[0])
    except Exception:
        pass

    return forecast, lower_ci, upper_ci, aic, residual_mean, lb_pvalue