import numpy as np
from model.forecasting.arima_model import run_arima
from model.forecasting.sarima_model import run_sarima
from model.forecasting.ridge_model import run_ridge
from model.backtest.backtest import rolling_backtest


def normalize(value, min_val, max_val):
    if max_val == min_val:
        return 0
    return (value - min_val) / (max_val - min_val)


def select_best_model(series):
    """
    Runs ARIMA, SARIMA, and Ridge; scores each on AIC + backtest RMSE/MAPE
    + direction accuracy; returns the best model name and its data dict.
    """
    results = {}

    # ── ARIMA ────────────────────────────────────────────────────────────
    try:
        f1, l1, u1, aic1, res1, lb1 = run_arima(series)
        rmse1, mape1, dir1, _ = rolling_backtest(series, run_arima)
        results["ARIMA"] = dict(
            forecast=f1, lower=l1, upper=u1,
            aic=float(aic1 or 0), residual_mean=float(res1 or 0),
            lb_pvalue=float(lb1 or 0),
            rmse=float(rmse1 or 0), mape=float(mape1 or 0),
            direction=float(dir1 or 0),
        )
    except Exception as e:
        print("AUTO: ARIMA failed:", e)

    # ── SARIMA ───────────────────────────────────────────────────────────
    try:
        f2, l2, u2, aic2, res2, lb2 = run_sarima(series)
        # Reuse ARIMA for the SARIMA backtest pass to keep it fast
        rmse2, mape2, dir2, _ = rolling_backtest(series, run_arima)
        results["SARIMA"] = dict(
            forecast=f2, lower=l2, upper=u2,
            aic=float(aic2 or 0), residual_mean=float(res2 or 0),
            lb_pvalue=float(lb2 or 0),
            rmse=float(rmse2 or 0), mape=float(mape2 or 0),
            direction=float(dir2 or 0),
        )
    except Exception as e:
        print("AUTO: SARIMA failed:", e)

    # ── Ridge ────────────────────────────────────────────────────────────
    try:
        f3, l3, u3, aic3, res3, lb3 = run_ridge(series)
        rmse3, mape3, dir3, _ = rolling_backtest(series, run_ridge)
        results["Ridge"] = dict(
            forecast=f3, lower=l3, upper=u3,
            aic=float(aic3 or 0), residual_mean=float(res3 or 0),
            lb_pvalue=float(lb3 or 0),
            rmse=float(rmse3 or 0), mape=float(mape3 or 0),
            direction=float(dir3 or 0),
        )
    except Exception as e:
        print("AUTO: Ridge failed:", e)

    # ── Fallback: if nothing worked, run ARIMA bare ───────────────────────
    if not results:
        f, l, u, aic, res, lb = run_arima(series)
        return "ARIMA", dict(
            forecast=f, lower=l, upper=u, aic=float(aic), 
            residual_mean=float(res), lb_pvalue=float(lb),
        )

    # ── Score each model (lower = better) ────────────────────────────────
    rmses = [v["rmse"] for v in results.values()]
    mapes = [v["mape"] for v in results.values()]
    aics  = [v["aic"]  for v in results.values()]

    min_r, max_r = min(rmses), max(rmses)
    min_m, max_m = min(mapes), max(mapes)
    min_a, max_a = min(aics),  max(aics)

    for name, v in results.items():
        nr  = normalize(v["rmse"],      min_r, max_r)
        nm  = normalize(v["mape"],      min_m, max_m)
        na  = normalize(v["aic"],       min_a, max_a)
        dir_ = v["direction"] / 100.0   # normalise to [0,1]
        lb   = v["lb_pvalue"]

        # Lower score = better
        v["score"] = 0.4 * nr + 0.2 * nm + 0.1 * na - 0.2 * dir_ - 0.1 * lb

    best = min(results, key=lambda k: results[k]["score"])
    return best, results[best]
