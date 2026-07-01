import numpy as np
from model.forecasting.arima_model import run_arima
from model.forecasting.sarima_model import run_sarima
from model.forecasting.ridge_model import run_ridge
from model.backtest.backtest import rolling_backtest


def normalize(value, min_val, max_val):
    if max_val - min_val == 0:
        return 0
    return (value - min_val) / (max_val - min_val)

def select_best_model(series):

    try:
        f1, l1, u1, aic1, res1, lb1 = run_arima(series)
    except:
        aic1 = float("inf")

    try:
        f2, l2, u2, aic2, res2, lb2 = run_sarima(series)
    except:
        aic2 = float("inf")

    if aic1 <= aic2:
        return "ARIMA", {
            "forecast": f1,
            "lower": l1,
            "upper": u1,
            "aic": aic1,
            "residual_mean": res1,
            "lb_pvalue": lb1,
        }

    return "SARIMA", {
        "forecast": f2,
        "lower": l2,
        "upper": u2,
        "aic": aic2,
        "residual_mean": res2,
        "lb_pvalue": lb2,
    }

    # SARIMA
    f2, l2, u2, aic2, res2, lb2 = run_sarima(series)
    rmse2, mape2, dir2, _ = rolling_backtest(series, run_sarima)

    results["SARIMA"] = {
        "forecast": f2,
        "lower": l2,
        "upper": u2,
        "aic": aic2,
        "residual_mean": res2,
        "lb_pvalue": lb2,
        "rmse": rmse2,
        "mape": mape2,
        "direction": dir2
    }

    # Ridge Regression
    try:
        f3, l3, u3, aic3, res3, lb3 = run_ridge(series)
        rmse3, mape3, dir3, _ = rolling_backtest(series, run_ridge)

        results["Ridge"] = {
            "forecast": f3,
            "lower": l3,
            "upper": u3,
            "aic": aic3,
            "residual_mean": res3,
            "lb_pvalue": lb3,
            "rmse": rmse3,
            "mape": mape3,
            "direction": dir3
        }
    except Exception as e:
        print("Ridge skipped:", e)

    # Remove models that failed
    results = {k: v for k, v in results.items() if v["rmse"] is not None}

    if not results:
        return "ARIMA", results.get("ARIMA")

    # Normalization
    rmses = [v["rmse"] for v in results.values()]
    mapes = [v["mape"] for v in results.values()]
    aics = [v["aic"] for v in results.values()]

    min_rmse, max_rmse = min(rmses), max(rmses)
    min_mape, max_mape = min(mapes), max(mapes)
    min_aic, max_aic = min(aics), max(aics)

    for m in results:

        norm_rmse = normalize(results[m]["rmse"], min_rmse, max_rmse)
        norm_mape = normalize(results[m]["mape"], min_mape, max_mape)
        norm_aic = normalize(results[m]["aic"], min_aic, max_aic)

        direction = results[m]["direction"] or 0
        lb = results[m]["lb_pvalue"] or 0

        score = (
            0.4 * norm_rmse +
            0.2 * norm_mape +
            0.1 * norm_aic -
            0.2 * direction -
            0.1 * lb
        )

        results[m]["score"] = score

    best_model = min(results, key=lambda x: results[x]["score"])

    return best_model, results[best_model]