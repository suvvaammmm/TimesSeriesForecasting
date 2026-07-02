import numpy as np
import pandas as pd


def backtest(series, model_func, test_size=0.2):

    n = len(series)

    if n < 20:
        return None, None, None

    split = int(n * (1 - test_size))

    train = series.iloc[:split]
    test = series.iloc[split:]

    try:
        forecast, _, _, _, _, _ = model_func(train)
    except:
        return None, None, None

    forecast = forecast[:len(test)]

    forecast = pd.Series(forecast, index=test.index)

    rmse = np.sqrt(np.mean((test - forecast) ** 2))

    mape = np.mean(
        np.abs((test - forecast) / test.replace(0, np.nan))
    ) * 100

    actual_direction = np.sign(test.diff().dropna())
    forecast_direction = np.sign(forecast.diff().dropna())

    direction_accuracy = (
        (actual_direction == forecast_direction).mean() * 100
    )

    return rmse, mape, direction_accuracy


def rolling_backtest(series, model_func, test_size=0.2):

    n = len(series)

    if n < 15:
        return None, None, None, None

    split = int(n * 0.7)

    predictions = []
    actuals = []

    # Reduced from 30 iterations to 5 for Render Free
    max_iterations = 5

    for i in range(split, min(split + max_iterations, n - 1)):

        train = series.iloc[:i]
        next_price = series.iloc[i + 1]

        try:
            forecast, _, _, _, _, _ = model_func(train)

            if forecast is None or len(forecast) == 0:
                continue

            predictions.append(float(forecast[0]))
            actuals.append(float(next_price))

        except Exception:
            continue

    if len(predictions) < 2:
        return None, None, None, None

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

    direction_accuracy = (
        np.mean(
            np.sign(np.diff(actuals)) ==
            np.sign(np.diff(predictions))
        ) * 100
    )

    return rmse, mape, direction_accuracy, predictions.tolist()