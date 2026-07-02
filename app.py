from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import math
import traceback

from model.anomaly.detect import detect_anomalies
from portfolio.portfolio_engine import run_portfolio_from_csv
from model.backtest.backtest import rolling_backtest
from model.forecasting.arima_model import run_arima
from model.forecasting.sarima_model import run_sarima
from model.forecasting.ridge_model import run_ridge
from model.forecasting.ets_model import run_ets
from model.forecasting.auto_selector import select_best_model

app = Flask(__name__)


# -----------------------------------------
# Safe JSON Conversion
# -----------------------------------------
def safe(x):
    if x is None:
        return 0.0
    if isinstance(x, (float, np.floating)):
        if math.isnan(x) or math.isinf(x):
            return 0.0
    return float(x)


@app.route("/")
def home():
    return render_template("index.html")


# ======================================================
# MULTI ASSET
# ======================================================
@app.route("/predict_multi_csv", methods=["POST"])
def predict_multi_csv():
    try:
        if "file" not in request.files:
            return jsonify({"error": "CSV required"}), 400
        file = request.files["file"]
        results = run_portfolio_from_csv(file)
        return jsonify(results)
    except Exception as e:
        print("MULTI ERROR:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ======================================================
# SINGLE ASSET
# ======================================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data_source  = request.form.get("data_source")
        model_type   = request.form.get("model_type")
        threshold    = float(request.form.get("threshold", 1.5))
        company_name = "Uploaded CSV"   # default – overwritten for Angel

        # ---------------------------
        # Load Data
        # ---------------------------
        if data_source == "file":
            if "file" not in request.files:
                return jsonify({"error": "CSV file required"}), 400
            file   = request.files["file"]
            df_raw = pd.read_csv(file)
            if "value" not in df_raw.columns:
                return jsonify({"error": "CSV must contain a 'value' column"}), 400
            series = pd.to_numeric(df_raw["value"], errors="coerce").dropna().reset_index(drop=True)

        elif data_source == "angel":
            from services.angel_service import get_angel_data
            symbol_token = request.form.get("symbol_token")
            if not symbol_token:
                return jsonify({"error": "Symbol token required"}), 400
            series, company_name = get_angel_data(symbol_token)
            if series is None or len(series) == 0:
                return jsonify({"error": "No data returned from Angel API"}), 400
            series = pd.Series(series).reset_index(drop=True)

        else:
            return jsonify({"error": "Invalid data source"}), 400

        df = pd.DataFrame({"value": series})

        if len(series) < 15:
            return jsonify({"error": "Need at least 15 data points"}), 400

        # ---------------------------
        # Model Selection
        # ---------------------------
        if model_type == "ARIMA":
            forecast, lower, upper, aic, residual_mean, lb_pvalue = run_arima(series)
            selected_model = "ARIMA"

        elif model_type == "SARIMA":
            forecast, lower, upper, aic, residual_mean, lb_pvalue = run_sarima(series)
            selected_model = "SARIMA"

        elif model_type == "Ridge":
            forecast, lower, upper, aic, residual_mean, lb_pvalue = run_ridge(series)
            selected_model = "Ridge"

        elif model_type == "ETS":
            forecast, lower, upper, aic, residual_mean, lb_pvalue = run_ets(series)
            selected_model = "ETS"

        elif model_type == "AUTO":
            best_model, best_data = select_best_model(series)
            forecast       = best_data["forecast"]
            lower          = best_data["lower"]
            upper          = best_data["upper"]
            aic            = best_data["aic"]
            residual_mean  = best_data["residual_mean"]
            lb_pvalue      = best_data["lb_pvalue"]
            selected_model = best_model + " (AUTO)"

        else:
            return jsonify({"error": "Invalid model type"}), 400

        # Ensure flat python lists
        forecast = list(np.array(forecast).flatten().astype(float))
        lower    = list(np.array(lower).flatten().astype(float))
        upper    = list(np.array(upper).flatten().astype(float))

        # ---------------------------
        # Rolling Backtest
        # Pick the right function for the backtest loop.
        # ETS / AUTO / SARIMA fall back to run_arima so the loop
        # finishes well within the gunicorn timeout.
        # ---------------------------
        if "Ridge" in selected_model:
            bt_func = run_ridge
        else:
            bt_func = run_arima

        bt_rmse, bt_mape, bt_dir, rolling_preds = rolling_backtest(series, bt_func)

        # ---------------------------
        # Strategy Simulation
        # ---------------------------
        split            = int(len(series) * 0.8)
        starting_capital = 100_000.0
        capital          = starting_capital
        transaction_cost = 0.001
        risk_per_trade   = 0.02

        equity  = [capital]
        returns = []

        if rolling_preds is not None and len(rolling_preds) > 0:
            rolling_preds_arr = np.array(rolling_preds)
            max_index = min(len(rolling_preds_arr), len(series) - split - 1)

            for i in range(max_index):
                current_price   = float(series.iloc[split + i])
                next_price      = float(series.iloc[split + i + 1])
                predicted_price = float(rolling_preds_arr[i])

                if predicted_price > current_price:
                    r = ((next_price - current_price) / current_price) - transaction_cost
                elif predicted_price < current_price:
                    r = ((current_price - next_price) / current_price) - transaction_cost
                else:
                    r = 0.0

                position_size = capital * risk_per_trade
                capital      += position_size * r

                equity.append(capital)
                returns.append(r)

        equity  = np.array(equity,  dtype=float)
        returns = np.array(returns, dtype=float)

        strategy_total_return = ((capital - starting_capital) / starting_capital) * 100
        num_days = len(series)
        strategy_annualized_return = (
            ((capital / starting_capital) ** (252 / num_days) - 1) * 100
            if num_days > 0 else 0.0
        )
        strategy_win_rate = (
            (float(np.sum(returns > 0)) / len(returns)) * 100
            if len(returns) > 0 else 0.0
        )
        strategy_sharpe = (
            (np.mean(returns) / np.std(returns)) * np.sqrt(252)
            if len(returns) > 1 and np.std(returns) != 0 else 0.0
        )
        strategy_max_dd = (
            np.min(equity / np.maximum.accumulate(equity) - 1) * 100
            if len(equity) > 1 else 0.0
        )
        buy_hold_return = (
            (float(series.iloc[-1]) - float(series.iloc[split]))
            / float(series.iloc[split])
        ) * 100

        # ---------------------------
        # Signal Engine
        # ---------------------------
        current_price = float(series.iloc[-1])
        forecast_mean = float(np.mean(forecast[:3]))
        lower_first   = float(lower[0])
        upper_first   = float(upper[0])

        expected_return = (forecast_mean - current_price) / current_price

        vol_series = series.pct_change().rolling(20).std()
        volatility = float(vol_series.iloc[-1]) if not pd.isna(vol_series.iloc[-1]) else 0.0
        regime = "NORMAL"
        if volatility > 0.03:
            regime = "HIGH VOLATILITY"
        elif volatility < 0.01:
            regime = "LOW VOLATILITY"

        ci_width         = upper_first - lower_first
        confidence_score = max(0.0, min(1.0, 1.0 - (ci_width / current_price)))

        signal = "HOLD"
        if confidence_score > 0.4:
            if expected_return > 0 and lower_first > current_price:
                signal = "BUY"
            elif expected_return < 0 and upper_first < current_price:
                signal = "SELL"

        # ---------------------------
        # Anomaly Detection
        # ---------------------------
        predictions, anomalies, mae, rmse = detect_anomalies(df, threshold)

        anomaly_points = [
            float(df["value"].iloc[i]) if anomalies[i] else None
            for i in range(len(anomalies))
        ]

        # ---------------------------
        # Final Response
        # ---------------------------
        return jsonify({
            "actual":      list(map(float, df["value"])),
            "predicted":   list(map(float, predictions)),
            "forecast":    forecast,
            "lower_ci":    lower,
            "upper_ci":    upper,
            "selected_model": selected_model,
            "aic":            round(safe(aic), 3),
            "residual_mean":  round(safe(residual_mean), 3),
            "ljung_box_pvalue": round(safe(lb_pvalue), 3),

            "signal":           signal,
            "market_regime":    regime,
            "expected_return":  round(safe(expected_return * 100), 3),
            "confidence_score": round(safe(confidence_score), 3),

            "anomaly_points": [float(x) if x is not None else None for x in anomaly_points],
            "anomaly_count":  int(sum(anomalies)),
            "mae":  round(safe(mae),  3),
            "rmse": round(safe(rmse), 3),

            "backtest_rmse":               round(safe(bt_rmse), 3) if bt_rmse is not None else None,
            "backtest_mape":               round(safe(bt_mape), 3) if bt_mape is not None else None,
            "backtest_direction_accuracy": round(safe(bt_dir),  3) if bt_dir  is not None else None,

            "strategy_total_return":      round(safe(strategy_total_return), 3),
            "strategy_annualized_return": round(safe(strategy_annualized_return), 3),
            "strategy_win_rate":          round(safe(strategy_win_rate), 3),
            "strategy_sharpe_ratio":      round(safe(strategy_sharpe), 3),
            "strategy_max_drawdown":      round(safe(strategy_max_dd), 3),
            "buy_hold_return":            round(safe(buy_hold_return), 3),
            "equity_curve":               [safe(x) for x in equity.tolist()],

            "company_name": company_name,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
