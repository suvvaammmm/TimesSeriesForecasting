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
        return jsonify({"error": str(e)}), 500


# ======================================================
# SINGLE ASSET
# ======================================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data_source = request.form.get("data_source")
        model_type = request.form.get("model_type")
        threshold = float(request.form.get("threshold", 1.5))

        # ---------------------------
        # Load Data
        # ---------------------------
        if data_source == "file":
            if "file" not in request.files:
                return jsonify({"error": "CSV file required"}), 400
            file = request.files["file"]
            df = pd.read_csv(file)
            if "value" not in df.columns:
                return jsonify({"error": "CSV must contain 'value' column"}), 400
            series = pd.to_numeric(df["value"], errors="coerce").dropna()
            series = series.reset_index(drop=True)
        elif data_source == "angel":
            from services.angel_service import get_angel_data
            symbol_token = request.form.get("symbol_token")
            if not symbol_token:
                return jsonify({"error": "Symbol token required"}), 400
            series,company_name = get_angel_data(symbol_token)
            if series is None or len(series)==0:
                return jsonify({"error" : "No data returned from Angel API" }), 400
            series = pd.Series(series).reset_index(drop=True)
        else:
            return jsonify({"error" : "Invalid data source"}), 400
        df = pd.DataFrame({"value": series})
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
            forecast = best_data["forecast"]
            lower = best_data["lower"]
            upper = best_data["upper"]
            aic = best_data["aic"]
            residual_mean = best_data["residual_mean"]
            lb_pvalue = best_data["lb_pvalue"]
            selected_model = best_model

        else:
            return jsonify({"error": "Invalid model type"}), 400

        # Ensure numpy-safe lists
        forecast = list(np.array(forecast).flatten())
        lower = list(np.array(lower).flatten())
        upper = list(np.array(upper).flatten())

        # ---------------------------
        # Backtest
        # ---------------------------
        bt_rmse = None
        bt_mape = None
        bt_dir = None
        rolling_preds = None

        # ---------------------------
        # Strategy Simulation
        # ---------------------------
        split = int(len(series) * 0.8)
        starting_capital = 100000
        capital = starting_capital
        transaction_cost = 0.001
        risk_per_trade = 0.02

        equity = [capital]
        returns = []

        if rolling_preds is not None:
            max_index = min(len(rolling_preds), len(series) - split - 1)

            for i in range(max_index):

                current_price = series.iloc[split + i]
                next_price = series.iloc[split + i + 1]
                predicted_price = rolling_preds[i]

                if predicted_price > current_price:
                    r = ((next_price - current_price) / current_price) - transaction_cost
                elif predicted_price < current_price:
                    r = ((current_price - next_price) / current_price) - transaction_cost
                else:
                    r = 0

                position_size = capital * risk_per_trade
                capital += position_size * r

                equity.append(capital)
                returns.append(r)

        equity = np.array(equity)
        returns = np.array(returns)

        strategy_total_return = ((capital - starting_capital) / starting_capital) * 100
        num_days = len(series)
        if num_days > 0:
            strategy_annualized_return = ((capital / starting_capital) ** (252 / num_days) - 1) * 100
        else:
            strategy_annualized_return=0
        if len(returns) > 0:
            wins=np.sum(returns > 0)
            strategy_win_rate = (wins / len(returns)) * 100
        else:
            strategy_win_rate = 0

        strategy_sharpe = 0
        if len(returns) > 1 and np.std(returns) != 0:
            strategy_sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)

        strategy_max_dd = (
            np.min(equity / np.maximum.accumulate(equity) - 1) * 100
            if len(equity) > 1 else 0
        )

        buy_hold_return = (
            (series.iloc[-1] - series.iloc[split]) / series.iloc[split]
        ) * 100

        # ---------------------------
        # Signal Engine
        # ---------------------------
        current_price = series.iloc[-1]
        forecast_mean = np.mean(forecast[:3])

        lower_first = lower[0]
        upper_first = upper[0]

        expected_return = (forecast_mean - current_price) / current_price

        volatility = series.pct_change().rolling(20).std().iloc[-1]
        regime = "NORMAL"
        if volatility > 0.03:
            regime = "HIGH VOLATILITY"
        elif volatility < 0.01:
            regime = "LOW VOLATILITY"

        ci_width = upper_first - lower_first
        confidence_score = max(0, min(1, 1 - (ci_width / current_price)))

        signal = "HOLD"
        if confidence_score > 0.4:
            if expected_return > 0 and lower_first > current_price:
                signal = "BUY"
            elif expected_return < 0 and upper_first < current_price:
                signal = "SELL"

        # ---------------------------
        # Anomaly
        # ---------------------------
        predictions, anomalies, mae, rmse = detect_anomalies(df, threshold)

        anomaly_points = [
            float(df["value"].iloc[i]) if anomalies[i] else None
            for i in range(len(anomalies))
        ]

        # ---------------------------
        # Final JSON
        # ---------------------------
        return jsonify({
            "actual": list(map(float, df["value"])),
            "predicted": list(map(float, predictions)),
            "forecast": list(map(float, forecast)),
            "lower_ci": list(map(float, lower)),
            "upper_ci": list(map(float, upper)),
            "selected_model": selected_model,
            "aic": round(safe(aic), 3),
            "residual_mean": round(safe(residual_mean), 3),
            "ljung_box_pvalue": round(safe(lb_pvalue), 3),
            "signal": signal,
            "market_regime": regime,
            "expected_return": round(safe(expected_return * 100), 3),
            "confidence_score": round(safe(confidence_score), 3),
            "anomaly_points": [
                float(x) if x is not None else None 
                for x in anomaly_points
            ],
            "anomaly_count": int(sum(anomalies)),
            "mae": round(safe(mae), 3),
            "rmse": round(safe(rmse), 3),
            "backtest_rmse": round(safe(bt_rmse), 3) if bt_rmse is not None else None,
            "backtest_mape": round(safe(bt_mape), 3) if bt_mape is not None else None,
            "backtest_direction_accuracy": round(safe(bt_dir), 3) if bt_dir is not None else None,
            "strategy_total_return": round(strategy_total_return, 3),
            "strategy_annualized_return": round(strategy_annualized_return, 3),
            "strategy_win_rate":round(strategy_win_rate, 3),
            "strategy_sharpe_ratio": round(strategy_sharpe, 3),
            "strategy_max_drawdown": round(strategy_max_dd, 3),
            "company_name" : company_name if data_source == "angel" else "Uploaded CSV",
            "buy_hold_return": round(buy_hold_return, 3),
            "equity_curve": list(map(float, equity))
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)