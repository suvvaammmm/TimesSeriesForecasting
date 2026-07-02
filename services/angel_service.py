import time
import requests
import pyotp
import pandas as pd
from datetime import datetime, timedelta
from SmartApi import SmartConnect
from config import API_KEY, CLIENT_ID, PASSWORD, TOTP_SECRET

# ── Module-level session cache so we don't re-authenticate on every request
_session_cache = {
    "obj":        None,
    "created_at": None,
}
SESSION_TTL_SECONDS = 600   # reuse session for 10 minutes


def _get_session() -> SmartConnect:
    """Return a cached SmartConnect session, or create a fresh one."""
    now = datetime.now()
    cached = _session_cache["obj"]
    created = _session_cache["created_at"]

    if cached and created and (now - created).total_seconds() < SESSION_TTL_SECONDS:
        return cached

    obj  = SmartConnect(api_key=API_KEY)
    totp = pyotp.TOTP(TOTP_SECRET).now()
    obj.generateSession(CLIENT_ID, PASSWORD, totp)

    _session_cache["obj"]        = obj
    _session_cache["created_at"] = now
    return obj


def _get_candle_data_with_retry(obj: SmartConnect, params: dict,
                                 max_retries: int = 4,
                                 base_delay:  float = 2.0) -> dict:
    """
    Call getCandleData with exponential backoff on rate-limit errors.
    Waits 2s → 4s → 8s → 16s between retries.
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            response = obj.getCandleData(params)
            return response

        except Exception as e:
            last_error = e
            err_str    = str(e).lower()

            # Rate-limit → wait and retry
            if "rate" in err_str or "access denied" in err_str or "429" in err_str:
                wait = base_delay * (2 ** attempt)   # 2, 4, 8, 16 seconds
                print(f"[Angel] Rate limited (attempt {attempt + 1}/{max_retries}). "
                      f"Retrying in {wait:.0f}s…")
                time.sleep(wait)
                continue

            # Session expired → refresh and retry once
            if "invalid token" in err_str or "unauthori" in err_str or "session" in err_str:
                print("[Angel] Session expired, refreshing…")
                _session_cache["obj"]        = None
                _session_cache["created_at"] = None
                obj = _get_session()
                continue

            # Any other error → fail immediately
            raise

    raise last_error


def get_angel_data(symbol_token: str):
    """
    Fetch the last year of daily close prices for symbol_token from Angel One.
    Returns (pd.Series of closes, company_name string).
    """
    obj = _get_session()

    # ── Company name lookup (with its own rate-limit guard)
    company_name = f"Token:{symbol_token}"
    try:
        url         = ("https://margincalculator.angelbroking.com/"
                       "OpenAPI_File/files/OpenAPIScripMaster.json")
        instruments = requests.get(url, timeout=10).json()
        for stock in instruments:
            if str(stock.get("token")) == str(symbol_token):
                company_name = stock.get("symbol", company_name)
                break
    except Exception as e:
        print(f"[Angel] Could not fetch instrument master: {e}")

    # ── Candle data params
    to_date   = datetime.now()
    from_date = to_date - timedelta(days=365)

    params = {
        "exchange":    "NSE",
        "symboltoken": symbol_token,
        "interval":    "ONE_DAY",
        "fromdate":    from_date.strftime("%Y-%m-%d %H:%M"),
        "todate":      to_date.strftime("%Y-%m-%d %H:%M"),
    }

    # Small initial delay to avoid back-to-back rate limits
    time.sleep(0.5)

    response = _get_candle_data_with_retry(obj, params)

    if not response or "data" not in response or not response["data"]:
        raise ValueError(f"No candle data returned for token {symbol_token}")

    df = pd.DataFrame(
        response["data"],
        columns=["Datetime", "Open", "High", "Low", "Close", "Volume"],
    )
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df.set_index("Datetime", inplace=True)

    series = df["Close"].dropna()

    if len(series) < 15:
        raise ValueError(f"Too few data points ({len(series)}) for token {symbol_token}")

    return series, company_name