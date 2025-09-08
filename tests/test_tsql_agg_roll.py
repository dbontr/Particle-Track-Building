import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from quantx.query.tsql import aggregate, rolling


def test_aggregate_groupby():
    df = pd.DataFrame({
        "symbol": ["A", "A", "B", "B"],
        "window": [1, 1, 1, 1],
        "price": [1.0, 2.0, 3.0, 4.0],
        "volume": [10, 20, 30, 40],
    })
    agg = {
        "price": ["sum", "mean", "ohlc", "last"],
        "volume": ["sum", "last"],
    }
    result = aggregate(df, ["symbol", "window"], agg)
    row_a = result[result["symbol"] == "A"].iloc[0]
    assert row_a["price_sum"] == 3.0
    assert row_a["price_mean"] == 1.5
    assert row_a["price_ohlc_open"] == 1.0
    assert row_a["price_ohlc_high"] == 2.0
    assert row_a["price_ohlc_low"] == 1.0
    assert row_a["price_ohlc_close"] == 2.0
    assert row_a["price_last"] == 2.0
    assert row_a["volume_sum"] == 30
    assert row_a["volume_last"] == 20


def test_rolling_functions():
    df = pd.DataFrame({
        "price": [1, 2, 3, 4, 5],
        "high": [1, 2, 3, 4, 5],
        "low": [0, 1, 2, 3, 4],
        "close": [1, 2, 3, 4, 5],
    })
    sma = rolling(df, 3, "sma", column="price")
    assert np.allclose(sma.tolist(), [np.nan, np.nan, 2.0, 3.0, 4.0], equal_nan=True)

    ema = rolling(df, 3, "ema", column="price")
    expected_ema = df["price"].ewm(span=3, adjust=False).mean()
    assert np.allclose(ema, expected_ema)

    atr = rolling(df, 3, "atr")
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    expected_atr = tr.rolling(3).mean()
    assert np.allclose(atr, expected_atr, equal_nan=True)
