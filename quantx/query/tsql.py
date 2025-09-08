import re
from typing import Dict, Iterable, Optional, List

import pandas as pd


def _transform_where(clause: str) -> str:
    """Transform DSL where clause to pandas-compatible query expression."""
    if not clause:
        return ''

    # Replace IN (...) with IN [...] so pandas.query understands it
    def repl(match: re.Match) -> str:
        items = match.group(1)
        return f"in [{items}]"

    clause = re.sub(r"in\s*\(([^)]+)\)", repl, clause, flags=re.IGNORECASE)
    return clause


def where(df: pd.DataFrame, clause: Optional[str]) -> pd.DataFrame:
    """Filter *df* using a small SQL-like DSL.

    Parameters
    ----------
    df : DataFrame
        Input data.
    clause : str
        Where clause supporting ``in`` lists and boolean operators ``and``/``or``.

    Returns
    -------
    DataFrame
        Filtered data frame.
    """
    if not clause:
        return df
    expr = _transform_where(clause)
    return df.query(expr, engine="python")


def _last(series: pd.Series):
    return series.iloc[-1]


_last.__name__ = "last"


_AGG_MAP = {
    "sum": "sum",
    "mean": "mean",
    "last": _last,
}


def aggregate(
    df: pd.DataFrame, groupby_cols: Iterable[str], agg_spec: Dict[str, Iterable[str]]
) -> pd.DataFrame:
    """Group *df* by *groupby_cols* and apply aggregations.

    ``agg_spec`` maps column names to a list of aggregations. Supported names
    are ``sum``, ``mean``, ``ohlc`` and ``last``.
    """
    grouped = df.groupby(list(groupby_cols))
    pieces: List[pd.DataFrame] = []

    for col, funcs in agg_spec.items():
        col_group = grouped[col]
        scalars = [f for f in funcs if f != "ohlc"]
        frames = []
        if scalars:
            res = col_group.agg([_AGG_MAP[f] for f in scalars])
            res.columns = [f"{col}_{c}" for c in res.columns]
            frames.append(res)
        if "ohlc" in funcs:
            ohlc = col_group.ohlc()
            ohlc.columns = [f"{col}_ohlc_{c}" for c in ohlc.columns]
            frames.append(ohlc)
        pieces.append(pd.concat(frames, axis=1))

    result = pd.concat(pieces, axis=1)
    return result.reset_index()


def rolling(df: pd.DataFrame, window: int, method: str, column: Optional[str] = None) -> pd.Series:
    """Apply rolling window calculations.

    Parameters
    ----------
    df : DataFrame
    window : int
        Window size.
    method : {"sma", "ema", "atr"}
    column : str, optional
        Column to operate on for ``sma``/``ema``. ``atr`` expects columns
        ``high``, ``low`` and ``close`` in *df*.
    """
    method = method.lower()
    if method == "sma":
        if column is None:
            raise ValueError("column is required for SMA")
        return df[column].rolling(window).mean()
    if method == "ema":
        if column is None:
            raise ValueError("column is required for EMA")
        return df[column].ewm(span=window, adjust=False).mean()
    if method == "atr":
        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(window).mean()
    raise ValueError(f"Unknown rolling method: {method}")


__all__ = ["where", "aggregate", "rolling"]
