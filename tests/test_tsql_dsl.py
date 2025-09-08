import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from quantx.query.tsql import where


def test_where_in_and():
    df = pd.DataFrame({
        "symbol": ["A", "B", "C"],
        "price": [5, 15, 25],
    })
    result = where(df, "symbol in ('A','B') and price > 10")
    assert result.shape[0] == 1
    assert result.iloc[0]["symbol"] == "B"
