import sys
from pathlib import Path

import pandas as pd

# Ensure project root on path when tests are run from an installed wheel.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantx.storage.rollup import ColumnStore, OnlineRollup


def test_stream_backwrite():
    # two minutes worth of ticks
    ticks = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2021-01-01 00:00:10",
                    "2021-01-01 00:00:50",
                    "2021-01-01 00:01:20",
                    "2021-01-01 00:01:40",
                ]
            ),
            "price": [100, 101, 102, 103],
            "size": [1, 2, 3, 4],
        }
    )

    store = ColumnStore()
    store.append("ticks", ticks)

    roll = OnlineRollup(store)
    roll.stream("ticks", "bars")

    bars = store.read("bars/1m")
    expected = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2021-01-01 00:00:00", "2021-01-01 00:01:00"]
            ),
            "open": [100, 102],
            "high": [101, 103],
            "low": [100, 102],
            "close": [101, 103],
            "volume": [3, 7],
        }
    )

    pd.testing.assert_frame_equal(bars.reset_index(drop=True), expected)
