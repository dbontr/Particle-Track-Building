"""Online roll-up of tick data into time-based bars.

This module contains a very small in-memory column store used for tests and a
simple :class:`OnlineRollup` class which consumes tick data from a ``ColumnStore``
``tail`` generator and writes completed bars back to a shard.  The
implementation is intentionally lightweight; it is only intended to support the
unit tests in this kata but models the behaviour of the real service closely.

Example
-------
>>> store = ColumnStore()
>>> store.append("ticks", pd.DataFrame([...]))  # doctest: +SKIP
>>> OnlineRollup(store).stream("ticks", "bars")  # doctest: +SKIP
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Optional

import pandas as pd


class ColumnStore:
    """Very small in-memory column store used for tests.

    ``ColumnStore`` keeps shards of data in memory.  Data is stored as
    :class:`pandas.DataFrame` objects.  The :meth:`tail` method yields rows from a
    shard one-by-one which mimics the behaviour of tailing an append-only log.
    """

    def __init__(self) -> None:
        self.shards: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Shard management
    def append(self, shard: str, df: pd.DataFrame) -> None:
        """Append *df* to *shard* creating it if necessary."""
        if shard in self.shards and not self.shards[shard].empty:
            self.shards[shard] = pd.concat([self.shards[shard], df], ignore_index=True)
        else:
            self.shards[shard] = df.reset_index(drop=True)

    def read(self, shard: str) -> pd.DataFrame:
        """Return the shard contents or an empty frame if it doesn't exist."""
        return self.shards.get(shard, pd.DataFrame())

    # ------------------------------------------------------------------
    # Streaming interface
    def tail(self, shard: str) -> Iterator[Dict]:
        """Yield rows from *shard* as dictionaries.

        The generator simply iterates over the stored DataFrame.  In a real
        system this would block waiting for new data, but for the purposes of
        the tests the stored rows are sufficient.
        """
        df = self.shards.get(shard)
        if df is None:
            return iter(())
        for _, row in df.iterrows():
            yield row.to_dict()


@dataclass
class OnlineRollup:
    """Stream ticks from a ``ColumnStore`` and write completed bars back."""

    store: ColumnStore

    def _write_bar(self, bar_topic: str, bar: Dict) -> None:
        df = pd.DataFrame([bar])
        shard = f"{bar_topic}/1m"
        self.store.append(shard, df)

    # ------------------------------------------------------------------
    def stream(self, tick_topic: str, out_topic: str) -> None:
        """Consume ticks from *tick_topic* and emit one minute bars.

        Parameters
        ----------
        tick_topic : str
            Name of the shard containing tick data.
        out_topic : str
            Name of the topic to write bars to.  The resulting shard name is
            ``"{out_topic}/1m"``.
        """

        cur_min: Optional[pd.Timestamp] = None
        bar: Optional[Dict] = None

        for tick in self.store.tail(tick_topic):
            ts = pd.to_datetime(tick["timestamp"])
            minute = ts.floor("T")
            price = tick["price"]
            size = tick.get("size", 0)

            if cur_min is None:
                cur_min = minute
                bar = {
                    "timestamp": cur_min,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": size,
                }
                continue

            if minute != cur_min:
                assert bar is not None
                self._write_bar(out_topic, bar)
                cur_min = minute
                bar = {
                    "timestamp": cur_min,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": size,
                }
            else:
                # update current bar
                assert bar is not None
                bar["high"] = max(bar["high"], price)
                bar["low"] = min(bar["low"], price)
                bar["close"] = price
                bar["volume"] += size

        # flush last bar
        if bar is not None:
            self._write_bar(out_topic, bar)
