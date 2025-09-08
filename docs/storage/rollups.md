# Roll-ups

The real QuantX project exposes a daemon that continuously reads ticks from a
`ColumnStore` and aggregates them into time based bars.  The implementation in
this kata is drastically simplified but mirrors the public API so examples can
be run end-to-end.

The behaviour is encapsulated in :class:`quantx.storage.rollup.OnlineRollup`.
It consumes ticks from the ``tail`` of a :class:`~quantx.storage.rollup.ColumnStore`
and writes completed one minute bars back to a shard named ``"<topic>/1m"``.

## Command line interface

The minimal CLI exposed by this kata provides a ``rollup`` subcommand.  The
following command consumes ticks from the ``ticks`` shard and publishes bars to
``bars/1m``:

```bash
qx rollup run --stream ticks --out-topic bars
```

The ``qx`` command is intentionally lightweight; in a production environment it
would accept further configuration options and the ``ColumnStore`` would be
connected to a persistent backend.
