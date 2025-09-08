"""Minimal command line interface for QuantX.

Only the bits required by the tests are implemented.  The real project exposes
many more subcommands but for the purposes of the exercises we only support
streaming roll-ups.
"""

from __future__ import annotations

import argparse
from typing import Sequence

from .storage.rollup import ColumnStore, OnlineRollup


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qx")
    sub = parser.add_subparsers(dest="command", required=True)

    rollup = sub.add_parser("rollup")
    roll_sub = rollup.add_subparsers(dest="rollup_cmd", required=True)
    run = roll_sub.add_parser("run")
    run.add_argument("--stream", required=True, help="Tick shard to consume")
    run.add_argument(
        "--out-topic", required=True, help="Topic to publish bars to"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "rollup" and args.rollup_cmd == "run":
        store = ColumnStore()
        roll = OnlineRollup(store)
        roll.stream(args.stream, args.out_topic)


__all__ = ["build_parser", "main"]
