"""
Command-line progress display similar to MATLAB's textwaitbar.

Usage:
    textwaitbar(i, n, msg, precision=2)
"""
from __future__ import annotations

import math
import sys
from typing import TextIO


_prev_val: float | None = None


def textwaitbar(i: float, n: float, msg: str, precision: int = 2, *, stream: TextIO = sys.stdout) -> None:
    """
    Display a textual progress indicator with decimal precision.

    Args:
        i: Current iteration (1-indexed or 0-indexed, matches MATLAB behavior).
        n: Total iterations.
        msg: Message prefix to print.
        precision: Number of decimal digits to show (default 2 => 0.01%).
        stream: Output stream (defaults to stdout).
    """
    global _prev_val

    if n == 0:
        raise ValueError("Total iterations 'n' must be non-zero.")

    mult = 10**precision
    curr_val = math.floor((i / n) * 100 * mult) / mult

    # Print initial message
    if _prev_val is None or curr_val < _prev_val:
        _prev_val = 0.0
        s_prev = _get_pct_str(_prev_val, precision)
        stream.write(f"{msg}: {s_prev}")
        stream.flush()

    # Print updated percentage if changed
    if curr_val != _prev_val:
        s_prev = _get_pct_str(_prev_val, precision)
        stream.write(_get_backspace_str(len(s_prev)))

        s_curr = _get_pct_str(curr_val, precision)
        stream.write(s_curr)
        stream.flush()

        _prev_val = curr_val

    # Clear persistent variable at 100%
    if curr_val >= 100:
        stream.write(" Done.\n")
        stream.flush()
        _reset()


def _get_pct_str(prct: float, precision: int) -> str:
    fmt = f"%.{precision}f%%  %s"
    return fmt % (prct, _get_dot_str(prct))


def _get_dot_str(prct: float) -> str:
    dots = [" "] * 10
    filled = min(int(math.floor(prct / 10)), 10)
    for idx in range(filled):
        dots[idx] = "."
    return "[" + "".join(dots) + "]"


def _get_backspace_str(length: int) -> str:
    return "\b" * length


def _reset() -> None:
    global _prev_val
    _prev_val = None
