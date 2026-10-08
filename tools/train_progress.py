#!/usr/bin/env python3
"""Render a compact training progress line from the IterSpeed callback output."""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import deque


ITER_SPEED_RE = re.compile(
    r"\b(?P<iteration>\d+)\s*:\s*iter_speed\s+"
    r"(?P<seconds>[0-9]+(?:\.[0-9]+)?)\s+seconds per iteration\s*\|\s*"
    r"Loss:\s*(?P<loss>[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?)"
)
IMPORTANT_RE = re.compile(
    r"ERROR|Traceback|RuntimeError|CUDA out of memory|OutOfMemory|Exception|"
    r"failed|failure|Saved checkpoint|Checkpoint save completed|Done \(exit",
    re.IGNORECASE,
)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def progress_line(
    iteration: int,
    total: int,
    average_seconds: float,
    elapsed: float,
    losses: deque[float],
) -> str:
    if total > 0:
        fraction = min(max(iteration / total, 0.0), 1.0)
        filled = int(round(fraction * 28))
        bar = "#" * filled + "." * (28 - filled)
        remaining = max(total - iteration, 0) * average_seconds
        progress = f"[{bar}] {iteration}/{total} ({fraction * 100:5.1f}%)"
        eta = f"ETA {format_duration(remaining)}"
    else:
        progress = f"iteration {iteration}"
        eta = "ETA unknown"

    recent = ",".join(f"{loss:.4f}" for loss in losses)
    return (
        f"{progress} | iter {average_seconds:5.2f}s | "
        f"elapsed {format_duration(elapsed)} | {eta} | "
        f"loss[{len(losses)}] {recent}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total", type=int, default=0, help="total optimizer iterations")
    parser.add_argument("--recent-losses", type=int, default=5)
    args = parser.parse_args()

    recent_speeds: deque[float] = deque(maxlen=5)
    recent_losses: deque[float] = deque(maxlen=max(1, args.recent_losses))
    start_time = time.monotonic()
    saw_progress = False
    is_tty = sys.stdout.isatty()

    for raw_line in sys.stdin:
        line = raw_line.rstrip("\n")
        match = ITER_SPEED_RE.search(line)
        if match is not None:
            iteration = int(match.group("iteration"))
            seconds = float(match.group("seconds"))
            loss = float(match.group("loss"))
            saw_progress = True
            recent_speeds.append(seconds)
            recent_losses.append(loss)
            average_seconds = sum(recent_speeds) / len(recent_speeds)
            elapsed = time.monotonic() - start_time
            rendered = progress_line(
                iteration,
                args.total,
                average_seconds,
                elapsed,
                recent_losses,
            )
            if is_tty:
                sys.stdout.write("\033[2K\r" + rendered)
            else:
                sys.stdout.write(rendered + "\n")
            sys.stdout.flush()
            continue

        if IMPORTANT_RE.search(line):
            if is_tty:
                sys.stdout.write("\033[2K\r" + line + "\n")
            else:
                sys.stdout.write(line + "\n")
            sys.stdout.flush()

    if is_tty and saw_progress:
        sys.stdout.write("\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
