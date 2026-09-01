"""`python -m emil_ml.watcher` — the folder watcher's process entry point.

A standalone, long-lived process (see service.py's module docstring for
why this must never be started from inside a Streamlit session). Runs
until interrupted (Ctrl-C, or SIGTERM where the OS actually delivers it).
"""

from __future__ import annotations

import argparse
import logging
import signal

from emil_ml.config.logging_config import configure_logging
from emil_ml.config.settings import (
    WATCHER_POLL_INTERVAL_SECONDS,
    WATCHER_STABILITY_CHECK_INTERVAL_SECONDS,
    WATCHER_STABILITY_REQUIRED_CHECKS,
)
from emil_ml.watcher.service import WatcherService

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m emil_ml.watcher",
        description="Watch every registered component's input/ directory and inspect files as they appear.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=WATCHER_POLL_INTERVAL_SECONDS,
        help=f"Seconds between full input/ directory rescans, the safety net for missed "
        f"filesystem events (default: {WATCHER_POLL_INTERVAL_SECONDS}).",
    )
    parser.add_argument(
        "--stability-check-interval",
        type=float,
        default=WATCHER_STABILITY_CHECK_INTERVAL_SECONDS,
        help="Seconds between consecutive file-size checks while waiting for a file to finish "
        f"writing (default: {WATCHER_STABILITY_CHECK_INTERVAL_SECONDS}).",
    )
    parser.add_argument(
        "--stability-checks",
        type=int,
        default=WATCHER_STABILITY_REQUIRED_CHECKS,
        help="Consecutive equal-size checks required before a file is considered fully written "
        f"(default: {WATCHER_STABILITY_REQUIRED_CHECKS}).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    configure_logging(level=args.log_level)

    service = WatcherService(
        poll_interval=args.poll_interval,
        stability_check_interval=args.stability_check_interval,
        stability_required_checks=args.stability_checks,
    )

    def _handle_signal(signum: int, _frame: object) -> None:
        logger.info("Received signal %s, shutting down...", signum)
        service.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    logger.info(
        "Starting folder watcher (poll_interval=%ss, stability=%s checks x %ss apart)",
        args.poll_interval,
        args.stability_checks,
        args.stability_check_interval,
    )
    service.run_forever()
    logger.info("Watcher stopped.")


if __name__ == "__main__":
    main()
