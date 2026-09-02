"""`python -m emil_ml.cascade_stream --component X` — the cascade stream
consumer's process entry point.

A standalone, long-lived process (see service.py's module docstring for why
this must never be started from inside a Streamlit session). Runs until
interrupted (Ctrl-C, or SIGTERM where the OS actually delivers it) or the
component's configured Kafka connection fails unrecoverably.
"""

from __future__ import annotations

import argparse
import logging
import signal

from emil_ml.config.logging_config import configure_logging
from emil_ml.config.settings import CASCADE_STREAM_HEARTBEAT_INTERVAL_SECONDS, CASCADE_STREAM_KAFKA_POLL_TIMEOUT_SECONDS
from emil_ml.cascade_stream.service import CascadeStreamService

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m emil_ml.cascade_stream",
        description="Consume a component's configured Kafka topic and run the object/face cascade "
        "on a throttled sample of its frames, forever.",
    )
    parser.add_argument("--component", required=True, help="Name of a ready coco_detector component.")
    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=CASCADE_STREAM_HEARTBEAT_INTERVAL_SECONDS,
        help=f"Seconds between liveness heartbeats (default: {CASCADE_STREAM_HEARTBEAT_INTERVAL_SECONDS}).",
    )
    parser.add_argument(
        "--poll-timeout",
        type=float,
        default=CASCADE_STREAM_KAFKA_POLL_TIMEOUT_SECONDS,
        help=f"Kafka Consumer.poll() timeout, seconds (default: {CASCADE_STREAM_KAFKA_POLL_TIMEOUT_SECONDS}).",
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

    service = CascadeStreamService(
        args.component,
        heartbeat_interval=args.heartbeat_interval,
        poll_timeout=args.poll_timeout,
    )

    def _handle_signal(signum: int, _frame: object) -> None:
        logger.info("Received signal %s, shutting down...", signum)
        service.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    logger.info("Starting cascade stream for component=%s", args.component)
    service.run_forever()
    logger.info("Cascade stream process exiting.")


if __name__ == "__main__":
    main()
