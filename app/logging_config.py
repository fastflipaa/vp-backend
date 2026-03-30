"""Vive Polanco Backend - structlog JSON logging configuration.

Configures structlog with JSON output to both stdout (for Coolify log viewer)
and a persistent log file. All log entries include timestamp, log_level,
and support trace_id binding via contextvars.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog


def configure_logging(
    json_output: bool = True,
    log_file: str = "vp-backend.log",
) -> None:
    """Configure structlog with JSON rendering and dual output (stdout + file).

    Args:
        json_output: True for JSON rendering (production), False for console (dev).
        log_file: Name of the persistent log file.
    """
    # Determine log directory -- /app/logs inside container, fallback to cwd
    try:
        log_dir = Path("/app/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        log_dir = Path(".")

    # Shared processors applied to every log entry
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    # Choose renderer based on environment
    if json_output:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    # Configure stdlib logging FIRST (structlog routes through it)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Stdout handler (Coolify viewer)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)

    # File handler (persistent history)
    file_handler = logging.FileHandler(
        log_dir / log_file,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)

    # Both handlers use structlog's ProcessorFormatter for consistent JSON output
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )
    stdout_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Clear existing handlers and add ours
    root_logger.handlers.clear()
    root_logger.addHandler(stdout_handler)
    root_logger.addHandler(file_handler)

    # Configure structlog to route through stdlib (reaches both handlers)
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.dict_tracebacks,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
