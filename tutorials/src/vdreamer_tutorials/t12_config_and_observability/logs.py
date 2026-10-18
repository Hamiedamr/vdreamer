"""Server-side logging that knows which run, thread and user a line belongs to."""

from __future__ import annotations

import logging

import vdreamer

LOGGER_NAME = "vdreamer_tutorials.t12"


class RunContextFilter(logging.Filter):
    """Adds ``run_id``, ``thread_id`` and ``user_id`` to every record (``-`` outside runs)."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            context = vdreamer.current_context()
        except RuntimeError:  # not inside a vdreamer run
            record.run_id = record.thread_id = record.user_id = "-"
        else:
            record.run_id = context.run_id
            record.thread_id = context.thread_id
            record.user_id = context.user_id or "-"
        return True


def get_logger(level: str) -> logging.Logger:
    """The lesson's logger: key=value lines on stderr, configured once per process."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s level=%(levelname)s logger=%(name)s run=%(run_id)s "
            "thread=%(thread_id)s user=%(user_id)s %(message)s"
        ))
        handler.addFilter(RunContextFilter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger
