import logging
import os

PROJECT_LOGGER_NAME = "SleepVST"
_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def setup_logging(log_dir=None, file_name=PROJECT_LOGGER_NAME, level=logging.INFO):
    """Configure the shared project logger. Call once from an entry point.

    Attaches a stream handler and, when ``log_dir`` is given, a file handler at
    ``log_dir/file_name``. Idempotent: handlers are only added on the first call.
    Returns the configured logger.
    """
    logger = logging.getLogger(PROJECT_LOGGER_NAME)
    logger.setLevel(level)
    # Keep project logs off the root logger to avoid duplicates / third-party noise.
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(_LOG_FORMAT)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, file_name))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name=None):
    """Return the shared project logger, or a named child of it.

    Use at module level: ``logger = get_logger(__name__)``. Children propagate to
    the project logger configured by :func:`setup_logging`, so modules never need
    to receive a logger as an argument.
    """
    if not name or name == PROJECT_LOGGER_NAME:
        return logging.getLogger(PROJECT_LOGGER_NAME)
    return logging.getLogger(f"{PROJECT_LOGGER_NAME}.{name}")
