"""Unified logging configuration for the project."""

import logging
import os
import sys


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"


class ColoredFormatter(logging.Formatter):
    """Logging formatter with color support.

    The level name is coloured only in the string this formatter returns; the
    shared ``LogRecord`` is left untouched so other handlers (e.g. a log file)
    never receive escape codes.
    """

    LEVEL_COLORS = {
        logging.DEBUG: Colors.CYAN,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.RED + Colors.BOLD,
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors (without mutating ``record``)."""
        level_color = self.LEVEL_COLORS.get(record.levelno, Colors.RESET)
        original_levelname = record.levelname
        record.levelname = f"{level_color}{original_levelname:8}{Colors.RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original_levelname


def _stream_supports_color(stream) -> bool:
    """True when ``stream`` is an interactive terminal and NO_COLOR is unset."""
    if os.environ.get("NO_COLOR"):
        return False
    isatty = getattr(stream, "isatty", None)
    try:
        return bool(isatty and isatty())
    except (ValueError, OSError):  # closed / detached stream
        return False


def setup_logging(
    name: str = "ulcer_detection",
    level: int = logging.INFO,
) -> logging.Logger:
    """Setup logging configuration for the project.

    Args:
        name: Logger name.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Console output is coloured only when stderr is a TTY (and ``NO_COLOR`` is unset).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # prevent duplicate output via root logger

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler (colored when attached to a terminal)
    stream = sys.stderr
    console_handler = logging.StreamHandler(stream)
    console_handler.setLevel(level)
    colour = _stream_supports_color(stream)
    if colour:
        console_formatter = ColoredFormatter(
            f"{Colors.BOLD}%(asctime)s{Colors.RESET} | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    else:
        console_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)


    return logger


def get_logger(name: str) -> logging.Logger:
    """Get logger instance with standard name.

    Args:
        name: Logger name (use __name__ in modules).

    Returns:
        Logger instance (should already be configured by setup_logging).
    """
    return logging.getLogger(name)
