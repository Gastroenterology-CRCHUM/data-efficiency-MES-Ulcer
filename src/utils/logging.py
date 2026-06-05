"""Unified logging configuration for the project."""

import logging
import logging.handlers
from pathlib import Path


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
    """Logging formatter with color support."""

    LEVEL_COLORS = {
        logging.DEBUG: Colors.CYAN,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.RED + Colors.BOLD,
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        level_color = self.LEVEL_COLORS.get(record.levelno, Colors.RESET)
        record.levelname = f"{level_color}{record.levelname:8}{Colors.RESET}"
        return super().format(record)


def setup_logging(
    name: str = "ulcer_detection",
    level: int = logging.INFO,
    log_dir: Path | None = None,
    use_color: bool = True,
) -> logging.Logger:
    """Setup logging configuration for the project.

    Args:
        name: Logger name.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir: Directory to write log files. If None, logs go to console only.
        use_color: Whether to use colored output (console only).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # prevent duplicate output via root logger

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler (colored)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = (
        f"{Colors.BOLD}%(asctime)s{Colors.RESET} | %(levelname)-8s | %(name)s | %(message)s"
    )
    if use_color:
        console_formatter = ColoredFormatter(console_format, datefmt="%H:%M:%S")
    else:
        console_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if log_dir provided)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{name}.log"

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s.%(funcName)s:%(lineno)d | %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.info(f"Log file: {log_file}")

    # Suppress verbose libraries
    logging.getLogger("alembic").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get logger instance with standard name.

    Args:
        name: Logger name (use __name__ in modules).

    Returns:
        Logger instance (should already be configured by setup_logging).
    """
    return logging.getLogger(name)
