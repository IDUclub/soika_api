import sys
from loguru import logger
from iduconfig import Config

def setup_logger(config: Config, log_level: str = "INFO") -> None:
    logger.remove()
    console_log_format = "<green>{time:MM-DD HH:mm}</green> | <level>{level:<8}</level> | <cyan>{message}</cyan>"
    logger.add(
        sys.stdout,
        format=console_log_format,
        level=log_level,
        colorize=True,
    )
    file_log_format = console_log_format
    log_file = config.get("LOG_FILE")
    logger.add(
        f"{log_file}.log",
        level=log_level,
        format=file_log_format,
        colorize=False,
        backtrace=True,
        diagnose=True,
    )
