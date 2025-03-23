import sys
from fastapi import FastAPI
from loguru import logger
from iduconfig import Config
from app.risk_calculation.social_risk_controller import calculation_router
from app.preprocessing.preprocessing_controller import (
    territories_router,
    groups_router,
    messages_router,
    named_objects_router,
    indicators_router,
    services_router
)
from app.common.system_controller import system_router
from iduconfig import Config

config = Config()

def setup_logger(log_level: str = "INFO") -> None:
    """
    Настраивает глобальный логгер с использованием loguru.
    Логи выводятся в консоль и записываются в файл.
    """
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

def include_routers(app: FastAPI, config: Config) -> None:
    """
    Регистрирует все роутеры в приложении с префиксом из конфигурации.
    """
    prefix = config.get("FASTAPI_PREFIX")
    
    app.include_router(system_router, prefix=prefix, tags=["System"])
    app.include_router(calculation_router, prefix=prefix, tags=["Analysis"])
    app.include_router(territories_router, prefix=prefix, tags=["Territories"])
    app.include_router(groups_router, prefix=prefix, tags=["Groups"])
    app.include_router(messages_router, prefix=prefix, tags=["Messages"])
    app.include_router(named_objects_router, prefix=prefix, tags=["Named objects"])
    app.include_router(indicators_router, prefix=prefix, tags=["Indicators"])
    app.include_router(services_router, prefix=prefix, tags=["Services"])

