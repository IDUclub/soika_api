from iduconfig import Config
from app.common.modules.logger_setup import setup_logger

config = Config()
setup_logger(config, log_level="INFO")

def get_config_value(key: str, default: str | None = None) -> str | None:
    try:
        return config.get(key)
    except ValueError:
        return default
