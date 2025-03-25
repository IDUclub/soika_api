"""
Constants module provides constants in dictionary format for routing input api values.
"""

from app.common.storage.implementations.disposable_json import DisposableJSON
from app.dependencies import config

bucket_name = config.get("FILESERVER_BUCKET_NAME")
constants_name = config.get("FILESERVER_CONSTANTS_NAME")

CONSTANTS = DisposableJSON()
CONSTANTS.try_init(bucket_name, constants_name)
