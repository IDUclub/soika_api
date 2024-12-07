"""
Constants module provides constants in dictionary format for routing input api values.
"""

from app.common.storage.implementations.disposable_gdf import DisposableTextGDF
from app.common.storage.implementations.disposable_json import DisposableJSON
from app.common.config import config

bucket_name = config.get("FILESERVER_BUCKET_NAME")
text_name = config.get("FILESERVER_TEXT_NAME")
constants_name = config.get("FILESERVER_CONSTANTS_NAME")

TEXTS = DisposableTextGDF()
TEXTS.try_init(bucket_name, text_name)

CONSTANTS = DisposableJSON()
CONSTANTS.try_init(bucket_name, constants_name)
