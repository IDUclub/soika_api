"""
Constants module provides constants in dictionary format for routing input api values.
"""

from app.common.storage.implementations.disposable_gdf import DisposableCityGDF
from app.common.storage.implementations.disposable_json import DisposableJSON
from app.common.config import config

bucket_name = config.get("FILESERVER_BUCKET_NAME")
object_name = config.get("FILESERVER_CITY_NAME")
text_name = config.get("FILESERVER_TEXT_NAME")
blocks_name = config.get("FILESERVER_BLOCKS_NAME")

TEXTS = DisposableCityGDF()
TEXTS.try_init(bucket_name, text_name)
BLOCKS = DisposableCityGDF()
BLOCKS.try_init(bucket_name, blocks_name)

CONSTANTS = DisposableJSON()
CONSTANTS.try_init(bucket_name, constants_name)
