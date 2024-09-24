import pandas as pd
import geopandas as gpd
from sloyka import TextClassifiers

from app.common.config import config
from app.common.exceptions import http_exception

text_cassifier = TextClassifiers()
