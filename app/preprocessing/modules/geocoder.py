import asyncio
import requests
from tqdm.asyncio import tqdm
from typing import List, Dict, Any

from shapely.geometry import Point
from flair.data import Sentence
from pymorphy3 import MorphAnalyzer
from loguru import logger

from iduconfig import Config
from app.common.db.db_engine import database
from app.dependencies import config
from app.preprocessing.modules import utils
from app.common.api.urbandb_api_gateway import urban_db_api
from app.preprocessing.modules.models import models_initialization


class Geocoder:
    """Геокодер входящих текстовых сообщений через Photon с NER‑парсингом."""

    def __init__(self, config: Config):
        self.config = config
        self.PHOTON_URL = "https://photon.komoot.io/api"
        self._rate_limit = asyncio.Semaphore(1)
        self.morph = MorphAnalyzer()

    def normalize_street_name(self, raw: str) -> str:
        """Лемматизирует каждое слово улицы в именительный падеж."""
        tokens = raw.split()
        lemmas = []
        for tok in tokens:
            p = self.morph.parse(tok)[0]
            nom = p.inflect({"nomn"})
            lemmas.append(nom.word if nom else p.normal_form)
        return " ".join(lemmas)

    @staticmethod
    def extract_address_texts(text: str) -> List[str]:
        tagger = models_initialization._ner_model
        sent = Sentence(text)
        tagger.predict(sent)
        spans = sent.get_spans("ner")
        return [span.text for span in spans][:2]

    @staticmethod
    def parse_address_components(texts: List[str]) -> Dict[str, str]:
        result = {"street_name": "", "house_number": ""}
        for elem in texts:
            if result["street_name"] and result["house_number"]:
                break
            tokens = elem.split()
            str_tokens: list[str] = []
            for token in tokens:
                if token.isdigit():
                    if not result["house_number"]:
                        result["house_number"] = token
                else:
                    str_tokens.append(token)
            if str_tokens and not result["street_name"]:
                result["street_name"] = " ".join(str_tokens)
        return result

    async def fetch_photon(self, query: str, bbox: str, layers: List[str], limit: int) -> List[Dict[str, Any]]:
        async with self._rate_limit:
            def _request():
                params = {"q": query, "bbox": bbox, "layer": layers, "limit": limit}
                resp = requests.get(self.PHOTON_URL, params=params)
                resp.raise_for_status()
                return resp.json().get("features", [])

            features = await asyncio.to_thread(_request)
            await asyncio.sleep(1)
            return features

    async def process_text(self, text: str, bbox: str) -> Dict[str, Any]:
        texts = self.extract_address_texts(text)
        addr = self.parse_address_components(texts)
        street = addr.get("street_name", "").strip()
        house = addr.get("house_number", "").strip()

        if not street:
            return {"geometry": None, "location": None, "osm_id": None}

        layers = ["house"] if house else ["street"]
        query = f"{house}, {street}" if house else street

        feats = await self.fetch_photon(query, bbox, layers, limit=1)

        if not feats:
            normalized = self.normalize_street_name(street)
            if normalized.lower() != street.lower():
                query = f"{house}, {normalized}" if house else normalized
                feats = await self.fetch_photon(query, bbox, layers, limit=1)

        if not feats:
            return {"geometry": None, "location": None, "osm_id": None}

        feat = feats[0]
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})

        point = None
        if geom.get("type") == "Point":
            lon, lat = geom["coordinates"]
            point = Point(lon, lat)

        name = props.get("name") or ", ".join(filter(None, [props.get("street", "").strip(), props.get("housenumber", "").strip()]))
        osm_id = props.get("osm_id")

        return {"geometry": point, "location": name or None, "osm_id": osm_id}

    async def get_territory_bbox(self, input_territory_name: str, token) -> str:
        territory_id = await urban_db_api.get_territory_by_name(input_territory_name, token)
        territory = await urban_db_api.get_territory(territory_id)
        minx, miny, maxx, maxy = territory.total_bounds
        return f"{minx},{miny},{maxx},{maxy}"

    async def extract_addresses_from_texts(self, input_territory_name: str, token, territory_id: int | None = None, top: int | None = None) -> dict:
        async with database.session() as session:
            messages = await utils.get_unprocessed_texts(
                session,
                process_type="geolocation_processed",
                top=top,
                territory_id=territory_id)

            if not messages:
                logger.info("No unprocessed messages found for address extraction.")
                return {"status": "No messages to process."}

            bbox = await self.get_territory_bbox(input_territory_name, token)

            updated = 0
            for msg in tqdm(messages, total=len(messages), desc="Geocoding"):
                text = (msg.text or "").strip()

                if not text:
                    msg.geometry = None
                    msg.location = None
                    msg.osm_id = None
                    msg.is_processed = True
                    continue

                res = await self.process_text(text, bbox)

                if isinstance(res["geometry"], Point):
                    msg.geometry = utils.to_ewkt(res["geometry"], srid=4326)
                    updated += 1
                else:
                    msg.geometry = None

                msg.location = res["location"]
                msg.osm_id = res["osm_id"]
                await utils.update_message_status(
                    session,
                    message_id=msg.message_id,
                    process_type="geolocation_processed"
                )

            await session.commit()
            logger.info(f"Batch extraction done. Updated {updated} messages (geometry set).")
            return {"status": f"Extraction of addresses completed. Updated {updated} messages."}


geocoder = Geocoder(config)
