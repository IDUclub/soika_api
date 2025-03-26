import asyncio
from sqlalchemy import select
from app.common.db.database import (
    Message,
    Service,
    MessageService
)
from app.common.db.db_engine import database
from app.common.modules.constants import CONSTANTS
from rapidfuzz import fuzz

class ServicesCalculation:
    @staticmethod
    def fuzzy_search(text, phrase, threshold=80):
        """
        Ищет фразу в тексте с использованием нестрогого сравнения.
        Возвращает True, если найден фрагмент текста, схожий с целевым по уровню similarity >= threshold.
        """
        text = text.lower()
        phrase = phrase.lower()
        words = text.split()
        phrase_words = phrase.split()
        n = len(phrase_words)
        if n == 0:
            return False
        for i in range(len(words) - n + 1):
            window = " ".join(words[i : i + n])
            if fuzz.ratio(phrase, window) >= threshold:
                return True
        return False

    @staticmethod
    def replace_service_names(services_list, ru_service_names):
        inverted = {eng: rus for rus, eng in ru_service_names.items()}
        return [inverted.get(service, service) for service in services_list]

    async def detect_services(self, text: str) -> list[str]:
        """
        Возвращает список сервисов (названий ключей) для одного текста.
        """
        detected_services = []
        text_lower = text.lower()
        service_keywords = CONSTANTS.json["service_keywords"]
        service_irrelevant_mentions = CONSTANTS.json["service_irrelevant_mentions"]
        services_priority_and_exact_keywords = CONSTANTS.json["services_priority_and_exact_keywords"]
        ru_service_names = CONSTANTS.json["ru_service_names"]

        for service, keywords in service_keywords.items():
            found = False
            for kw in keywords:
                if await asyncio.to_thread(self.fuzzy_search, text_lower, kw):
                    found = True
                    break
            if not found:
                continue

            for irr in service_irrelevant_mentions.get(service, []):
                if await asyncio.to_thread(self.fuzzy_search, text_lower, irr):
                    found = False
                    break
            if not found:
                continue

            if service in services_priority_and_exact_keywords:
                local_config = services_priority_and_exact_keywords[service]
                exact_found = False
                for ex_kw in local_config.get("exact_keywords", []):
                    if await asyncio.to_thread(self.fuzzy_search, text_lower, ex_kw, 85):
                        exact_found = True
                        break
                if exact_found:
                    detected_services.append(service)
                    continue

                priority_over_found = False
                for po_kw in local_config.get("priority_over", []):
                    if await asyncio.to_thread(self.fuzzy_search, text_lower, po_kw, 85):
                        priority_over_found = True
                        break
                if priority_over_found:
                    continue

                for excl in local_config.get("exclude_verbs", []):
                    if await asyncio.to_thread(self.fuzzy_search, text_lower, excl, 85):
                        found = False
                        break
                if not found:
                    continue

            detected_services.append(service)

        detected_services = self.replace_service_names(detected_services, ru_service_names)
        return detected_services

    async def extract_services_in_messages(self, top: int = None) -> dict:
        """
        1) Находит все сообщения is_processed=False (огранич. top, если нужно).
        2) Для каждого сообщения вызывает detect_services(text).
        3) По каждому найденному сервису ищет или создаёт запись в Service,
           затем создаёт запись в MessageService.
        4) Возвращает инфо о количестве созданных связей.
        """
        async with database.session() as session:
            query = select(Message).where(Message.is_processed == False)
            if top is not None and top > 0:
                query = query.limit(top)

            result = await session.execute(query)
            messages = result.scalars().all()

            if not messages:
                return {
                    "detail": "No unprocessed messages found.",
                    "processed_messages": 0,
                }

            total_links_created = 0

            for msg in messages:
                services_found = await self.detect_services(msg.text)
                if not services_found:
                    continue
                for serv_name in services_found:
                    stmt = select(Service).where(Service.name == serv_name)
                    existing = await session.execute(stmt)
                    service_obj = existing.scalar_one_or_none()

                    if not service_obj:
                        service_obj = Service(name=serv_name)
                        session.add(service_obj)
                        await session.flush()

                    link_stmt = select(MessageService).where(
                        MessageService.message_id == msg.message_id,
                        MessageService.service_id == service_obj.service_id,
                    )
                    link_res = await session.execute(link_stmt)
                    link_exists = link_res.scalar_one_or_none()

                    if not link_exists:
                        link = MessageService(
                            message_id=msg.message_id, service_id=service_obj.service_id
                        )
                        session.add(link)
                        total_links_created += 1

            await session.commit()

        return {
            "detail": f"Created {total_links_created} message_service records.",
            "processed_messages": len(messages),
        }
    
    @staticmethod
    async def get_all_services():
        async with database.session() as session:
            result = await session.execute(select(Service))
            services = result.scalars().all()
        services_list = [{"service_id": s.service_id, "name": s.name, "value_id": s.value_id} for s in services]
        return {"services": services_list}

    @staticmethod
    async def get_all_message_service_pairs():
        async with database.session() as session:
            result = await session.execute(select(MessageService))
            services = result.scalars().all()
        services_list = [{"message_id": s.message_id, "service_id": s.service_id} for s in services]
        return {"message_service_pairs": services_list}

    @staticmethod
    async def extract_services_func(top: int = None):
        result = await services_calculation.extract_services_in_messages(top=top)
        return result
    
services_calculation = ServicesCalculation()
