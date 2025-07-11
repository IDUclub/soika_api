import asyncio
from rapidfuzz import fuzz
from sqlalchemy import select


from app.common.db.db_engine import database
from app.common.modules.constants import CONSTANTS
from app.preprocessing.modules import utils
from app.common.db.database import (
    Service,
    MessageService
)

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

    async def extract_services_in_messages(
        self,
        territory_id: int | None = None,
        top: int | None = None,
    ) -> dict:
        """
        1) Берёт сообщения, для которых step 'services_processed' ещё не отмечен
        (при необходимости ограничивает top).
        2) Для каждого сообщения вызывает detect_services(text).
        3) Для каждого найденного сервиса ищет/создаёт запись в Service и связь
        MessageService.
        4) Ставит/создаёт статус services_processed в MessageStatus.
        5) Возвращает статистику созданных связей.
        """
        async with database.session() as session:
            messages = await utils.get_unprocessed_texts(
                session,
                process_type="services_processed",
                top=top,
                territory_id=territory_id,
            )
            if not messages:
                return {
                    "detail": "No unprocessed messages found.",
                    "processed_messages": 0,
                }

            total_links_created = 0

            for msg in messages:
                services_found = await self.detect_services(msg.text)
                if services_found:
                    for serv_name in services_found:
                        service_obj = await session.scalar(
                            select(Service).where(Service.name == serv_name)
                        )
                        if not service_obj:
                            service_obj = Service(name=serv_name)
                            session.add(service_obj)
                            await session.flush()            # нужен id

                        link_exists = await session.scalar(
                            select(MessageService).where(
                                MessageService.message_id == msg.message_id,
                                MessageService.service_id == service_obj.service_id,
                            )
                        )
                        if not link_exists:
                            session.add(
                                MessageService(
                                    message_id=msg.message_id,
                                    service_id=service_obj.service_id,
                                )
                            )
                            total_links_created += 1

                await utils.update_message_status(
                    session=session,
                    message_id=msg.message_id,
                    process_type="services_processed",
                )

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
    async def extract_services_func(territory_id: int = None, top: int = None):
        result = await services_calculation.extract_services_in_messages(territory_id=territory_id, top=top)
        return result
    
services_calculation = ServicesCalculation()
