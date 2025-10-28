import asyncio

from sqlalchemy import select, func, case, distinct

from app.common.db.database import (
    Message,
    MessageStatus,
    Status,
    Group,
    Territory,
)
from app.common.db.db_engine import database
from sqlalchemy import delete
from app.common.exceptions.http_exception_wrapper import http_exception

class TerritoriesCalculation:
    @staticmethod
    def build_territories_list(territories):
            territories_list = []
            for t in territories:
                territories_list.append(
                    {
                        "territory_id": t.territory_id,
                        "name": t.name,
                        "matched_territory": t.matched_territory,
                    }
                )
            return territories_list
    
    @staticmethod
    async def get_territory_processing_stats(scope: str = "all"):

        async with database.session() as session:

            messages_sq = (
                select(Territory.territory_id)
                .select_from(Territory)
                .join(Group, Group.matched_territory == Territory.name)
                .join(Message, Message.group_id == Group.group_id)
                .group_by(Territory.territory_id)
                .having(func.count(distinct(Message.message_id)) > 0)
            )
            unprocessed_sq = (
                select(Territory.territory_id)
                .select_from(Territory)
                .join(Group, Group.matched_territory == Territory.name)
                .join(Message, Message.group_id == Group.group_id)
                .join(MessageStatus, MessageStatus.message_id == Message.message_id)
                .group_by(Territory.territory_id)
                .having(
                    func.sum(
                        case((MessageStatus.process_status.is_(False), 1), else_=0)
                    ) > 0
                )
            )
            total_q = (
                select(
                    Territory.territory_id.label("territory_id"),
                    Territory.name.label("territory_name"),
                    func.count(distinct(Message.message_id)).label("messages_total"),
                )
                .select_from(Territory)
                .join(Group, Group.matched_territory == Territory.name, isouter=True)
                .join(Message, Message.group_id == Group.group_id, isouter=True)
                .group_by(Territory.territory_id, Territory.name)
            )
            if scope == "with_messages":
                total_q = total_q.where(Territory.territory_id.in_(messages_sq))
            elif scope == "with_unprocessed":
                total_q = total_q.where(Territory.territory_id.in_(unprocessed_sq))
            total_rows = (await session.execute(total_q)).all()
            totals = {
                row.territory_id: {
                    "territory_id": row.territory_id,
                    "territory_name": row.territory_name,
                    "messages_total": int(row.messages_total or 0),
                    "statuses": [],
                }
                for row in total_rows
            }
            per_status_q = (
                select(
                    Territory.territory_id.label("territory_id"),
                    Status.process_status_name.label("status_name"),
                    func.sum(
                        case((MessageStatus.process_status.is_(False), 1), else_=0)
                    ).label("unprocessed_count"),
                )
                .select_from(Territory)
                .join(Group, Group.matched_territory == Territory.name)
                .join(Message, Message.group_id == Group.group_id)
                .join(MessageStatus, MessageStatus.message_id == Message.message_id)
                .join(Status, Status.process_status_id == MessageStatus.process_status_id)
                .group_by(Territory.territory_id, Status.process_status_name)
                .order_by(Territory.territory_id, Status.process_status_name)
            )
            if scope == "with_messages":
                per_status_q = per_status_q.where(Territory.territory_id.in_(messages_sq))
            elif scope == "with_unprocessed":
                per_status_q = per_status_q.where(
                    Territory.territory_id.in_(unprocessed_sq)
                )
            per_status_rows = (await session.execute(per_status_q)).all()
        for row in per_status_rows:
            tid = row.territory_id
            if tid not in totals:
                totals[tid] = {
                    "territory_id": tid,
                    "territory_name": None,
                    "messages_total": 0,
                    "statuses": [],
                }
            total = totals[tid]["messages_total"]
            unproc = int(row.unprocessed_count or 0)
            share = (unproc / total) if total else 0.0
            totals[tid]["statuses"].append(
                {"name": row.status_name, "unprocessed": unproc, "share": share}
            )
        territories = sorted(totals.values(), key=lambda x: x["territory_id"])
        return {"territories": territories}
    
    @staticmethod
    async def collect_territories():
        async with database.session() as session:
            result = await session.execute(select(Territory))
            territories = result.scalars().all()
            
        territories_list = await asyncio.to_thread(territories_calculation.build_territories_list, territories)
        return {"territories": territories_list}
    
    @staticmethod
    async def get_all_territories(scope):
        return await territories_calculation.get_territory_processing_stats(scope)
    
    @staticmethod
    async def create_territory(payload: Territory):
         async with database.session() as session:
             new_territory = Territory(
                 territory_id=payload.territory_id,
                 name=payload.name,
                 matched_territory=payload.matched_territory
             )
             session.add(new_territory)
             await session.commit()
 
             return new_territory
         
    @staticmethod
    async def create_new_territory(payload):
        try:
            new_territory = await territories_calculation.create_territory(payload)
            return {
                "territory_id": new_territory.territory_id,
                "name": new_territory.name,
                "matched_territory": new_territory.matched_territory,
            }
        except Exception as e:
            raise http_exception(status_code=400, msg="Error occurred", input_data=None, detail=str(e))

    @staticmethod
    async def delete_all_territories_func():
        async with database.session() as session:
            await session.execute(delete(Territory))
            await session.commit()
        return {"detail": "All territories deleted"}
    
territories_calculation = TerritoriesCalculation()