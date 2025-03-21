import asyncio
from app.common.db.database import (
    Territory,
    Territory
)
from app.common.db.db_engine import database
from sqlalchemy import select, delete
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
    
    async def collect_territories():
        async with database.session() as session:
            result = await session.execute(select(Territory))
            territories = result.scalars().all()
            
        territories_list = await asyncio.to_thread(territories_calculation.build_territories_list, territories)
        return {"territories": territories_list}
    
    async def get_all_territories():
        return await territories_calculation.collect_territories()

    async def create_new_territory(payload):
        try:
            new_territory = await territories_calculation.create_territory(payload)
            return {
                "territory_id": new_territory.territory_id,
                "name": new_territory.name,
                "matched_territory": new_territory.matched_territory,
            }
        except Exception as e:
            raise http_exception(status_code=400, detail=str(e))

    async def delete_all_territories_func():
        async with database.session() as session:
            await session.execute(delete(Territory))
            await session.commit()
        return {"detail": "All territories deleted"}
    
territories_calculation = TerritoriesCalculation()