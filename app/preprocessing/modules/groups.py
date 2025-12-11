import asyncio
import requests
import pandas as pd
from sqlalchemy import select, delete

from iduconfig import Config
from app.common.db.database import Territory, Group
from app.common.db.db_engine import database
from app.common.exceptions.http_exception_wrapper import http_exception
from app.dependencies import config


class GroupsCalculation:
    def __init__(self, config: Config):
        self.config = config

    @staticmethod
    def process_vk_groups_df(data, territory_name):
        df = pd.DataFrame(data["response"]["items"])[["id", "name", "screen_name"]]
        df.rename(
            columns={"screen_name": "group_domain", "id": "group_id"}, inplace=True
        )
        df["matched_territory"] = territory_name
        return df.to_dict("records")

    @staticmethod
    async def get_groups_by_territory_id(territory_id: int):
        async with database.session() as session:
            territory = await session.get(Territory, territory_id)
            if not territory:
                raise http_exception(
                    404,
                    f"Territory with id={territory_id} not found",
                    territory_id,
                    None,
                )
            territory_name = territory.name
            query = select(Group).where(Group.matched_territory == territory_name)
            result = await session.execute(query)
            groups = result.scalars().all()

        return groups

    async def search_vk_groups(
        self, territory_id: int, sort: int = 4, count: int = 20, version: str = "5.131"
    ) -> str:
        async with database.session() as session:
            territory = await session.get(Territory, territory_id)
            if not territory:
                raise http_exception(
                    404,
                    f"Territory with id={territory_id} not found",
                    territory_id,
                    None,
                )
            territory_name = territory.name

        group_access_key = config.get("VK_GROUP_ACCESS_KEY")
        params = {
            "q": territory_name,
            "sort": sort,
            "count": count,
            "access_token": group_access_key,
            "v": version,
        }

        response = await asyncio.to_thread(
            requests.get, "https://api.vk.com/method/groups.search", params=params
        )
        data = response.json()
        records = await asyncio.to_thread(
            groups_calculation.process_vk_groups_df, data, territory_name
        )

        async with database.session() as session:
            group_ids = [int(r["group_id"]) for r in records]

            existing_ids: set[int] = set()
            if group_ids:
                result = await session.execute(
                    select(Group.group_id).where(Group.group_id.in_(group_ids))
                )
                existing_ids = set(result.scalars().all())
            for record in records:
                gid = int(record["group_id"])
                if gid in existing_ids:
                    continue

                group_obj = Group(
                    group_id=gid,
                    name=record["name"],
                    group_domain=record["group_domain"],
                    matched_territory=record["matched_territory"],
                )
                session.add(group_obj)

            await session.commit()

        return territory_name

    @staticmethod
    async def get_all_groups():
        async with database.session() as session:
            result = await session.execute(select(Group))
            groups = result.scalars().all()
        groups_list = [
            {
                "group_id": g.group_id,
                "name": g.name,
                "group_domain": g.group_domain,
                "matched_territory": g.matched_territory,
            }
            for g in groups
        ]
        return {"groups": groups_list}

    @staticmethod
    async def collect_vk_groups_func(data):
        result = await groups_calculation.search_vk_groups(data.territory_id)
        return {
            "status": f"VK groups for id {data.territory_id} {result} collected and saved to database"
        }

    @staticmethod
    async def delete_all_groups_func():
        async with database.session() as session:
            await session.execute(delete(Group))
            await session.commit()
        return {"detail": "All groups deleted"}


groups_calculation = GroupsCalculation(config)