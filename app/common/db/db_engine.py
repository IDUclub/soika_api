from sqlalchemy import URL, NullPool
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from iduconfig import Config
from sqlalchemy.sql import Executable
from loguru import logger
from app.dependencies import config

class DatabaseEngine:
    def __init__(self, config: Config):
        self.config = config
        self.url = URL.create(
            config.get("DB_ENGINE"),
            username=config.get("DB_USERNAME"),
            password=config.get("DB_PASSWORD"),
            host=config.get("DB_HOST"),
            port=config.get("DB_PORT"),
            database=config.get("DB_DATABASE"),
        )
        self.engine = create_async_engine(
            self.url,
            echo=True,
            poolclass=NullPool,
        )
        self.conn = self.engine.connect()
        self.session = sessionmaker(
            self.engine, expire_on_commit=False, class_=AsyncSession
        )

    async def execute_query(self, query: Executable):
        async with self.session() as session:
            try:
                result = await session.execute(query)
                await session.commit()
                return result
            except Exception as e:
                await session.rollback()
                logger.exception("Ошибка выполнения запроса: {}", query)
                raise e
        
database = DatabaseEngine(config)