import os

from dotenv import load_dotenv
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncAttrs


class Base(AsyncAttrs, DeclarativeBase):
    pass


async_engine = create_async_engine(
    f"mariadb+aiomysql://{os.getenv("DB_URI_WITHOUT_PROTO")}", echo=True
)

async_engine_factory = async_sessionmaker(async_engine, expire_on_commit=False)


async def get_db_session():
    async with async_engine_factory() as session:
        yield session
