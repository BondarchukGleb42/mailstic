from sqlalchemy.orm import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

MariaDBBase = declarative_base()

async_engine = create_async_engine()
async_engine_factory = async_sessionmaker(async_engine, expire_on_commit=False)


async def get_db_session():
    async with async_engine_factory() as session:
        yield session
