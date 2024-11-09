from typing import Annotated, List

from pydantic import BaseModel
from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from lib.db.connection import get_db_session
from lib.models.qa import QA, QACreate
from lib.models.thread import Thread
from lib.models.user import User

app = FastAPI()


@app.get("/health")
async def health():
    return "ok"


@app.post("/signup")
async def signup(user: User, db: Annotated[AsyncSession, Depends(get_db_session)]):
    found_query = select(User).where(User.login == user.login)
    found = (await db.execute(found_query)).scalars().first()

    if found is not None:
        raise HTTPException(status_code=400, detail=f"login {user.login} is busy")

    db.add(user)

    await db.commit()


@app.post("/signin")
async def signin(
    user: User, db: Annotated[AsyncSession, Depends(get_db_session)]
) -> bool:
    query = select(User).where(User.login == user.login)
    found = (await db.execute(query)).scalars().first()

    if found is None:
        return False

    return found.password == user.password


@app.get("/qa")
async def get_all_qa(db: Annotated[AsyncSession, Depends(get_db_session)]) -> List[QA]:
    query = select(QA)
    found = (await db.execute(query)).scalars()

    return list(found)


@app.post("/qa")
async def add_qa(body: QACreate, db: Annotated[AsyncSession, Depends(get_db_session)]):
    db.add(body)
    await db.commit()


@app.get("/threads")
async def get_threads(
    db: Annotated[AsyncSession, Depends(get_db_session)]
) -> List[Thread]:
    res = await db.execute(select(Thread))
    entries = res.scalars().all()

    return list(entries)
