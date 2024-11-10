from typing import Annotated, List

from fastapi import Depends, FastAPI, HTTPException
from slugify import slugify
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from lib.db.connection import get_db_session
from lib.models.qa import QA, QACreate
from lib.models.thread import Thread
from lib.models.user import User
from lib.models.pof import POF, POFCreate
from lib.processing.few_shot_inference.new_class_processing import add_new_class

app = FastAPI()


@app.get("/health")
async def health():
    return "ok"


@app.delete("/qa/{id}")
async def delete_qa(id: int, db: Annotated[AsyncSession, Depends(get_db_session)]):
    query = select(QA).where(QA.id == id)
    qa = (await db.execute(query)).scalar_one()

    await db.delete(qa)
    await db.commit()


@app.post("/pof/{slug}/qa")
async def create_qa(
    body: QACreate, slug: str, db: Annotated[AsyncSession, Depends(get_db_session)]
):
    query = select(POF).where(POF.slug == slug)
    pof = (await db.execute(query)).scalar()

    if pof is None:
        raise HTTPException(status_code=404, detail=f"pof {slug} is not exists")

    qa = QA(question=body.question, answer=body.answer, pof_slug=slug)

    db.add(qa)

    await db.commit()


@app.delete("/pof/{slug}")
async def delete_pof(slug: str, db: Annotated[AsyncSession, Depends(get_db_session)]):
    query = select(POF).where(POF.slug == slug)
    pof = (await db.execute(query)).scalar_one()

    print(pof)

    await db.delete(pof)
    await db.commit()


@app.post("/pof")
async def create_pof(
    body: POFCreate, db: Annotated[AsyncSession, Depends(get_db_session)]
):
    pof = POF(
        name=body.name,
        slug=slugify(body.name),
    )

    add_new_class(pof.slug, body.dataset)

    db.add(pof)
    await db.commit()


@app.get("/pof")
async def get_pofs(db: Annotated[AsyncSession, Depends(get_db_session)]) -> List[POF]:
    query = select(POF)

    res = list((await db.execute(query)).scalars())

    return res


@app.get("/pof/{pof_slug}/qa")
async def get_pof_qas(
    pof_slug: str, db: Annotated[AsyncSession, Depends(get_db_session)]
) -> List[QA]:
    query = select(QA).where(QA.pof_slug == pof_slug)
    found = (await db.execute(query)).scalars()

    return list(found)


@app.get("/threads")
async def get_threads(
    db: Annotated[AsyncSession, Depends(get_db_session)]
) -> List[Thread]:
    res = await db.execute(select(Thread))
    entries = res.scalars().all()

    return list(entries)


# просмотр статы
# загрузка статы (csv)

# история входящих писем
