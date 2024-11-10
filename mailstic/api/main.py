import os
from typing import Annotated, List

from dotenv import load_dotenv

load_dotenv()

from fastapi import Depends, FastAPI, HTTPException
from fastapi_utilities import repeat_every
from slugify import slugify
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from lib.db.connection import get_db_session
from lib.email.imap import IMAP
from lib.processing.few_shot_inference.new_class_processing import add_new_class
from lib.processing.message_processing import generate_answer, process_message
from lib.models.qa import QA, QACreate
from lib.models.thread import Thread, ThreadStatus
from lib.models.message import Message
from lib.models.pof import POF, POFCreate, POFCreateClassifier

app = FastAPI()


@app.on_event("startup")
@repeat_every(seconds=60)
async def get_email(db: Annotated[AsyncSession, Depends(get_db_session)]):
    print("checking out email")

    imap = IMAP(os.getenv("MAIL_EMAIL"), os.getenv("MAIL_PASSWORD"))

    messages = imap.receive()
    print(f"got {len(messages)} messages")

    for sender, subject, text, img_path in messages:
        query = select(Thread).where(
            Thread.status == ThreadStatus.IN_PROGRESS and Thread.sender == sender
        )

        thread = (await db.execute(query)).scalar()

        if thread is None:
            thread = Thread(
                sender=sender, status=ThreadStatus.IN_PROGRESS, subject=subject
            )

            db.add(thread)
            await db.commit()
            await db.refresh(thread)

        print(thread)

        db.add(Message(text=text, image=img_path, thread_id=thread.id))
        await db.commit()

        query = select(Message).where(Message.thread_id == thread.id)
        messages = list((await db.execute(query)).scalars())

        print(messages)

        dialogue = [
            {
                "mail": {
                    "theme": thread.subject,
                    "text": message.text,
                    "image": message.img_path,
                },
                "output": process_message(
                    thread.subject, message.text, message.img_path
                ),
            }
            for message in messages
        ]

        print(dialogue)

        answer = generate_answer(dialogue)

        print(answer)

        if answer["completed"]:
            thread.status = ThreadStatus.SUCCESSFUL

        thread.device = answer["data"]["device"]
        thread.problem = answer["data"]["problem_type"]
        thread.serial = answer["data"]["serial_number"]
        thread.recogize_progress = answer["code"]

        db.add(thread)
        await db.commit()

    imap.close()


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
) -> QA:
    query = select(POF).where(POF.slug == slug)
    pof = (await db.execute(query)).scalar()

    if pof is None:
        raise HTTPException(status_code=404, detail=f"pof {slug} is not exists")

    qa = QA(question=body.question, answer=body.answer, pof_slug=slug)

    db.add(qa)

    await db.commit()
    await db.refresh(qa)

    return qa


@app.delete("/pof/{slug}")
async def delete_pof(slug: str, db: Annotated[AsyncSession, Depends(get_db_session)]):
    query = select(POF).where(POF.slug == slug)
    pof = (await db.execute(query)).scalar_one()

    print(pof)

    await db.delete(pof)
    await db.commit()


@app.post("/pof/classifier")
async def create_pof_classifier(body: POFCreateClassifier):
    slug = slugify(body.name)
    add_new_class(slug, body.dataset)


@app.post("/pof")
async def create_pof(
    body: POFCreate, db: Annotated[AsyncSession, Depends(get_db_session)]
) -> POF:
    pof = POF(
        name=body.name,
        slug=slugify(body.name),
    )

    db.add(pof)
    await db.commit()
    await db.refresh(pof)

    return pof


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
